#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr float kAbsoluteTolerance = 2.0e-6F;
constexpr float kRelativeTolerance = 2.0e-5F;
constexpr double kRowSumTolerance = 2.0e-5;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value,
                  __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void softmax_fp32(const float* input, float* output, int cols) {
  const int row = blockIdx.x;
  const std::size_t row_offset =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ float warp_values[kWarpsPerBlock];
  __shared__ float shared_row_max;
  __shared__ float shared_row_sum;

  float thread_max = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_max = fmaxf(thread_max, input[row_offset + col]);
  }
  thread_max = warp_reduce_max(thread_max);
  if (lane == 0) {
    warp_values[warp] = thread_max;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max =
        lane < kWarpsPerBlock ? warp_values[lane] : -CUDART_INF_F;
    block_max = warp_reduce_max(block_max);
    if (lane == 0) {
      shared_row_max = block_max;
    }
  }
  __syncthreads();

  float thread_sum = 0.0F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float exponent = expf(input[row_offset + col] - shared_row_max);
    output[row_offset + col] = exponent;
    thread_sum += exponent;
  }
  thread_sum = warp_reduce_sum(thread_sum);
  if (lane == 0) {
    warp_values[warp] = thread_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_values[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      shared_row_sum = block_sum;
    }
  }
  __syncthreads();

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    output[row_offset + col] /= shared_row_sum;
  }
}

class DeviceFloatBuffer {
 public:
  explicit DeviceFloatBuffer(std::size_t element_count) {
    CUDA_CHECK(cudaMalloc(&pointer_, element_count * sizeof(float)));
  }

  ~DeviceFloatBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
  DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;

  float* get() { return pointer_; }

 private:
  float* pointer_ = nullptr;
};

std::size_t element_count_for(int rows, int cols) {
  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t column_count = static_cast<std::size_t>(cols);
  if (row_count > std::numeric_limits<std::size_t>::max() / column_count) {
    throw std::invalid_argument("rows * cols overflows size_t");
  }
  return row_count * column_count;
}

std::vector<float> make_input(std::size_t element_count, std::uint32_t seed) {
  std::vector<float> input(element_count);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-8.0F, 8.0F);
  for (float& value : input) {
    value = distribution(generator);
  }
  return input;
}

void copy_input_to_device(const std::vector<float>& input,
                          DeviceFloatBuffer& device_input) {
  const std::size_t bytes = input.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes,
                        cudaMemcpyHostToDevice));
}

float benchmark_us(const float* input, float* output, int rows, int cols,
                   int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    softmax_fp32<<<rows, kThreads>>>(input, output, cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    softmax_fp32<<<rows, kThreads>>>(input, output, cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms * 1000.0F / static_cast<float>(iterations);
}

void validate(int rows, int cols, std::uint32_t seed) {
  const std::size_t element_count = element_count_for(rows, cols);
  const std::vector<float> input = make_input(element_count, seed);
  const std::size_t bytes = element_count * sizeof(float);
  DeviceFloatBuffer device_input(element_count);
  DeviceFloatBuffer device_output(element_count);
  copy_input_to_device(input, device_input);

  softmax_fp32<<<rows, kThreads>>>(device_input.get(), device_output.get(),
                                  cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(element_count);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.get(), bytes,
                        cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  int row_sum_violations = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  double max_row_sum_error = 0.0;
  for (int row = 0; row < rows; ++row) {
    const std::size_t row_offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    double row_max = -std::numeric_limits<double>::infinity();
    for (int col = 0; col < cols; ++col) {
      row_max = std::max(row_max,
                         static_cast<double>(input[row_offset + col]));
    }

    double row_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      row_sum +=
          std::exp(static_cast<double>(input[row_offset + col]) - row_max);
    }

    double gpu_row_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      const std::size_t index = row_offset + static_cast<std::size_t>(col);
      const float reference = static_cast<float>(
          std::exp(static_cast<double>(input[index]) - row_max) / row_sum);
      const float absolute_error = std::abs(gpu_output[index] - reference);
      const float scale =
          std::max(std::abs(reference), std::abs(gpu_output[index]));
      const float relative_error =
          scale > 0.0F ? absolute_error / scale : 0.0F;
      max_absolute_error = std::max(max_absolute_error, absolute_error);
      max_relative_error = std::max(max_relative_error, relative_error);
      if (absolute_error >
          kAbsoluteTolerance + kRelativeTolerance * std::abs(reference)) {
        ++tolerance_violations;
      }
      gpu_row_sum += static_cast<double>(gpu_output[index]);
    }

    const double row_sum_error = std::abs(gpu_row_sum - 1.0);
    max_row_sum_error = std::max(max_row_sum_error, row_sum_error);
    if (row_sum_error > kRowSumTolerance) {
      ++row_sum_violations;
    }
  }

  std::cout << "Validation shape: [" << rows << ", " << cols << "]\n";
  std::cout << "Tolerance violations: " << tolerance_violations << " / "
            << element_count << '\n';
  std::cout << "Row sum violations: " << row_sum_violations << " / " << rows
            << '\n';
  std::cout << "Max absolute error: " << max_absolute_error << '\n';
  std::cout << "Max relative error: " << max_relative_error << '\n';
  std::cout << "Max row sum error: " << max_row_sum_error << '\n';
  if (tolerance_violations != 0 || row_sum_violations != 0) {
    throw std::runtime_error("FP32 softmax validation failed");
  }
  std::cout << "Validation: PASSED\n";
}

int positive_int(const char* text, const char* name) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

std::uint32_t seed_value(const char* text) {
  const unsigned long value = std::stoul(text);
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("seed must fit in uint32");
  }
  return static_cast<std::uint32_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 6) {
      throw std::invalid_argument(
          "usage: softmax.exe [rows] [cols] [iterations] [validation_rows] "
          "[seed]");
    }

    const int rows = argc > 1 ? positive_int(argv[1], "rows") : 4096;
    const int cols = argc > 2 ? positive_int(argv[2], "cols") : 1024;
    const int iterations = argc > 3 ? positive_int(argv[3], "iterations") : 100;
    const int validation_rows =
        argc > 4 ? positive_int(argv[4], "validation_rows") : 32;
    const std::uint32_t seed = argc > 5 ? seed_value(argv[5]) : 12345U;

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const std::size_t element_count = element_count_for(rows, cols);
    const std::vector<float> input = make_input(element_count, seed);
    const std::size_t bytes = element_count * sizeof(float);
    DeviceFloatBuffer device_input(element_count);
    DeviceFloatBuffer device_output(element_count);
    copy_input_to_device(input, device_input);

    const float elapsed_us = benchmark_us(
        device_input.get(), device_output.get(), rows, cols, iterations);
    const double transferred_bytes = 2.0 * static_cast<double>(bytes);
    const double bandwidth_gbps =
        transferred_bytes / (static_cast<double>(elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Shape: [" << rows << ", " << cols << "] FP32\n";
    std::cout << "Row-wise softmax: " << elapsed_us << " us\n";
    std::cout << "Effective bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    validate(validation_rows, cols, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
