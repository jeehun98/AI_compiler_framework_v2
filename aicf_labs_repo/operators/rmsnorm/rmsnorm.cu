#include <cuda_runtime.h>

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
constexpr float kEpsilon = 1.0e-5F;
constexpr float kAbsoluteTolerance = 1.0e-5F;
constexpr float kRelativeTolerance = 2.0e-5F;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void rmsnorm_fp32(const float* input, const float* gamma,
                             float* output, int cols) {
  const int row = blockIdx.x;
  const std::size_t row_offset =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ float warp_values[kWarpsPerBlock];
  __shared__ float shared_inv_rms;

  float thread_square_sum = 0.0F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = input[row_offset + col];
    thread_square_sum = fmaf(value, value, thread_square_sum);
  }
  thread_square_sum = warp_reduce_sum(thread_square_sum);
  if (lane == 0) {
    warp_values[warp] = thread_square_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_square_sum =
        lane < kWarpsPerBlock ? warp_values[lane] : 0.0F;
    block_square_sum = warp_reduce_sum(block_square_sum);
    if (lane == 0) {
      const float mean_square =
          block_square_sum / static_cast<float>(cols);
      shared_inv_rms = rsqrtf(mean_square + kEpsilon);
    }
  }
  __syncthreads();

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized = input[row_offset + col] * shared_inv_rms;
    output[row_offset + col] = normalized * gamma[col];
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

struct HostData {
  HostData(std::size_t element_count, std::size_t column_count)
      : input(element_count), gamma(column_count) {}

  std::vector<float> input;
  std::vector<float> gamma;
};

std::size_t element_count_for(int rows, int cols) {
  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t column_count = static_cast<std::size_t>(cols);
  if (row_count > std::numeric_limits<std::size_t>::max() / column_count) {
    throw std::invalid_argument("rows * cols overflows size_t");
  }
  return row_count * column_count;
}

HostData make_data(int rows, int cols, std::uint32_t seed) {
  const std::size_t element_count = element_count_for(rows, cols);
  HostData data(element_count, static_cast<std::size_t>(cols));
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> input_distribution(-2.0F, 2.0F);
  std::uniform_real_distribution<float> gamma_distribution(0.5F, 1.5F);
  for (float& value : data.input) {
    value = input_distribution(generator);
  }
  for (float& value : data.gamma) {
    value = gamma_distribution(generator);
  }
  return data;
}

void copy_data_to_device(const HostData& data, DeviceFloatBuffer& input,
                         DeviceFloatBuffer& gamma) {
  CUDA_CHECK(cudaMemcpy(input.get(), data.input.data(),
                        data.input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gamma.get(), data.gamma.data(),
                        data.gamma.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

float benchmark_us(const float* input, const float* gamma, float* output,
                   int rows, int cols, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    rmsnorm_fp32<<<rows, kThreads>>>(input, gamma, output, cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    rmsnorm_fp32<<<rows, kThreads>>>(input, gamma, output, cols);
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
  const HostData data = make_data(rows, cols, seed);
  const std::size_t element_count = data.input.size();
  DeviceFloatBuffer device_input(element_count);
  DeviceFloatBuffer device_gamma(static_cast<std::size_t>(cols));
  DeviceFloatBuffer device_output(element_count);
  copy_data_to_device(data, device_input, device_gamma);

  rmsnorm_fp32<<<rows, kThreads>>>(device_input.get(), device_gamma.get(),
                                  device_output.get(), cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(element_count);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.get(),
                        element_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  int non_finite_outputs = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (int row = 0; row < rows; ++row) {
    const std::size_t row_offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    double square_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      const double value = static_cast<double>(data.input[row_offset + col]);
      square_sum += value * value;
    }
    const double mean_square = square_sum / static_cast<double>(cols);
    const double inv_rms =
        1.0 / std::sqrt(mean_square + static_cast<double>(kEpsilon));

    for (int col = 0; col < cols; ++col) {
      const std::size_t index = row_offset + static_cast<std::size_t>(col);
      const float reference = static_cast<float>(
          static_cast<double>(data.input[index]) * inv_rms *
          static_cast<double>(data.gamma[col]));
      const float gpu_value = gpu_output[index];
      if (!std::isfinite(gpu_value)) {
        ++non_finite_outputs;
        continue;
      }
      const float absolute_error = std::abs(gpu_value - reference);
      const float scale = std::max(std::abs(reference), std::abs(gpu_value));
      const float relative_error =
          scale > 0.0F ? absolute_error / scale : 0.0F;
      max_absolute_error = std::max(max_absolute_error, absolute_error);
      max_relative_error = std::max(max_relative_error, relative_error);
      if (absolute_error >
          kAbsoluteTolerance + kRelativeTolerance * std::abs(reference)) {
        ++tolerance_violations;
      }
    }
  }

  std::cout << "Validation shape: [" << rows << ", " << cols << "]\n";
  std::cout << "Tolerance violations: " << tolerance_violations << " / "
            << element_count << '\n';
  std::cout << "Non-finite outputs: " << non_finite_outputs << " / "
            << element_count << '\n';
  std::cout << "Max absolute error: " << max_absolute_error << '\n';
  std::cout << "Max relative error: " << max_relative_error << '\n';
  if (tolerance_violations != 0 || non_finite_outputs != 0) {
    throw std::runtime_error("FP32 rmsnorm validation failed");
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
          "usage: rmsnorm.exe [rows] [cols] [iterations] "
          "[validation_rows] [seed]");
    }

    const int rows = argc > 1 ? positive_int(argv[1], "rows") : 4096;
    const int cols = argc > 2 ? positive_int(argv[2], "cols") : 1024;
    const int iterations = argc > 3 ? positive_int(argv[3], "iterations") : 100;
    const int validation_rows =
        argc > 4 ? positive_int(argv[4], "validation_rows") : 32;
    const std::uint32_t seed = argc > 5 ? seed_value(argv[5]) : 12345U;

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const HostData data = make_data(rows, cols, seed);
    const std::size_t element_count = data.input.size();
    const std::size_t bytes = element_count * sizeof(float);
    DeviceFloatBuffer device_input(element_count);
    DeviceFloatBuffer device_gamma(static_cast<std::size_t>(cols));
    DeviceFloatBuffer device_output(element_count);
    copy_data_to_device(data, device_input, device_gamma);

    const float elapsed_us = benchmark_us(
        device_input.get(), device_gamma.get(), device_output.get(), rows, cols,
        iterations);
    const double logical_bytes =
        2.0 * static_cast<double>(bytes) +
        static_cast<double>(cols) * sizeof(float);
    const double bandwidth_gbps =
        logical_bytes / (static_cast<double>(elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Shape: [" << rows << ", " << cols << "] FP32\n";
    std::cout << "Row-wise rmsnorm: " << elapsed_us << " us\n";
    std::cout << "Logical bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    validate(validation_rows, cols, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
