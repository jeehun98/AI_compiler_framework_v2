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
constexpr float kAbsoluteTolerance = 1.0e-5F;
constexpr float kRelativeTolerance = 2.0e-6F;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__global__ void exp_fp32(const float* input, float* output,
                         int element_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < element_count) {
    output[index] = expf(input[index]);
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

std::vector<float> make_input(std::size_t element_count, std::uint32_t seed) {
  std::vector<float> input(element_count);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-10.0F, 10.0F);
  for (float& value : input) {
    value = distribution(generator);
  }
  return input;
}

int blocks_for(int element_count) {
  return ((element_count - 1) / kThreads) + 1;
}

void copy_input_to_device(const std::vector<float>& input,
                          DeviceFloatBuffer& device_input) {
  const std::size_t bytes = input.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes,
                        cudaMemcpyHostToDevice));
}

float benchmark_us(const float* input, float* output, int element_count,
                   int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  const int blocks = blocks_for(element_count);
  for (int i = 0; i < warmup_iterations; ++i) {
    exp_fp32<<<blocks, kThreads>>>(input, output, element_count);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    exp_fp32<<<blocks, kThreads>>>(input, output, element_count);
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

void validate(int element_count, std::uint32_t seed) {
  const std::vector<float> input = make_input(element_count, seed);
  const std::size_t bytes =
      static_cast<std::size_t>(element_count) * sizeof(float);
  DeviceFloatBuffer device_input(element_count);
  DeviceFloatBuffer device_output(element_count);
  copy_input_to_device(input, device_input);

  exp_fp32<<<blocks_for(element_count), kThreads>>>(
      device_input.get(), device_output.get(), element_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(element_count);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.get(), bytes,
                        cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (int i = 0; i < element_count; ++i) {
    const float reference =
        static_cast<float>(std::exp(static_cast<double>(input[i])));
    const float absolute_error = std::abs(gpu_output[i] - reference);
    const float scale = std::max(std::abs(reference), std::abs(gpu_output[i]));
    const float relative_error = scale > 0.0F ? absolute_error / scale : 0.0F;
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, relative_error);
    if (absolute_error >
        kAbsoluteTolerance + kRelativeTolerance * std::abs(reference)) {
      ++tolerance_violations;
    }
  }

  std::cout << "Validation elements: " << element_count << '\n';
  std::cout << "Tolerance violations: " << tolerance_violations << " / "
            << element_count << '\n';
  std::cout << "Max absolute error: " << max_absolute_error << '\n';
  std::cout << "Max relative error: " << max_relative_error << '\n';
  if (tolerance_violations != 0) {
    throw std::runtime_error("FP32 exp validation failed");
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
    if (argc > 5) {
      throw std::invalid_argument(
          "usage: exp.exe [elements] [iterations] [validation_elements] "
          "[seed]");
    }

    const int elements =
        argc > 1 ? positive_int(argv[1], "elements") : (1 << 24);
    const int iterations = argc > 2 ? positive_int(argv[2], "iterations") : 100;
    const int validation_elements =
        argc > 3 ? positive_int(argv[3], "validation_elements") : 4096;
    const std::uint32_t seed = argc > 4 ? seed_value(argv[4]) : 12345U;

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const std::vector<float> input = make_input(elements, seed);
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
    DeviceFloatBuffer device_input(elements);
    DeviceFloatBuffer device_output(elements);
    copy_input_to_device(input, device_input);

    const float elapsed_us = benchmark_us(
        device_input.get(), device_output.get(), elements, iterations);
    const double transferred_bytes = 2.0 * static_cast<double>(bytes);
    const double bandwidth_gbps =
        transferred_bytes / (static_cast<double>(elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Elements: " << elements << " FP32\n";
    std::cout << "Elementwise exp: " << elapsed_us << " us\n";
    std::cout << "Effective bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<float>::max_digits10);

    validate(validation_elements, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
