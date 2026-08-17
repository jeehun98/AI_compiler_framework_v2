#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
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

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__device__ __forceinline__ float warp_reduce_min(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fminf(value,
                  __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ void atomic_min_fp32(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = atomicCAS(address_as_int, 0, 0);
  while (value < __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void initialize_min_fp32(float* output) {
  output[0] = CUDART_INF_F;
}

__global__ void reduce_min_fp32(const float* input, float* output,
                                int element_count) {
  float thread_min = CUDART_INF_F;
  const int grid_stride = blockDim.x * gridDim.x;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < element_count; index += grid_stride) {
    thread_min = fminf(thread_min, input[index]);
  }

  thread_min = warp_reduce_min(thread_min);
  __shared__ float warp_minima[kWarpsPerBlock];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_minima[warp] = thread_min;
  }
  __syncthreads();

  if (warp == 0) {
    float block_min =
        lane < kWarpsPerBlock ? warp_minima[lane] : CUDART_INF_F;
    block_min = warp_reduce_min(block_min);
    if (lane == 0) {
      atomic_min_fp32(output, block_min);
    }
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
  std::uniform_real_distribution<float> distribution(-1000.0F, 1000.0F);
  for (float& value : input) {
    value = distribution(generator);
  }
  return input;
}

int reduction_blocks(int element_count, int multiprocessor_count) {
  const int required_blocks = ((element_count - 1) / kThreads) + 1;
  const int occupancy_blocks = std::max(1, multiprocessor_count * 8);
  return std::min(required_blocks, occupancy_blocks);
}

void copy_input_to_device(const std::vector<float>& input,
                          DeviceFloatBuffer& device_input) {
  const std::size_t bytes = input.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes,
                        cudaMemcpyHostToDevice));
}

void launch_reduce_min(const float* input, float* output, int element_count,
                       int blocks) {
  initialize_min_fp32<<<1, 1>>>(output);
  reduce_min_fp32<<<blocks, kThreads>>>(input, output, element_count);
}

float benchmark_us(const float* input, float* output, int element_count,
                   int blocks, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    launch_reduce_min(input, output, element_count, blocks);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_reduce_min(input, output, element_count, blocks);
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

void validate(int element_count, int multiprocessor_count, std::uint32_t seed) {
  const std::vector<float> input = make_input(element_count, seed);
  DeviceFloatBuffer device_input(element_count);
  DeviceFloatBuffer device_output(1);
  copy_input_to_device(input, device_input);

  const int blocks = reduction_blocks(element_count, multiprocessor_count);
  launch_reduce_min(device_input.get(), device_output.get(), element_count,
                    blocks);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float gpu_output = 0.0F;
  CUDA_CHECK(cudaMemcpy(&gpu_output, device_output.get(), sizeof(float),
                        cudaMemcpyDeviceToHost));

  float reference = std::numeric_limits<float>::infinity();
  for (float value : input) {
    reference = std::min(reference, value);
  }
  const bool passed = gpu_output == reference;

  std::cout << "Validation elements: " << element_count << '\n';
  std::cout << "CPU reference: " << reference << '\n';
  std::cout << "GPU result: " << gpu_output << '\n';
  std::cout << "Validation mismatch: " << (passed ? 0 : 1) << " / 1\n";
  if (!passed) {
    throw std::runtime_error("FP32 reduce_min validation failed");
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
          "usage: reduce_min.exe [elements] [iterations] "
          "[validation_elements] [seed]");
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
    const std::size_t input_bytes =
        static_cast<std::size_t>(elements) * sizeof(float);
    DeviceFloatBuffer device_input(elements);
    DeviceFloatBuffer device_output(1);
    copy_input_to_device(input, device_input);
    const int blocks = reduction_blocks(elements, device.multiProcessorCount);

    const float elapsed_us = benchmark_us(
        device_input.get(), device_output.get(), elements, blocks, iterations);
    const double bandwidth_gbps =
        static_cast<double>(input_bytes) /
        (static_cast<double>(elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Elements: " << elements << " FP32\n";
    std::cout << "Reduction blocks: " << blocks << '\n';
    std::cout << "Reduce min: " << elapsed_us << " us\n";
    std::cout << "Input bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    validate(validation_elements, device.multiProcessorCount, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}


