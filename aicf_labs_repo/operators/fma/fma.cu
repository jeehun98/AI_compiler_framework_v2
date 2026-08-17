#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "validation.hpp"


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
// Numerical validation policy is implemented in validation.cu.



#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

// Separated step 1: temporary[i] = a[i] * b[i].
// One thread handles two adjacent FP16 elements through one half2 load/store.
__global__ void mul_half2(const half* a, const half* b, half* temporary,
                          int pair_count) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < pair_count) {
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    half2* temporary2 = reinterpret_cast<half2*>(temporary);
    temporary2[pair] = __hmul2(a2[pair], b2[pair]);
  }
}

// Separated step 2: output[i] = temporary[i] + c[i].
__global__ void add_half2(const half* temporary, const half* c, half* output,
                          int pair_count) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < pair_count) {
    const half2* temporary2 = reinterpret_cast<const half2*>(temporary);
    const half2* c2 = reinterpret_cast<const half2*>(c);
    half2* output2 = reinterpret_cast<half2*>(output);
    output2[pair] = __hadd2(temporary2[pair], c2[pair]);
  }
}

// Optimized version: no intermediate tensor and only one kernel launch.
// This is an elementwise CUDA-core FMA, not a Tensor Core MMA operation.
__global__ void fma_half2(const half* a, const half* b, const half* c,
                          half* output, int pair_count) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < pair_count) {
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    const half2* c2 = reinterpret_cast<const half2*>(c);
    half2* output2 = reinterpret_cast<half2*>(output);
    output2[pair] = __hfma2(a2[pair], b2[pair], c2[pair]);
  }
}

class DeviceHalfBuffer {
 public:
  explicit DeviceHalfBuffer(std::size_t element_count)
      : bytes_(element_count * sizeof(half)) {
    const cudaError_t error = cudaMalloc(&pointer_, bytes_);
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc: ") +
                               cudaGetErrorString(error));
    }
  }

  ~DeviceHalfBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceHalfBuffer(const DeviceHalfBuffer&) = delete;
  DeviceHalfBuffer& operator=(const DeviceHalfBuffer&) = delete;

  half* get() { return pointer_; }

 private:
  half* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};


struct HostInputs {
  explicit HostInputs(std::size_t element_count)
      : a(element_count), b(element_count), c(element_count) {}

  std::vector<half> a;
  std::vector<half> b;
  std::vector<half> c;
};

half quantize_to_half(float value) { return __float2half_rn(value); }

HostInputs ordinary_inputs(std::size_t element_count, std::uint32_t seed) {
  HostInputs inputs(element_count);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
  for (std::size_t i = 0; i < element_count; ++i) {
    inputs.a[i] = quantize_to_half(distribution(generator));
    inputs.b[i] = quantize_to_half(distribution(generator));
    inputs.c[i] = quantize_to_half(distribution(generator));
  }
  return inputs;
}

template <typename Launch>
float benchmark_us(Launch launch, int iterations) {
  constexpr int kWarmup = 10;
  for (int i = 0; i < kWarmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms * 1000.0F / static_cast<float>(iterations);
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

void launch_fma_validation_kernels(const half* a, const half* b, const half* c,
                                   half* temporary, half* separated_output,
                                   half* fused_output, int element_count) {
  const int pair_count = element_count / 2;
  const int blocks = (pair_count + kThreads - 1) / kThreads;
  mul_half2<<<blocks, kThreads>>>(a, b, temporary, pair_count);
  add_half2<<<blocks, kThreads>>>(temporary, c, separated_output, pair_count);
  fma_half2<<<blocks, kThreads>>>(a, b, c, fused_output, pair_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  try {
    const int elements =
        argc > 1 ? positive_int(argv[1], "benchmark_elements") : (1 << 24);
    const int iterations = argc > 2 ? positive_int(argv[2], "iterations") : 100;
    const int validation_elements =
        argc > 3 ? positive_int(argv[3], "validation_elements") : 4096;
    const std::uint32_t seed = argc > 4 ? seed_value(argv[4]) : 12345U;
    if ((elements % 2) != 0) {
      throw std::invalid_argument(
          "benchmark_elements must be even for half2");
    }
    if ((validation_elements % 2) != 0) {
      throw std::invalid_argument(
          "validation_elements must be even for half2");
    }

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(half);
    HostInputs benchmark_inputs = ordinary_inputs(elements, seed);
    DeviceHalfBuffer a(elements);
    DeviceHalfBuffer b(elements);
    DeviceHalfBuffer c(elements);
    DeviceHalfBuffer temporary(elements);
    DeviceHalfBuffer separated_output(elements);
    DeviceHalfBuffer fused_output(elements);
    CUDA_CHECK(cudaMemcpy(a.get(), benchmark_inputs.a.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.get(), benchmark_inputs.b.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.get(), benchmark_inputs.c.data(), bytes,
                          cudaMemcpyHostToDevice));

    const int pair_count = elements / 2;
    const int blocks = (pair_count + kThreads - 1) / kThreads;

    const float separated_us = benchmark_us(
        [&] {
          mul_half2<<<blocks, kThreads>>>(a.get(), b.get(), temporary.get(),
                                          pair_count);
          add_half2<<<blocks, kThreads>>>(temporary.get(), c.get(),
                                          separated_output.get(), pair_count);
        },
        iterations);

    const float fused_us = benchmark_us(
        [&] {
          fma_half2<<<blocks, kThreads>>>(a.get(), b.get(), c.get(),
                                          fused_output.get(), pair_count);
        },
        iterations);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major << device.minor
              << ")\n";
    std::cout << "Elements: " << elements << " FP16\n";
    std::cout << "Separated Mul -> Add: " << separated_us
              << " us  [2 launches, " << (bytes / (1024.0 * 1024.0))
              << " MiB temporary]\n";
    std::cout << "Fused half2 FMA:      " << fused_us
              << " us  [1 launch, no temporary]\n";
    std::cout << "Observed speedup:     " << (separated_us / fused_us) << "x\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    run_fma_validation(validation_elements, seed);

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
