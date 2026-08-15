#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;

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

}  // namespace

int main(int argc, char** argv) {
  try {
    const int elements = argc > 1 ? positive_int(argv[1], "elements") : (1 << 24);
    const int iterations = argc > 2 ? positive_int(argv[2], "iterations") : 100;
    if ((elements % 2) != 0) {
      throw std::invalid_argument("elements must be even for half2");
    }

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(half);
    half* a = nullptr;
    half* b = nullptr;
    half* c = nullptr;
    half* temporary = nullptr;
    half* output = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMalloc(&temporary, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));

    CUDA_CHECK(cudaMemset(a, 0, bytes));
    CUDA_CHECK(cudaMemset(b, 0, bytes));
    CUDA_CHECK(cudaMemset(c, 0, bytes));

    const int pair_count = elements / 2;
    const int blocks = (pair_count + kThreads - 1) / kThreads;

    const float separated_us = benchmark_us(
        [&] {
          mul_half2<<<blocks, kThreads>>>(a, b, temporary, pair_count);
          add_half2<<<blocks, kThreads>>>(temporary, c, output, pair_count);
        },
        iterations);

    const float fused_us = benchmark_us(
        [&] { fma_half2<<<blocks, kThreads>>>(a, b, c, output, pair_count); },
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

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(temporary));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
