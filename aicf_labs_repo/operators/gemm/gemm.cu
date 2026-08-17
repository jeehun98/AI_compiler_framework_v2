#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

constexpr int kBlockRows = 16;
constexpr int kBlockCols = 16;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr float kAbsoluteTolerance = 1.0e-4F;
constexpr float kRelativeTolerance = 1.0e-4F;
constexpr float kTensorCoreAbsoluteTolerance = 1.0e-3F;
constexpr float kTensorCoreRelativeTolerance = 1.0e-3F;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__global__ void gemm_fp32(const float* a, const float* b, float* c, int m,
                          int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  const std::size_t a_row_offset =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
  float accumulator = 0.0F;
  for (int inner = 0; inner < k; ++inner) {
    const float a_value = a[a_row_offset + static_cast<std::size_t>(inner)];
    const std::size_t b_index =
        static_cast<std::size_t>(inner) * static_cast<std::size_t>(n) +
        static_cast<std::size_t>(col);
    const float b_value = b[b_index];
    accumulator = fmaf(a_value, b_value, accumulator);
  }

  const std::size_t c_index =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
      static_cast<std::size_t>(col);
  c[c_index] = accumulator;
}

__global__ void gemm_wmma_fp16_fp32(const __half* a, const __half* b,
                                     float* c, int n, int k) {
  const int tile_row = static_cast<int>(blockIdx.y);
  const int tile_col = static_cast<int>(blockIdx.x);

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                         __half, nvcuda::wmma::row_major>
      a_fragment;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                         __half, nvcuda::wmma::row_major>
      b_fragment;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         float>
      accumulator_fragment;
  nvcuda::wmma::fill_fragment(accumulator_fragment, 0.0F);

  for (int inner = 0; inner < k; inner += kWmmaK) {
    const __half* a_tile =
        a + static_cast<std::size_t>(tile_row * kWmmaM) *
                static_cast<std::size_t>(k) +
        static_cast<std::size_t>(inner);
    const __half* b_tile =
        b + static_cast<std::size_t>(inner) * static_cast<std::size_t>(n) +
        static_cast<std::size_t>(tile_col * kWmmaN);
    nvcuda::wmma::load_matrix_sync(a_fragment, a_tile, k);
    nvcuda::wmma::load_matrix_sync(b_fragment, b_tile, n);
    nvcuda::wmma::mma_sync(accumulator_fragment, a_fragment, b_fragment,
                           accumulator_fragment);
  }

  float* c_tile =
      c + static_cast<std::size_t>(tile_row * kWmmaM) *
              static_cast<std::size_t>(n) +
      static_cast<std::size_t>(tile_col * kWmmaN);
  nvcuda::wmma::store_matrix_sync(c_tile, accumulator_fragment, n,
                                  nvcuda::wmma::mem_row_major);
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

class DeviceHalfBuffer {
 public:
  explicit DeviceHalfBuffer(std::size_t element_count) {
    CUDA_CHECK(cudaMalloc(&pointer_, element_count * sizeof(__half)));
  }

  ~DeviceHalfBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceHalfBuffer(const DeviceHalfBuffer&) = delete;
  DeviceHalfBuffer& operator=(const DeviceHalfBuffer&) = delete;

  __half* get() { return pointer_; }

 private:
  __half* pointer_ = nullptr;
};

struct HostMatrices {
  HostMatrices(std::size_t a_elements, std::size_t b_elements)
      : a(a_elements), b(b_elements) {}

  std::vector<float> a;
  std::vector<float> b;
};

struct HostHalfMatrices {
  HostHalfMatrices(std::size_t a_elements, std::size_t b_elements)
      : a(a_elements), b(b_elements) {}

  std::vector<__half> a;
  std::vector<__half> b;
};

std::size_t matrix_elements(int rows, int cols) {
  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t column_count = static_cast<std::size_t>(cols);
  if (row_count > std::numeric_limits<std::size_t>::max() / column_count) {
    throw std::invalid_argument("matrix dimensions overflow size_t");
  }
  return row_count * column_count;
}

HostMatrices make_matrices(int m, int n, int k, std::uint32_t seed) {
  HostMatrices matrices(matrix_elements(m, k), matrix_elements(k, n));
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  for (float& value : matrices.a) {
    value = distribution(generator);
  }
  for (float& value : matrices.b) {
    value = distribution(generator);
  }
  return matrices;
}

HostHalfMatrices convert_to_half(const HostMatrices& matrices) {
  HostHalfMatrices half_matrices(matrices.a.size(), matrices.b.size());
  std::transform(matrices.a.begin(), matrices.a.end(), half_matrices.a.begin(),
                 [](float value) { return __float2half(value); });
  std::transform(matrices.b.begin(), matrices.b.end(), half_matrices.b.begin(),
                 [](float value) { return __float2half(value); });
  return half_matrices;
}

void copy_matrices_to_device(const HostMatrices& matrices,
                             DeviceFloatBuffer& a, DeviceFloatBuffer& b) {
  CUDA_CHECK(cudaMemcpy(a.get(), matrices.a.data(),
                        matrices.a.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.get(), matrices.b.data(),
                        matrices.b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void copy_half_matrices_to_device(const HostHalfMatrices& matrices,
                                  DeviceHalfBuffer& a, DeviceHalfBuffer& b) {
  CUDA_CHECK(cudaMemcpy(a.get(), matrices.a.data(),
                        matrices.a.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.get(), matrices.b.data(),
                        matrices.b.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
}

dim3 grid_for(int m, int n) {
  return dim3((n + kBlockCols - 1) / kBlockCols,
              (m + kBlockRows - 1) / kBlockRows);
}

float benchmark_us(const float* a, const float* b, float* c, int m, int n,
                   int k, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  const dim3 block(kBlockCols, kBlockRows);
  const dim3 grid = grid_for(m, n);
  for (int i = 0; i < warmup_iterations; ++i) {
    gemm_fp32<<<grid, block>>>(a, b, c, m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    gemm_fp32<<<grid, block>>>(a, b, c, m, n, k);
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

float benchmark_tensor_core_us(const __half* a, const __half* b, float* c,
                               int m, int n, int k, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  const dim3 block(32);
  const dim3 grid(n / kWmmaN, m / kWmmaM);
  for (int i = 0; i < warmup_iterations; ++i) {
    gemm_wmma_fp16_fp32<<<grid, block>>>(a, b, c, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    gemm_wmma_fp16_fp32<<<grid, block>>>(a, b, c, n, k);
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

void validate(int size, std::uint32_t seed) {
  const int m = size;
  const int n = size;
  const int k = size;
  const HostMatrices matrices = make_matrices(m, n, k, seed);
  const std::size_t c_elements = matrix_elements(m, n);
  DeviceFloatBuffer device_a(matrices.a.size());
  DeviceFloatBuffer device_b(matrices.b.size());
  DeviceFloatBuffer device_c(c_elements);
  copy_matrices_to_device(matrices, device_a, device_b);

  const dim3 block(kBlockCols, kBlockRows);
  gemm_fp32<<<grid_for(m, n), block>>>(device_a.get(), device_b.get(),
                                      device_c.get(), m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(c_elements);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_c.get(),
                        c_elements * sizeof(float), cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  int non_finite_outputs = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double reference_sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        const std::size_t a_index =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(k) +
            static_cast<std::size_t>(inner);
        const std::size_t b_index =
            static_cast<std::size_t>(inner) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(col);
        reference_sum += static_cast<double>(matrices.a[a_index]) *
                         static_cast<double>(matrices.b[b_index]);
      }

      const float reference = static_cast<float>(reference_sum);
      const std::size_t c_index =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
          static_cast<std::size_t>(col);
      const float gpu_value = gpu_output[c_index];
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

  std::cout << "Validation shape: [" << m << ", " << n << ", " << k
            << "]\n";
  std::cout << "Tolerance violations: " << tolerance_violations << " / "
            << c_elements << '\n';
  std::cout << "Non-finite outputs: " << non_finite_outputs << " / "
            << c_elements << '\n';
  std::cout << "Max absolute error: " << max_absolute_error << '\n';
  std::cout << "Max relative error: " << max_relative_error << '\n';
  if (tolerance_violations != 0 || non_finite_outputs != 0) {
    throw std::runtime_error("FP32 gemm validation failed");
  }
  std::cout << "Validation: PASSED\n";
}

void validate_tensor_core(int size, std::uint32_t seed) {
  const int m = size;
  const int n = size;
  const int k = size;
  const HostMatrices source_matrices = make_matrices(m, n, k, seed);
  const HostHalfMatrices matrices = convert_to_half(source_matrices);
  const std::size_t c_elements = matrix_elements(m, n);
  DeviceHalfBuffer device_a(matrices.a.size());
  DeviceHalfBuffer device_b(matrices.b.size());
  DeviceFloatBuffer device_c(c_elements);
  copy_half_matrices_to_device(matrices, device_a, device_b);

  const dim3 block(32);
  const dim3 grid(n / kWmmaN, m / kWmmaM);
  gemm_wmma_fp16_fp32<<<grid, block>>>(device_a.get(), device_b.get(),
                                      device_c.get(), n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(c_elements);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_c.get(),
                        c_elements * sizeof(float), cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  int non_finite_outputs = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double reference_sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        const std::size_t a_index =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(k) +
            static_cast<std::size_t>(inner);
        const std::size_t b_index =
            static_cast<std::size_t>(inner) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(col);
        reference_sum +=
            static_cast<double>(__half2float(matrices.a[a_index])) *
            static_cast<double>(__half2float(matrices.b[b_index]));
      }

      const float reference = static_cast<float>(reference_sum);
      const std::size_t c_index =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
          static_cast<std::size_t>(col);
      const float gpu_value = gpu_output[c_index];
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
          kTensorCoreAbsoluteTolerance +
              kTensorCoreRelativeTolerance * std::abs(reference)) {
        ++tolerance_violations;
      }
    }
  }

  std::cout << "Tensor Core validation shape: [" << m << ", " << n << ", "
            << k << "]\n";
  std::cout << "Tensor Core tolerance violations: " << tolerance_violations
            << " / " << c_elements << '\n';
  std::cout << "Tensor Core non-finite outputs: " << non_finite_outputs
            << " / " << c_elements << '\n';
  std::cout << "Tensor Core max absolute error: " << max_absolute_error
            << '\n';
  std::cout << "Tensor Core max relative error: " << max_relative_error
            << '\n';
  if (tolerance_violations != 0 || non_finite_outputs != 0) {
    throw std::runtime_error("Tensor Core gemm validation failed");
  }
  std::cout << "Tensor Core validation: PASSED\n";
}

void require_wmma_multiple(int value, const char* name) {
  if (value % 16 != 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a multiple of 16 for WMMA");
  }
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
    if (argc > 7) {
      throw std::invalid_argument(
          "usage: gemm.exe [M] [N] [K] [iterations] [validation_size] "
          "[seed]");
    }

    const int m = argc > 1 ? positive_int(argv[1], "M") : 512;
    const int n = argc > 2 ? positive_int(argv[2], "N") : 512;
    const int k = argc > 3 ? positive_int(argv[3], "K") : 512;
    const int iterations = argc > 4 ? positive_int(argv[4], "iterations") : 100;
    const int validation_size =
        argc > 5 ? positive_int(argv[5], "validation_size") : 64;
    const std::uint32_t seed = argc > 6 ? seed_value(argv[6]) : 12345U;
    require_wmma_multiple(m, "M");
    require_wmma_multiple(n, "N");
    require_wmma_multiple(k, "K");
    require_wmma_multiple(validation_size, "validation_size");

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    if (device.major < 7) {
      throw std::runtime_error("FP16 WMMA requires compute capability 7.0+");
    }

    const HostMatrices matrices = make_matrices(m, n, k, seed);
    const HostHalfMatrices half_matrices = convert_to_half(matrices);
    const std::size_t c_elements = matrix_elements(m, n);
    DeviceFloatBuffer device_a(matrices.a.size());
    DeviceFloatBuffer device_b(matrices.b.size());
    DeviceFloatBuffer device_c(c_elements);
    DeviceHalfBuffer device_half_a(half_matrices.a.size());
    DeviceHalfBuffer device_half_b(half_matrices.b.size());
    DeviceFloatBuffer device_tensor_c(c_elements);
    copy_matrices_to_device(matrices, device_a, device_b);
    copy_half_matrices_to_device(half_matrices, device_half_a, device_half_b);

    const float naive_elapsed_us =
        benchmark_us(device_a.get(), device_b.get(), device_c.get(), m, n, k,
                     iterations);
    const float tensor_elapsed_us = benchmark_tensor_core_us(
        device_half_a.get(), device_half_b.get(), device_tensor_c.get(), m, n,
        k, iterations);
    const double operation_count =
        2.0 * static_cast<double>(m) * static_cast<double>(n) *
        static_cast<double>(k);
    const double naive_throughput_gflops =
        operation_count / (static_cast<double>(naive_elapsed_us) * 1000.0);
    const double tensor_throughput_gflops =
        operation_count / (static_cast<double>(tensor_elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "GEMM shape: [" << m << ", " << n << ", " << k << "]\n";
    std::cout << "Naive FP32 GEMM: " << naive_elapsed_us << " us\n";
    std::cout << "Naive FP32 throughput: " << naive_throughput_gflops
              << " GFLOP/s\n";
    std::cout << "Tensor Core WMMA FP16->FP32 GEMM: " << tensor_elapsed_us
              << " us\n";
    std::cout << "Tensor Core WMMA throughput: " << tensor_throughput_gflops
              << " GFLOP/s\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    validate(validation_size, seed);
    validate_tensor_core(validation_size, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
