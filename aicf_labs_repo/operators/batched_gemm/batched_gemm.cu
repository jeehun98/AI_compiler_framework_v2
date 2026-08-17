#include <cuda_runtime.h>

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
constexpr float kAbsoluteTolerance = 1.0e-4F;
constexpr float kRelativeTolerance = 1.0e-4F;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__global__ void batched_gemm_fp32(const float* a, const float* b, float* c,
                                   int batch_count, int m, int n, int k) {
  const int batch = static_cast<int>(blockIdx.z);
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batch_count || row >= m || col >= n) {
    return;
  }

  const std::size_t a_batch_offset =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(m) *
      static_cast<std::size_t>(k);
  const std::size_t b_batch_offset =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) *
      static_cast<std::size_t>(n);
  const std::size_t a_row_offset =
      a_batch_offset +
      static_cast<std::size_t>(row) * static_cast<std::size_t>(k);

  float accumulator = 0.0F;
  for (int inner = 0; inner < k; ++inner) {
    const float a_value = a[a_row_offset + static_cast<std::size_t>(inner)];
    const std::size_t b_index =
        b_batch_offset +
        static_cast<std::size_t>(inner) * static_cast<std::size_t>(n) +
        static_cast<std::size_t>(col);
    const float b_value = b[b_index];
    accumulator = fmaf(a_value, b_value, accumulator);
  }

  const std::size_t c_index =
      (static_cast<std::size_t>(batch) * static_cast<std::size_t>(m) +
       static_cast<std::size_t>(row)) *
          static_cast<std::size_t>(n) +
      static_cast<std::size_t>(col);
  c[c_index] = accumulator;
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

struct BatchedGemmShape {
  int batch_count;
  int m;
  int n;
  int k;
};

struct HostMatrices {
  HostMatrices(std::size_t a_elements, std::size_t b_elements)
      : a(a_elements), b(b_elements) {}

  std::vector<float> a;
  std::vector<float> b;
};

std::size_t checked_multiply(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument("batched GEMM dimensions overflow size_t");
  }
  return left * right;
}

std::size_t tensor_elements(int first, int second, int third) {
  std::size_t elements = static_cast<std::size_t>(first);
  elements = checked_multiply(elements, static_cast<std::size_t>(second));
  return checked_multiply(elements, static_cast<std::size_t>(third));
}

std::size_t a_elements(const BatchedGemmShape& shape) {
  return tensor_elements(shape.batch_count, shape.m, shape.k);
}

std::size_t b_elements(const BatchedGemmShape& shape) {
  return tensor_elements(shape.batch_count, shape.k, shape.n);
}

std::size_t c_elements(const BatchedGemmShape& shape) {
  return tensor_elements(shape.batch_count, shape.m, shape.n);
}

HostMatrices make_matrices(const BatchedGemmShape& shape, std::uint32_t seed) {
  HostMatrices matrices(a_elements(shape), b_elements(shape));
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

void copy_matrices_to_device(const HostMatrices& matrices,
                             DeviceFloatBuffer& a, DeviceFloatBuffer& b) {
  CUDA_CHECK(cudaMemcpy(a.get(), matrices.a.data(),
                        matrices.a.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.get(), matrices.b.data(),
                        matrices.b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

dim3 grid_for(const BatchedGemmShape& shape) {
  return dim3((shape.n + kBlockCols - 1) / kBlockCols,
              (shape.m + kBlockRows - 1) / kBlockRows, shape.batch_count);
}

void launch_batched_gemm(const float* a, const float* b, float* c,
                         const BatchedGemmShape& shape) {
  const dim3 block(kBlockCols, kBlockRows);
  batched_gemm_fp32<<<grid_for(shape), block>>>(
      a, b, c, shape.batch_count, shape.m, shape.n, shape.k);
}

float benchmark_us(const float* a, const float* b, float* c,
                   const BatchedGemmShape& shape, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    launch_batched_gemm(a, b, c, shape);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_batched_gemm(a, b, c, shape);
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

std::size_t a_offset(const BatchedGemmShape& shape, int batch, int row,
                     int inner) {
  return (static_cast<std::size_t>(batch) * shape.m + row) * shape.k + inner;
}

std::size_t b_offset(const BatchedGemmShape& shape, int batch, int inner,
                     int col) {
  return (static_cast<std::size_t>(batch) * shape.k + inner) * shape.n + col;
}

std::size_t c_offset(const BatchedGemmShape& shape, int batch, int row,
                     int col) {
  return (static_cast<std::size_t>(batch) * shape.m + row) * shape.n + col;
}

void validate(int size, std::uint32_t seed) {
  const BatchedGemmShape shape{2, size, size, size};
  const HostMatrices matrices = make_matrices(shape, seed);
  const std::size_t result_elements = c_elements(shape);
  DeviceFloatBuffer device_a(matrices.a.size());
  DeviceFloatBuffer device_b(matrices.b.size());
  DeviceFloatBuffer device_c(result_elements);
  copy_matrices_to_device(matrices, device_a, device_b);

  launch_batched_gemm(device_a.get(), device_b.get(), device_c.get(), shape);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(result_elements);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_c.get(),
                        result_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  int non_finite_outputs = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (int batch = 0; batch < shape.batch_count; ++batch) {
    for (int row = 0; row < shape.m; ++row) {
      for (int col = 0; col < shape.n; ++col) {
        double reference_sum = 0.0;
        for (int inner = 0; inner < shape.k; ++inner) {
          reference_sum +=
              static_cast<double>(
                  matrices.a[a_offset(shape, batch, row, inner)]) *
              static_cast<double>(
                  matrices.b[b_offset(shape, batch, inner, col)]);
        }

        const float reference = static_cast<float>(reference_sum);
        const float gpu_value =
            gpu_output[c_offset(shape, batch, row, col)];
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
  }

  std::cout << "Validation shape: B=" << shape.batch_count
            << " M=" << shape.m << " N=" << shape.n << " K=" << shape.k
            << '\n';
  std::cout << "Tolerance violations: " << tolerance_violations << " / "
            << result_elements << '\n';
  std::cout << "Non-finite outputs: " << non_finite_outputs << " / "
            << result_elements << '\n';
  std::cout << "Max absolute error: " << max_absolute_error << '\n';
  std::cout << "Max relative error: " << max_relative_error << '\n';
  if (tolerance_violations != 0 || non_finite_outputs != 0) {
    throw std::runtime_error("FP32 batched GEMM validation failed");
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
    if (argc > 8) {
      throw std::invalid_argument(
          "usage: batched_gemm.exe [B] [M] [N] [K] [iterations] "
          "[validation_size] [seed]");
    }

    const BatchedGemmShape shape{
        argc > 1 ? positive_int(argv[1], "B") : 16,
        argc > 2 ? positive_int(argv[2], "M") : 128,
        argc > 3 ? positive_int(argv[3], "N") : 128,
        argc > 4 ? positive_int(argv[4], "K") : 128};
    const int iterations = argc > 5 ? positive_int(argv[5], "iterations") : 100;
    const int validation_size =
        argc > 6 ? positive_int(argv[6], "validation_size") : 32;
    const std::uint32_t seed = argc > 7 ? seed_value(argv[7]) : 12345U;

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    if (shape.batch_count > device.maxGridSize[2]) {
      throw std::invalid_argument("B exceeds the device grid.z limit");
    }

    const HostMatrices matrices = make_matrices(shape, seed);
    const std::size_t result_elements = c_elements(shape);
    DeviceFloatBuffer device_a(matrices.a.size());
    DeviceFloatBuffer device_b(matrices.b.size());
    DeviceFloatBuffer device_c(result_elements);
    copy_matrices_to_device(matrices, device_a, device_b);

    const float elapsed_us = benchmark_us(device_a.get(), device_b.get(),
                                          device_c.get(), shape, iterations);
    const double operation_count =
        2.0 * static_cast<double>(shape.batch_count) * shape.m * shape.n *
        shape.k;
    const double throughput_gflops =
        operation_count / (static_cast<double>(elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Batched GEMM shape: B=" << shape.batch_count
              << " M=" << shape.m << " N=" << shape.n << " K=" << shape.k
              << " FP32\n";
    std::cout << "Naive batched GEMM: " << elapsed_us << " us\n";
    std::cout << "Throughput: " << throughput_gflops << " GFLOP/s\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    validate(validation_size, seed);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
