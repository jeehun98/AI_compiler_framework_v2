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

constexpr int kBlockWidth = 16;
constexpr int kBlockHeight = 16;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

__global__ void transpose_naive_fp32(const float* input, float* output,
                                     int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const std::size_t input_index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x;
    const std::size_t output_index =
        static_cast<std::size_t>(x) * static_cast<std::size_t>(height) + y;
    output[output_index] = input[input_index];
  }
}

constexpr int kTileDimension = 32;
constexpr int kTileBlockRows = 8;

__global__ void transpose_tiled_fp32(const float* input, float* output,
                                     int width, int height) {
  __shared__ float tile[kTileDimension][kTileDimension];

  int x = blockIdx.x * kTileDimension + threadIdx.x;
  int y = blockIdx.y * kTileDimension + threadIdx.y;
  for (int row = 0; row < kTileDimension; row += kTileBlockRows) {
    if (x < width && y + row < height) {
      const std::size_t input_index =
          static_cast<std::size_t>(y + row) *
              static_cast<std::size_t>(width) +
          x;
      tile[threadIdx.y + row][threadIdx.x] = input[input_index];
    }
  }
  __syncthreads();

  x = blockIdx.y * kTileDimension + threadIdx.x;
  y = blockIdx.x * kTileDimension + threadIdx.y;
  for (int row = 0; row < kTileDimension; row += kTileBlockRows) {
    if (x < height && y + row < width) {
      const std::size_t output_index =
          static_cast<std::size_t>(y + row) *
              static_cast<std::size_t>(height) +
          x;
      output[output_index] = tile[threadIdx.x][threadIdx.y + row];
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

std::size_t matrix_elements(int width, int height) {
  const std::uint64_t count =
      static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::invalid_argument("matrix byte size exceeds size_t");
  }
  return static_cast<std::size_t>(count);
}

std::vector<float> make_input(std::size_t element_count, std::uint32_t seed) {
  std::vector<float> input(element_count);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1000.0F, 1000.0F);
  for (float& value : input) {
    value = distribution(generator);
  }
  return input;
}

dim3 transpose_grid(int width, int height) {
  return dim3(((width - 1) / kBlockWidth) + 1,
              ((height - 1) / kBlockHeight) + 1);
}

dim3 tiled_transpose_grid(int width, int height) {
  return dim3(((width - 1) / kTileDimension) + 1,
              ((height - 1) / kTileDimension) + 1);
}

void copy_input_to_device(const std::vector<float>& input,
                          DeviceFloatBuffer& device_input) {
  const std::size_t bytes = input.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), bytes,
                        cudaMemcpyHostToDevice));
}

template <typename Launch>
float benchmark_us(Launch launch, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch();
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

void validate(const char* variant, const std::vector<float>& input,
              const float* device_output, int width, int height) {
  const std::size_t element_count = matrix_elements(width, height);
  const std::size_t bytes = element_count * sizeof(float);
  std::vector<float> gpu_output(element_count);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output, bytes,
                        cudaMemcpyDeviceToHost));

  std::size_t mismatches = 0;
  float max_absolute_error = 0.0F;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t input_index =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x;
      const std::size_t output_index =
          static_cast<std::size_t>(x) * static_cast<std::size_t>(height) + y;
      const float reference = input[input_index];
      const float absolute_error =
          std::abs(gpu_output[output_index] - reference);
      max_absolute_error = std::max(max_absolute_error, absolute_error);
      if (gpu_output[output_index] != reference) {
        ++mismatches;
      }
    }
  }

  std::cout << variant << " validation elements: " << element_count << '\n';
  std::cout << variant << " validation mismatches: " << mismatches << " / "
            << element_count << '\n';
  std::cout << variant << " max absolute error: " << max_absolute_error << '\n';
  if (mismatches != 0) {
    throw std::runtime_error(std::string("FP32 ") + variant +
                             " transpose validation failed");
  }
  std::cout << variant << " validation: PASSED\n";
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
          "usage: transpose.exe [width] [height] [iterations] [seed]");
    }

    const int width = argc > 1 ? positive_int(argv[1], "width") : 4096;
    const int height = argc > 2 ? positive_int(argv[2], "height") : 4096;
    const int iterations = argc > 3 ? positive_int(argv[3], "iterations") : 100;
    const std::uint32_t seed = argc > 4 ? seed_value(argv[4]) : 12345U;
    const std::size_t element_count = matrix_elements(width, height);
    const std::size_t bytes = element_count * sizeof(float);

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const std::vector<float> input = make_input(element_count, seed);
    DeviceFloatBuffer device_input(element_count);
    DeviceFloatBuffer naive_output(element_count);
    DeviceFloatBuffer tiled_output(element_count);
    copy_input_to_device(input, device_input);

    const dim3 naive_block(kBlockWidth, kBlockHeight);
    const dim3 naive_grid = transpose_grid(width, height);
    const dim3 tiled_block(kTileDimension, kTileBlockRows);
    const dim3 tiled_grid = tiled_transpose_grid(width, height);
    const float naive_us = benchmark_us(
        [&] {
          transpose_naive_fp32<<<naive_grid, naive_block>>>(
              device_input.get(), naive_output.get(), width, height);
        },
        iterations);
    const float tiled_us = benchmark_us(
        [&] {
          transpose_tiled_fp32<<<tiled_grid, tiled_block>>>(
              device_input.get(), tiled_output.get(), width, height);
        },
        iterations);
    const double transferred_bytes = 2.0 * static_cast<double>(bytes);
    const double naive_bandwidth_gbps =
        transferred_bytes / (static_cast<double>(naive_us) * 1000.0);
    const double tiled_bandwidth_gbps =
        transferred_bytes / (static_cast<double>(tiled_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Input matrix: " << height << " x " << width
              << " (rows x columns)\n";
    std::cout << "Output matrix: " << width << " x " << height
              << " (rows x columns)\n";
    std::cout << "Naive transpose: " << naive_us << " us\n";
    std::cout << "Naive effective bandwidth: " << naive_bandwidth_gbps
              << " GB/s\n";
    std::cout << "Tiled transpose (32x32, no padding): " << tiled_us
              << " us\n";
    std::cout << "Tiled effective bandwidth: " << tiled_bandwidth_gbps
              << " GB/s\n";
    std::cout << "Observed speedup: " << (naive_us / tiled_us) << "x\n";
    std::cout << std::defaultfloat
              << std::setprecision(std::numeric_limits<float>::max_digits10);

    validate("Naive", input, naive_output.get(), width, height);
    validate("Tiled", input, tiled_output.get(), width, height);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
