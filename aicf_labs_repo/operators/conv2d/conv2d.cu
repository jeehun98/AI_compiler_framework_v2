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

constexpr int kThreads = 256;
constexpr int kFilterSize = 3;
constexpr int kPadding = 1;
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

__global__ void conv2d_nchw_fp32(const float* input, const float* weight,
                                  float* output, int batch_size,
                                  int input_channels, int height, int width,
                                  int output_channels) {
  const std::size_t output_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_elements =
      static_cast<std::size_t>(batch_size) * output_channels * height * width;
  if (output_index >= output_elements) {
    return;
  }

  std::size_t remaining = output_index;
  const int output_x = static_cast<int>(remaining % width);
  remaining /= width;
  const int output_y = static_cast<int>(remaining % height);
  remaining /= height;
  const int output_channel =
      static_cast<int>(remaining % output_channels);
  const int batch = static_cast<int>(remaining / output_channels);

  float accumulator = 0.0F;
  for (int input_channel = 0; input_channel < input_channels;
       ++input_channel) {
    for (int filter_y = 0; filter_y < kFilterSize; ++filter_y) {
      const int input_y = output_y + filter_y - kPadding;
      if (input_y < 0 || input_y >= height) {
        continue;
      }
      for (int filter_x = 0; filter_x < kFilterSize; ++filter_x) {
        const int input_x = output_x + filter_x - kPadding;
        if (input_x < 0 || input_x >= width) {
          continue;
        }

        const std::size_t input_index =
            ((static_cast<std::size_t>(batch) * input_channels +
              input_channel) *
                 height +
             input_y) *
                width +
            input_x;
        const std::size_t weight_index =
            ((static_cast<std::size_t>(output_channel) * input_channels +
              input_channel) *
                 kFilterSize +
             filter_y) *
                kFilterSize +
            filter_x;
        const float input_value = input[input_index];
        const float weight_value = weight[weight_index];
        accumulator = fmaf(input_value, weight_value, accumulator);
      }
    }
  }

  output[output_index] = accumulator;
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

struct Conv2dShape {
  int batch_size;
  int input_channels;
  int height;
  int width;
  int output_channels;
};

struct HostData {
  HostData(std::size_t input_elements, std::size_t weight_elements)
      : input(input_elements), weight(weight_elements) {}

  std::vector<float> input;
  std::vector<float> weight;
};

std::size_t checked_multiply(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::invalid_argument("Conv2D dimensions overflow size_t");
  }
  return left * right;
}

std::size_t input_elements(const Conv2dShape& shape) {
  std::size_t elements = static_cast<std::size_t>(shape.batch_size);
  elements = checked_multiply(elements, shape.input_channels);
  elements = checked_multiply(elements, shape.height);
  return checked_multiply(elements, shape.width);
}

std::size_t weight_elements(const Conv2dShape& shape) {
  std::size_t elements = static_cast<std::size_t>(shape.output_channels);
  elements = checked_multiply(elements, shape.input_channels);
  elements = checked_multiply(elements, kFilterSize);
  return checked_multiply(elements, kFilterSize);
}

std::size_t output_elements(const Conv2dShape& shape) {
  std::size_t elements = static_cast<std::size_t>(shape.batch_size);
  elements = checked_multiply(elements, shape.output_channels);
  elements = checked_multiply(elements, shape.height);
  return checked_multiply(elements, shape.width);
}

HostData make_data(const Conv2dShape& shape, std::uint32_t seed) {
  HostData data(input_elements(shape), weight_elements(shape));
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  for (float& value : data.input) {
    value = distribution(generator);
  }
  for (float& value : data.weight) {
    value = distribution(generator);
  }
  return data;
}

void copy_data_to_device(const HostData& data, DeviceFloatBuffer& input,
                         DeviceFloatBuffer& weight) {
  CUDA_CHECK(cudaMemcpy(input.get(), data.input.data(),
                        data.input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(weight.get(), data.weight.data(),
                        data.weight.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

unsigned int blocks_for(std::size_t element_count) {
  const std::size_t blocks = (element_count + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("Conv2D output requires too many CUDA blocks");
  }
  return static_cast<unsigned int>(blocks);
}

void launch_conv2d(const float* input, const float* weight, float* output,
                   const Conv2dShape& shape) {
  conv2d_nchw_fp32<<<blocks_for(output_elements(shape)), kThreads>>>(
      input, weight, output, shape.batch_size, shape.input_channels,
      shape.height, shape.width, shape.output_channels);
}

float benchmark_us(const float* input, const float* weight, float* output,
                   const Conv2dShape& shape, int iterations) {
  const int warmup_iterations = iterations > 1 ? std::min(10, iterations) : 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    launch_conv2d(input, weight, output, shape);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_conv2d(input, weight, output, shape);
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

std::size_t input_offset(const Conv2dShape& shape, int batch,
                         int input_channel, int y, int x) {
  return ((static_cast<std::size_t>(batch) * shape.input_channels +
           input_channel) *
              shape.height +
          y) *
             shape.width +
         x;
}

std::size_t weight_offset(const Conv2dShape& shape, int output_channel,
                          int input_channel, int filter_y, int filter_x) {
  return ((static_cast<std::size_t>(output_channel) * shape.input_channels +
           input_channel) *
              kFilterSize +
          filter_y) *
             kFilterSize +
         filter_x;
}

std::size_t output_offset(const Conv2dShape& shape, int batch,
                          int output_channel, int y, int x) {
  return ((static_cast<std::size_t>(batch) * shape.output_channels +
           output_channel) *
              shape.height +
          y) *
             shape.width +
         x;
}

void validate(int spatial_size, std::uint32_t seed) {
  const Conv2dShape shape{1, 3, spatial_size, spatial_size, 4};
  const HostData data = make_data(shape, seed);
  const std::size_t result_elements = output_elements(shape);
  DeviceFloatBuffer device_input(data.input.size());
  DeviceFloatBuffer device_weight(data.weight.size());
  DeviceFloatBuffer device_output(result_elements);
  copy_data_to_device(data, device_input, device_weight);

  launch_conv2d(device_input.get(), device_weight.get(), device_output.get(),
                shape);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(result_elements);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), device_output.get(),
                        result_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int tolerance_violations = 0;
  int non_finite_outputs = 0;
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (int batch = 0; batch < shape.batch_size; ++batch) {
    for (int output_channel = 0; output_channel < shape.output_channels;
         ++output_channel) {
      for (int output_y = 0; output_y < shape.height; ++output_y) {
        for (int output_x = 0; output_x < shape.width; ++output_x) {
          double reference_sum = 0.0;
          for (int input_channel = 0;
               input_channel < shape.input_channels; ++input_channel) {
            for (int filter_y = 0; filter_y < kFilterSize; ++filter_y) {
              const int input_y = output_y + filter_y - kPadding;
              if (input_y < 0 || input_y >= shape.height) {
                continue;
              }
              for (int filter_x = 0; filter_x < kFilterSize; ++filter_x) {
                const int input_x = output_x + filter_x - kPadding;
                if (input_x < 0 || input_x >= shape.width) {
                  continue;
                }
                const float input_value =
                    data.input[input_offset(shape, batch, input_channel,
                                            input_y, input_x)];
                const float weight_value =
                    data.weight[weight_offset(shape, output_channel,
                                              input_channel, filter_y,
                                              filter_x)];
                reference_sum += static_cast<double>(input_value) *
                                 static_cast<double>(weight_value);
              }
            }
          }

          const float reference = static_cast<float>(reference_sum);
          const float gpu_value =
              gpu_output[output_offset(shape, batch, output_channel, output_y,
                                       output_x)];
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
  }

  std::cout << "Validation shape: N=" << shape.batch_size
            << " C=" << shape.input_channels << " H=" << shape.height
            << " W=" << shape.width << " K=" << shape.output_channels
            << '\n';
  std::cout << "Tolerance violations: " << tolerance_violations << " / "
            << result_elements << '\n';
  std::cout << "Non-finite outputs: " << non_finite_outputs << " / "
            << result_elements << '\n';
  std::cout << "Max absolute error: " << max_absolute_error << '\n';
  std::cout << "Max relative error: " << max_relative_error << '\n';
  if (tolerance_violations != 0 || non_finite_outputs != 0) {
    throw std::runtime_error("FP32 Conv2D validation failed");
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
    if (argc > 9) {
      throw std::invalid_argument(
          "usage: conv2d.exe [N] [C] [H] [W] [K] [iterations] "
          "[validation_size] [seed]");
    }

    const Conv2dShape shape{
        argc > 1 ? positive_int(argv[1], "N") : 1,
        argc > 2 ? positive_int(argv[2], "C") : 32,
        argc > 3 ? positive_int(argv[3], "H") : 64,
        argc > 4 ? positive_int(argv[4], "W") : 64,
        argc > 5 ? positive_int(argv[5], "K") : 32};
    const int iterations = argc > 6 ? positive_int(argv[6], "iterations") : 100;
    const int validation_size =
        argc > 7 ? positive_int(argv[7], "validation_size") : 8;
    const std::uint32_t seed = argc > 8 ? seed_value(argv[8]) : 12345U;

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    const HostData data = make_data(shape, seed);
    const std::size_t result_elements = output_elements(shape);
    DeviceFloatBuffer device_input(data.input.size());
    DeviceFloatBuffer device_weight(data.weight.size());
    DeviceFloatBuffer device_output(result_elements);
    copy_data_to_device(data, device_input, device_weight);

    const float elapsed_us = benchmark_us(
        device_input.get(), device_weight.get(), device_output.get(), shape,
        iterations);
    const double operation_count =
        2.0 * static_cast<double>(result_elements) *
        static_cast<double>(shape.input_channels) * kFilterSize * kFilterSize;
    const double throughput_gflops =
        operation_count / (static_cast<double>(elapsed_us) * 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Conv2D shape: N=" << shape.batch_size
              << " C=" << shape.input_channels << " H=" << shape.height
              << " W=" << shape.width << " K=" << shape.output_channels
              << " R=3 S=3 stride=1 padding=1\n";
    std::cout << "Naive direct Conv2D: " << elapsed_us << " us\n";
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
