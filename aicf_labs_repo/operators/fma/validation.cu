#include "validation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kAbsoluteTolerance = 1.0 / 512.0;  // 2^-9: about two FP16 ULPs near 1.
constexpr double kRelativeTolerance = 1.0 / 512.0;
constexpr double kRelativeEpsilon = 1.0 / 16384.0;  // FP16 minimum normal, 2^-14.

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                     \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

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

std::uint16_t half_bits(half value) {
  std::uint16_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

half half_from_bits(std::uint16_t bits) {
  half value{};
  std::memcpy(&value, &bits, sizeof(bits));
  return value;
}

half quantize_to_half(float value) { return __float2half_rn(value); }

double half_to_double(half value) {
  return static_cast<double>(__half2float(value));
}

std::string bit_string(half value) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::setw(4) << std::setfill('0')
         << half_bits(value);
  return stream.str();
}

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

HostInputs cancellation_inputs(std::size_t element_count,
                               std::uint32_t seed) {
  HostInputs inputs(element_count);
  std::mt19937 generator(seed ^ 0x9e3779b9U);
  std::uniform_real_distribution<float> magnitude(0.5F, 2.0F);
  for (std::size_t i = 0; i < element_count; ++i) {
    const float sign_a = (generator() & 1U) ? 1.0F : -1.0F;
    const float sign_b = (generator() & 1U) ? 1.0F : -1.0F;
    inputs.a[i] = quantize_to_half(sign_a * magnitude(generator));
    inputs.b[i] = quantize_to_half(sign_b * magnitude(generator));

    // c is the exact sign-negation of the FP16-rounded product. Separated
    // Mul/Add therefore cancels that rounded value, while FMA retains the
    // product residue before its single final rounding.
    const half rounded_product = quantize_to_half(static_cast<float>(
        half_to_double(inputs.a[i]) * half_to_double(inputs.b[i])));
    inputs.c[i] = half_from_bits(half_bits(rounded_product) ^ 0x8000U);
  }
  return inputs;
}

HostInputs range_stress_inputs(std::size_t element_count) {
  struct Pattern {
    std::uint16_t a;
    std::uint16_t b;
    std::uint16_t c;
  };
  constexpr std::array<Pattern, 12> patterns = {{
      {0x7bffU, 0x4000U, 0xfbffU},  // max * 2 - max: intermediate overflow.
      {0xfbffU, 0x4000U, 0x7bffU},
      {0x6800U, 0x5000U, 0xfbffU},  // 2048 * 32 - 65504.
      {0x0001U, 0x3800U, 0x0000U},  // minimum subnormal * 0.5.
      {0x0001U, 0x3800U, 0x0001U},
      {0x03ffU, 0x3800U, 0x8001U},
      {0x0400U, 0x3800U, 0x8001U},
      {0x7bffU, 0x0001U, 0x0000U},
      {0x0000U, 0x7bffU, 0x8000U},
      {0x8000U, 0x7bffU, 0x0000U},
      {0x7bffU, 0x0000U, 0x0001U},
      {0x0001U, 0x7bffU, 0x8001U},
  }};

  HostInputs inputs(element_count);
  for (std::size_t i = 0; i < element_count; ++i) {
    const Pattern& pattern = patterns[i % patterns.size()];
    inputs.a[i] = half_from_bits(pattern.a);
    inputs.b[i] = half_from_bits(pattern.b);
    inputs.c[i] = half_from_bits(pattern.c);
  }
  return inputs;
}

HostInputs special_inputs(std::size_t element_count) {
  struct Pattern {
    std::uint16_t a;
    std::uint16_t b;
    std::uint16_t c;
  };
  constexpr std::array<Pattern, 12> patterns = {{
      {0x0000U, 0x3c00U, 0x8000U},  // +0 * 1 + -0.
      {0x8000U, 0x3c00U, 0x0000U},  // -0 * 1 + +0.
      {0x8000U, 0x3c00U, 0x8000U},  // -0 * 1 + -0.
      {0x7c00U, 0x3c00U, 0x0000U},  // +Inf.
      {0xfc00U, 0x3c00U, 0x0000U},  // -Inf.
      {0x7c00U, 0x0000U, 0x3c00U},  // Inf * 0 -> NaN.
      {0x7c00U, 0x3c00U, 0xfc00U},  // Inf + -Inf -> NaN.
      {0x7e00U, 0x3c00U, 0x4000U},  // quiet NaN input.
      {0xfe00U, 0x3c00U, 0xc000U},  // negative quiet NaN input.
      {0x3c00U, 0x7c00U, 0xfc00U},
      {0x0000U, 0xfc00U, 0x0000U},
      {0x8000U, 0xfc00U, 0x8000U},
  }};

  HostInputs inputs(element_count);
  for (std::size_t i = 0; i < element_count; ++i) {
    const Pattern& pattern = patterns[i % patterns.size()];
    inputs.a[i] = half_from_bits(pattern.a);
    inputs.b[i] = half_from_bits(pattern.b);
    inputs.c[i] = half_from_bits(pattern.c);
  }
  return inputs;
}


struct DifferenceSample {
  std::size_t index = 0;
  std::string difference_type;
  half a{};
  half b{};
  half c{};
  half separated{};
  half fused{};
  double reference = 0.0;
  double separated_error = 0.0;
  double fused_error = 0.0;
};

struct ValidationStats {
  std::size_t total = 0;
  std::size_t bitwise_equal = 0;
  std::size_t bitwise_different = 0;
  std::size_t finite_pairs = 0;
  std::size_t finite_numerical_divergence = 0;
  std::size_t finite_tolerance_comparisons = 0;
  std::size_t exact_numerical_equal = 0;
  double max_absolute_difference = 0.0;
  double max_relative_difference = 0.0;
  double absolute_difference_sum = 0.0;
  std::size_t finite_difference_count = 0;
  std::size_t separated_nan = 0;
  std::size_t fused_nan = 0;
  std::size_t separated_inf = 0;
  std::size_t fused_inf = 0;
  std::size_t signed_zero_differences = 0;
  std::size_t classification_mismatches = 0;
  std::size_t finite_nonfinite_mismatches = 0;
  std::size_t nan_classification_mismatches = 0;
  std::size_t inf_classification_mismatches = 0;
  std::size_t inf_sign_mismatches = 0;
  std::size_t separated_inf_fused_finite = 0;
  std::size_t fused_inf_separated_finite = 0;
  std::size_t tolerance_exceeded = 0;
  double separated_reference_max_error = 0.0;
  double fused_reference_max_error = 0.0;
  double separated_reference_error_sum = 0.0;
  double fused_reference_error_sum = 0.0;
  std::size_t separated_reference_finite_count = 0;
  std::size_t fused_reference_finite_count = 0;
  std::size_t separated_closer_to_reference = 0;
  std::size_t fused_closer_to_reference = 0;
  std::size_t equal_reference_error = 0;
  std::size_t reference_comparable = 0;
  std::vector<DifferenceSample> samples;
};

enum class ResultClass { kFinite, kInfinity, kNaN };

ResultClass result_class(double value) {
  if (std::isnan(value)) {
    return ResultClass::kNaN;
  }
  if (std::isinf(value)) {
    return ResultClass::kInfinity;
  }
  return ResultClass::kFinite;
}

std::string difference_type(double separated, double fused) {
  const bool separated_finite = std::isfinite(separated);
  const bool fused_finite = std::isfinite(fused);
  if (separated_finite && fused_finite) {
    if (separated == 0.0 && fused == 0.0 &&
        std::signbit(separated) != std::signbit(fused)) {
      return "signed_zero_difference";
    }
    return "finite_rounding_difference";
  }
  if (std::isinf(separated) && fused_finite) {
    return "separated_overflow_fused_finite";
  }
  if (separated_finite && std::isinf(fused)) {
    return "fused_overflow_separated_finite";
  }
  if (std::isnan(separated) != std::isnan(fused)) {
    return "nan_classification_difference";
  }
  if (std::isinf(separated) && std::isinf(fused) &&
      std::signbit(separated) != std::signbit(fused)) {
    return "infinity_sign_difference";
  }
  if (std::isnan(separated) && std::isnan(fused)) {
    return "nan_payload_difference";
  }
  return "classification_difference";
}

double output_reference_error(double output, double reference) {
  if (std::isfinite(output) && std::isfinite(reference)) {
    return std::abs(output - reference);
  }
  if (std::isnan(output) && std::isnan(reference)) {
    return 0.0;
  }
  if (std::isinf(output) && std::isinf(reference) &&
      std::signbit(output) == std::signbit(reference)) {
    return 0.0;
  }
  return std::numeric_limits<double>::infinity();
}

ValidationStats compare_results(const HostInputs& inputs,
                                const std::vector<half>& separated,
                                const std::vector<half>& fused) {
  ValidationStats stats;
  stats.total = separated.size();
  for (std::size_t i = 0; i < separated.size(); ++i) {
    const std::uint16_t separated_bits = half_bits(separated[i]);
    const std::uint16_t fused_bits = half_bits(fused[i]);
    if (separated_bits == fused_bits) {
      ++stats.bitwise_equal;
    } else {
      ++stats.bitwise_different;
    }

    const double a = half_to_double(inputs.a[i]);
    const double b = half_to_double(inputs.b[i]);
    const double c = half_to_double(inputs.c[i]);
    const double separated_value = half_to_double(separated[i]);
    const double fused_value = half_to_double(fused[i]);
    const double reference = a * b + c;

    const bool separated_is_nan = std::isnan(separated_value);
    const bool fused_is_nan = std::isnan(fused_value);
    const bool separated_is_inf = std::isinf(separated_value);
    const bool fused_is_inf = std::isinf(fused_value);
    const bool separated_is_finite = std::isfinite(separated_value);
    const bool fused_is_finite = std::isfinite(fused_value);
    const ResultClass separated_class = result_class(separated_value);
    const ResultClass fused_class = result_class(fused_value);

    stats.separated_nan += separated_is_nan;
    stats.fused_nan += fused_is_nan;
    stats.separated_inf += separated_is_inf;
    stats.fused_inf += fused_is_inf;
    stats.nan_classification_mismatches += separated_is_nan != fused_is_nan;
    stats.inf_classification_mismatches += separated_is_inf != fused_is_inf;
    stats.classification_mismatches += separated_class != fused_class;
    stats.finite_nonfinite_mismatches +=
        separated_is_finite != fused_is_finite;
    stats.separated_inf_fused_finite +=
        separated_is_inf && fused_is_finite;
    stats.fused_inf_separated_finite +=
        fused_is_inf && separated_is_finite;
    if (separated_is_inf && fused_is_inf &&
        std::signbit(separated_value) != std::signbit(fused_value)) {
      ++stats.inf_sign_mismatches;
    }

    if (separated_value == fused_value) {
      ++stats.exact_numerical_equal;
    }
    if (separated_value == 0.0 && fused_value == 0.0 &&
        std::signbit(separated_value) != std::signbit(fused_value)) {
      ++stats.signed_zero_differences;
    }

    if (separated_is_finite && fused_is_finite) {
      ++stats.finite_pairs;
      if (separated_value != fused_value) {
        ++stats.finite_numerical_divergence;
      }
      const double absolute_difference =
          std::abs(separated_value - fused_value);
      const double denominator =
          std::max(std::abs(reference), kRelativeEpsilon);
      const double relative_difference = absolute_difference / denominator;
      stats.max_absolute_difference =
          std::max(stats.max_absolute_difference, absolute_difference);
      stats.max_relative_difference =
          std::max(stats.max_relative_difference, relative_difference);
      stats.absolute_difference_sum += absolute_difference;
      ++stats.finite_difference_count;
      if (std::isfinite(reference)) {
        ++stats.finite_tolerance_comparisons;
        const double allowed =
            kAbsoluteTolerance + kRelativeTolerance * std::abs(reference);
        if (absolute_difference > allowed) {
          ++stats.tolerance_exceeded;
        }
      }
    }

    const double separated_reference_error =
        output_reference_error(separated_value, reference);
    const double fused_reference_error =
        output_reference_error(fused_value, reference);
    if (separated_is_finite && std::isfinite(reference)) {
      stats.separated_reference_max_error = std::max(
          stats.separated_reference_max_error, separated_reference_error);
      stats.separated_reference_error_sum += separated_reference_error;
      ++stats.separated_reference_finite_count;
    }
    if (fused_is_finite && std::isfinite(reference)) {
      stats.fused_reference_max_error =
          std::max(stats.fused_reference_max_error, fused_reference_error);
      stats.fused_reference_error_sum += fused_reference_error;
      ++stats.fused_reference_finite_count;
    }

    if (std::isfinite(reference)) {
      ++stats.reference_comparable;
      if (separated_reference_error < fused_reference_error) {
        ++stats.separated_closer_to_reference;
      } else if (fused_reference_error < separated_reference_error) {
        ++stats.fused_closer_to_reference;
      } else {
        ++stats.equal_reference_error;
      }
    }

    if (separated_bits != fused_bits && stats.samples.size() < 10) {
      stats.samples.push_back(DifferenceSample{
          i,
          difference_type(separated_value, fused_value),
          inputs.a[i],
          inputs.b[i],
          inputs.c[i],
          separated[i],
          fused[i],
          reference,
          separated_reference_error,
          fused_reference_error,
      });
    }
  }
  return stats;
}

std::string format_count(std::size_t count) {
  std::string text = std::to_string(count);
  for (std::size_t position = text.size(); position > 3;) {
    position -= 3;
    text.insert(position, ",");
  }
  return text;
}

double percentage(std::size_t count, std::size_t denominator) {
  if (denominator == 0) {
    return 0.0;
  }
  return 100.0 * static_cast<double>(count) /
         static_cast<double>(denominator);
}

std::string count_with_rate(std::size_t count, std::size_t denominator) {
  std::ostringstream stream;
  stream << format_count(count) << " / " << format_count(denominator) << " ("
         << std::fixed << std::setprecision(2)
         << percentage(count, denominator) << "%)";
  return stream.str();
}

void print_mean(const char* label, double sum, std::size_t count) {
  std::cout << "  " << label << ": ";
  if (count == 0) {
    std::cout << "n/a\n";
  } else {
    std::cout << (sum / static_cast<double>(count)) << '\n';
  }
}

void print_validation(const std::string& name, const ValidationStats& stats) {
  std::cout << "\nValidation case: " << name << '\n';
  std::cout << "  Elements: " << format_count(stats.total) << '\n';
  std::cout << "  Bitwise equal / all elements: "
            << count_with_rate(stats.bitwise_equal, stats.total) << '\n';
  std::cout << "  Bitwise divergence / all elements: "
            << count_with_rate(stats.bitwise_different, stats.total) << '\n';
  std::cout << "  Jointly finite / all elements: "
            << count_with_rate(stats.finite_pairs, stats.total) << '\n';
  std::cout << "  Finite numerical divergence / jointly finite: "
            << count_with_rate(stats.finite_numerical_divergence,
                               stats.finite_pairs)
            << '\n';
  std::cout << "  Exact numerical equal / all elements: "
            << count_with_rate(stats.exact_numerical_equal, stats.total)
            << '\n';
  std::cout << "  Max |separated - fused|: "
            << stats.max_absolute_difference << '\n';
  std::cout << "  Max relative difference: "
            << stats.max_relative_difference << '\n';
  print_mean("Mean |separated - fused|", stats.absolute_difference_sum,
             stats.finite_difference_count);
  std::cout << "  Separated reference max error: "
            << stats.separated_reference_max_error << '\n';
  print_mean("Separated reference mean error",
             stats.separated_reference_error_sum,
             stats.separated_reference_finite_count);
  std::cout << "  Fused reference max error: "
            << stats.fused_reference_max_error << '\n';
  print_mean("Fused reference mean error", stats.fused_reference_error_sum,
             stats.fused_reference_finite_count);
  std::cout << "  Reference-comparable / all elements: "
            << count_with_rate(stats.reference_comparable, stats.total)
            << '\n';
  std::cout << "  Separated closer / reference-comparable: "
            << count_with_rate(stats.separated_closer_to_reference,
                               stats.reference_comparable)
            << '\n';
  std::cout << "  Fused closer / reference-comparable: "
            << count_with_rate(stats.fused_closer_to_reference,
                               stats.reference_comparable)
            << '\n';
  std::cout << "  Equal reference error / reference-comparable: "
            << count_with_rate(stats.equal_reference_error,
                               stats.reference_comparable)
            << '\n';
  std::cout << "  NaN count separated / all elements: "
            << count_with_rate(stats.separated_nan, stats.total) << '\n';
  std::cout << "  NaN count fused / all elements: "
            << count_with_rate(stats.fused_nan, stats.total) << '\n';
  std::cout << "  Inf count separated / all elements: "
            << count_with_rate(stats.separated_inf, stats.total) << '\n';
  std::cout << "  Inf count fused / all elements: "
            << count_with_rate(stats.fused_inf, stats.total) << '\n';
  std::cout << "  Classification divergence / all elements: "
            << count_with_rate(stats.classification_mismatches, stats.total)
            << '\n';
  std::cout << "  Finite/non-finite classification divergence / all elements: "
            << count_with_rate(stats.finite_nonfinite_mismatches, stats.total)
            << '\n';
  std::cout << "  NaN classification divergence / all elements: "
            << count_with_rate(stats.nan_classification_mismatches,
                               stats.total)
            << '\n';
  std::cout << "  Inf classification divergence / all elements: "
            << count_with_rate(stats.inf_classification_mismatches,
                               stats.total)
            << '\n';
  std::cout << "  Signed-zero divergence / all elements: "
            << count_with_rate(stats.signed_zero_differences, stats.total)
            << '\n';
  std::cout << "  Finite tolerance violations / jointly finite comparable: "
            << count_with_rate(stats.tolerance_exceeded,
                               stats.finite_tolerance_comparisons)
            << '\n';
  std::cout << "  Observation status: COMPLETED\n";

  if (!stats.samples.empty()) {
    std::cout << "  Different-result samples (up to 10):\n";
  }
  for (const DifferenceSample& sample : stats.samples) {
    std::cout << "    index " << sample.index << '\n';
    std::cout << "      difference type=" << sample.difference_type << '\n';
    std::cout << "      a=" << half_to_double(sample.a)
              << " bits=" << bit_string(sample.a) << '\n';
    std::cout << "      b=" << half_to_double(sample.b)
              << " bits=" << bit_string(sample.b) << '\n';
    std::cout << "      c=" << half_to_double(sample.c)
              << " bits=" << bit_string(sample.c) << '\n';
    std::cout << "      separated=" << half_to_double(sample.separated)
              << " bits=" << bit_string(sample.separated) << '\n';
    std::cout << "      fused=" << half_to_double(sample.fused)
              << " bits=" << bit_string(sample.fused) << '\n';
    std::cout << "      double reference=" << sample.reference << '\n';
    std::cout << "      separated absolute error="
              << sample.separated_error << '\n';
    std::cout << "      fused absolute error=" << sample.fused_error << '\n';
  }
}

struct NumericalSummary {
  std::size_t total = 0;
  std::size_t bitwise_equal = 0;
  std::size_t bitwise_different = 0;
  std::size_t finite_pairs = 0;
  std::size_t finite_numerical_divergence = 0;
  std::size_t finite_tolerance_comparisons = 0;
  std::size_t tolerance_exceeded = 0;
  std::size_t classification_mismatches = 0;
  std::size_t finite_nonfinite_mismatches = 0;
  std::size_t nan_classification_mismatches = 0;
  std::size_t inf_classification_mismatches = 0;
  std::size_t signed_zero_differences = 0;
  std::size_t reference_comparable = 0;
  std::size_t separated_closer = 0;
  std::size_t fused_closer = 0;
  std::size_t equal_reference_error = 0;

  void add(const ValidationStats& stats) {
    total += stats.total;
    bitwise_equal += stats.bitwise_equal;
    bitwise_different += stats.bitwise_different;
    finite_pairs += stats.finite_pairs;
    finite_numerical_divergence += stats.finite_numerical_divergence;
    finite_tolerance_comparisons += stats.finite_tolerance_comparisons;
    tolerance_exceeded += stats.tolerance_exceeded;
    classification_mismatches += stats.classification_mismatches;
    finite_nonfinite_mismatches += stats.finite_nonfinite_mismatches;
    nan_classification_mismatches +=
        stats.nan_classification_mismatches;
    inf_classification_mismatches +=
        stats.inf_classification_mismatches;
    signed_zero_differences += stats.signed_zero_differences;
    reference_comparable += stats.reference_comparable;
    separated_closer += stats.separated_closer_to_reference;
    fused_closer += stats.fused_closer_to_reference;
    equal_reference_error += stats.equal_reference_error;
  }
};

void print_numerical_summary(const NumericalSummary& summary) {
  std::cout << "\nNumerical comparison summary\n\n";
  std::cout << "  Total elements: " << format_count(summary.total) << '\n';
  std::cout << "  Bitwise equal: "
            << count_with_rate(summary.bitwise_equal, summary.total) << '\n';
  std::cout << "  Bitwise different: "
            << count_with_rate(summary.bitwise_different, summary.total)
            << '\n';
  std::cout << "  Jointly finite comparisons: "
            << count_with_rate(summary.finite_pairs, summary.total) << '\n';
  std::cout << "  Finite numerical divergence: "
            << count_with_rate(summary.finite_numerical_divergence,
                               summary.finite_pairs)
            << '\n';
  std::cout << "  Finite tolerance violations: "
            << count_with_rate(summary.tolerance_exceeded,
                               summary.finite_tolerance_comparisons)
            << '\n';
  std::cout << "  Classification mismatches: "
            << count_with_rate(summary.classification_mismatches,
                               summary.total)
            << '\n';
  std::cout << "  Finite/non-finite classification divergence: "
            << count_with_rate(summary.finite_nonfinite_mismatches,
                               summary.total)
            << '\n';
  std::cout << "  NaN classification divergence: "
            << count_with_rate(summary.nan_classification_mismatches,
                               summary.total)
            << '\n';
  std::cout << "  Inf classification divergence: "
            << count_with_rate(summary.inf_classification_mismatches,
                               summary.total)
            << '\n';
  std::cout << "  Signed-zero differences: "
            << count_with_rate(summary.signed_zero_differences, summary.total)
            << '\n';
  std::cout << "  Reference-comparable elements: "
            << count_with_rate(summary.reference_comparable, summary.total)
            << '\n';
  std::cout << "  Separated closer: "
            << count_with_rate(summary.separated_closer,
                               summary.reference_comparable)
            << '\n';
  std::cout << "  Fused closer: "
            << count_with_rate(summary.fused_closer,
                               summary.reference_comparable)
            << '\n';
  std::cout << "  Equal reference error: "
            << count_with_rate(summary.equal_reference_error,
                               summary.reference_comparable)
            << '\n';

  const bool strict_bitwise_accepted = summary.bitwise_different == 0;
  const bool finite_tolerance_accepted = summary.tolerance_exceeded == 0;
  const bool classification_preserved =
      summary.classification_mismatches == 0;
  const bool accuracy_favorable = summary.reference_comparable != 0 &&
                                  summary.separated_closer == 0 &&
                                  summary.tolerance_exceeded == 0;

  std::cout << "\nPolicy verdicts\n\n";
  std::cout << "  strict_bitwise: "
            << (strict_bitwise_accepted ? "ACCEPTED" : "REJECTED") << '\n';
  std::cout << "    reason: "
            << count_with_rate(summary.bitwise_different, summary.total)
            << " bitwise divergences observed\n";
  std::cout << "  finite_tolerance: "
            << (finite_tolerance_accepted
                    ? "ACCEPTED_FOR_JOINTLY_FINITE_RESULTS"
                    : "REJECTED_FOR_JOINTLY_FINITE_RESULTS")
            << '\n';
  std::cout << "    reason: "
            << count_with_rate(summary.tolerance_exceeded,
                               summary.finite_tolerance_comparisons)
            << " jointly-finite results exceeded tolerance\n";
  std::cout << "  classification_preserving: "
            << (classification_preserved ? "ACCEPTED" : "REJECTED")
            << '\n';
  std::cout << "    reason: "
            << count_with_rate(summary.classification_mismatches,
                               summary.total)
            << " classification divergences observed\n";
  std::cout << "  accuracy_oriented_contraction: ";
  if (summary.reference_comparable == 0) {
    std::cout << "INCONCLUSIVE\n";
  } else {
    std::cout << (accuracy_favorable ? "FAVORABLE" : "NOT_FAVORABLE")
              << '\n';
  }
  std::cout << "    reason: fused closer "
            << count_with_rate(summary.fused_closer,
                               summary.reference_comparable)
            << ", equal "
            << count_with_rate(summary.equal_reference_error,
                               summary.reference_comparable)
            << ", fused worse "
            << count_with_rate(summary.separated_closer,
                               summary.reference_comparable)
            << " on this input set\n";
}

void print_range_stress_observation(const ValidationStats& stats) {
  std::cout << "\nRange-stress observation\n";
  std::cout << "  separated intermediate overflowed to Inf while fused FMA "
               "retained a finite final result: "
            << count_with_rate(stats.separated_inf_fused_finite, stats.total)
            << '\n';
  std::cout << "  classification divergence: "
            << count_with_rate(stats.classification_mismatches, stats.total)
            << '\n';
}

ValidationStats run_validation_case(const HostInputs& inputs,
                                    DeviceHalfBuffer& device_a,
                                    DeviceHalfBuffer& device_b,
                                    DeviceHalfBuffer& device_c,
                                    DeviceHalfBuffer& temporary,
                                    DeviceHalfBuffer& separated_output,
                                    DeviceHalfBuffer& fused_output) {
  const std::size_t element_count = inputs.a.size();
  const std::size_t bytes = element_count * sizeof(half);
  CUDA_CHECK(cudaMemcpy(device_a.get(), inputs.a.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b.get(), inputs.b.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_c.get(), inputs.c.data(), bytes,
                        cudaMemcpyHostToDevice));

  launch_fma_validation_kernels(
      device_a.get(), device_b.get(), device_c.get(), temporary.get(),
      separated_output.get(), fused_output.get(),
      static_cast<int>(element_count));


  std::vector<half> host_separated(element_count);
  std::vector<half> host_fused(element_count);
  CUDA_CHECK(cudaMemcpy(host_separated.data(), separated_output.get(), bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_fused.data(), fused_output.get(), bytes,
                        cudaMemcpyDeviceToHost));
  return compare_results(inputs, host_separated, host_fused);
}

}  // namespace

void run_fma_validation(int validation_elements, std::uint32_t seed) {
    std::cout << "Validation elements:  " << validation_elements << '\n';
    std::cout << "Validation seed:      " << seed << '\n';
    std::cout << "Finite tolerance: |separated - fused| <= "
              << kAbsoluteTolerance << " + " << kRelativeTolerance
              << " * |double reference|\n";
    std::cout << "Relative error denominator: max(|double reference|, "
              << kRelativeEpsilon << ")\n";

    DeviceHalfBuffer validation_a(validation_elements);
    DeviceHalfBuffer validation_b(validation_elements);
    DeviceHalfBuffer validation_c(validation_elements);
    DeviceHalfBuffer validation_temporary(validation_elements);
    DeviceHalfBuffer validation_separated(validation_elements);
    DeviceHalfBuffer validation_fused(validation_elements);

    const auto validate = [&](const std::string& name, HostInputs inputs) {
      const ValidationStats stats = run_validation_case(
          inputs, validation_a, validation_b, validation_c,
          validation_temporary, validation_separated, validation_fused);
      print_validation(name, stats);
      return stats;
    };

    const ValidationStats ordinary =
        validate("ordinary", ordinary_inputs(validation_elements, seed));
    const ValidationStats cancellation = validate(
        "cancellation", cancellation_inputs(validation_elements, seed));
    const ValidationStats range_stress =
        validate("range_stress", range_stress_inputs(validation_elements));
    const ValidationStats special =
        validate("special", special_inputs(validation_elements));

    NumericalSummary summary;
    summary.add(ordinary);
    summary.add(cancellation);
    summary.add(range_stress);
    summary.add(special);
    print_numerical_summary(summary);
    print_range_stress_observation(range_stress);

    std::cout << "\nObservation completed successfully.\n";
    std::cout << "Policy rejection records a numerical-semantics decision, "
                 "not an execution failure.\n";
}
