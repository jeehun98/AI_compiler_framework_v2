#pragma once

#include <cuda_fp16.h>

#include <cstdint>

void launch_fma_validation_kernels(const half* a, const half* b, const half* c,
                                   half* temporary, half* separated_output,
                                   half* fused_output, int element_count);

void run_fma_validation(int validation_elements, std::uint32_t seed);
