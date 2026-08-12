from __future__ import annotations

from ...contracts import ImplementationSpec, WorkloadSpec
from ..kernels import KernelSpec, generated_naive_gemm_bias_relu
from .base import CUDAImplementation


class GeneratedNaive(CUDAImplementation):
    """Frozen v0.20 CUDA baseline exposed as one backend implementation."""

    spec = ImplementationSpec(
        name="generated_naive_v020",
        backend="cuda",
        source="frozen_cuda_kernel",
        configuration={"role": "observation_baseline"},
    )

    def supports(self, workload: WorkloadSpec) -> bool:
        if workload.kind != "gemm_bias_relu" or len(workload.inputs) < 3:
            return False
        x, weight, bias = workload.inputs[:3]
        return (
            x.shape == (32, 64)
            and weight.shape == (64, 128)
            and bias.shape == (128,)
        )

    def plan(self, workload: WorkloadSpec) -> KernelSpec:
        if not self.supports(workload):
            raise ValueError("generated_naive_v020 supports only the frozen M32/N128/K64 baseline")
        return generated_naive_gemm_bias_relu()
