from __future__ import annotations

from dataclasses import dataclass

from ...contracts import ImplementationSpec, WorkloadSpec
from .base import CUDAImplementation


@dataclass(frozen=True)
class GeneratedNaivePlan:
    m: int
    n: int
    k: int
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    threads: int = 256


class GeneratedNaive(CUDAImplementation):
    """Adapter identity for the frozen v0.20 generated CUDA baseline."""

    spec = ImplementationSpec(
        name="generated_naive",
        backend="cuda",
        source="aicf_codegen_v0_20",
        configuration={"policy": "fixed_baseline"},
    )

    def supports(self, workload: WorkloadSpec) -> bool:
        return workload.kind in {"gemm", "gemm_bias_relu"}

    def plan(self, workload: WorkloadSpec) -> GeneratedNaivePlan:
        if not self.supports(workload):
            raise ValueError(f"generated_naive does not support {workload.kind}")

        x, weight = workload.inputs[:2]
        if len(x.shape) != 2 or len(weight.shape) != 2:
            raise ValueError("generated_naive GEMM expects rank-2 X and weight")

        m, k = x.shape
        wk, n = weight.shape
        if k != wk:
            raise ValueError(f"GEMM K mismatch: {x.shape} @ {weight.shape}")

        return GeneratedNaivePlan(m=m, n=n, k=k)
