from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...ir.module import IRModule


@dataclass(frozen=True)
class GEMMProblem:
    """Target-side description of a 2D GEMM problem.

    This describes *what* matrix multiplication must be computed. It does not
    yet describe *how* CUDA should execute it; tiling, launch geometry and
    memory strategy are intentionally deferred to later lowering stages.
    """

    m: int
    n: int
    k: int
    dtype: str


@dataclass
class CUDAKernelPlan:
    """Minimal target-side representation of one planned CUDA kernel.

    The plan records the lowering decision that the backend will later
    materialize as code. `problem` describes the mathematical workload when
    the kernel is GEMM-based; execution details such as tile and block shapes
    are intentionally not represented yet.
    """

    name: str
    source_op: str
    strategy: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    problem: GEMMProblem | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDALoweredModule:
    """CUDA-target lowering result composed of kernel plans."""

    source_ir: IRModule
    kernels: list[CUDAKernelPlan]