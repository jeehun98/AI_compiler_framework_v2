from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...ir.module import IRModule


@dataclass(frozen=True)
class GEMMProblem:
    """Target-side description of a 2D GEMM problem.

    This describes *what* matrix multiplication must be computed. It is kept
    separate from the execution tile and schedule so later stages can compare
    different execution mappings for the same mathematical problem.
    """

    m: int
    n: int
    k: int
    dtype: str


@dataclass(frozen=True)
class GEMMTile:
    """Logical output/reduction tile used to partition a GEMM problem.

    `block_m` x `block_n` is the output region assigned to one logical tile.
    `block_k` is the reduction-depth chunk consumed per K iteration.

    This is still a logical decomposition. It does not yet define CUDA block
    dimensions, warp layout, thread mapping, memory layout or predicates.
    """

    block_m: int
    block_n: int
    block_k: int


@dataclass(frozen=True)
class GEMMSchedule:
    """Logical traversal schedule induced by a GEMM problem and tile.

    `grid_m` and `grid_n` count the output tiles required to cover M and N.
    `k_tiles` counts the reduction chunks required to cover K.

    Despite the word "grid", this is not yet a physical CUDA launch config.
    A later execution-mapping stage connects these logical tile counts to
    blockIdx/threadIdx and concrete dim3 launch dimensions.
    """

    grid_m: int
    grid_n: int
    k_tiles: int


@dataclass(frozen=True)
class GEMMBlockMapping:
    """Minimal physical CUDA thread-block allocation for one GEMM tile.

    `threads` is the number of CUDA threads assigned to one logical output
    tile. `warps` is the corresponding warp count for the current CUDA warp
    size. `outputs_per_thread` is the maximum number of logical C elements
    each thread must visit when the tile is traversed with a simple strided
    loop.

    This object answers *how many physical workers are assigned*. The exact
    threadIdx/blockIdx -> matrix-coordinate relation is represented separately
    by `GEMMThreadMapping`.
    """

    threads: int
    warps: int
    outputs_per_thread: int


@dataclass(frozen=True)
class GEMMThreadMapping:
    """Explicit coordinate convention for mapping CUDA indices to GEMM C.

    v0.13 defines the first concrete mapping contract:

        output = threadIdx.x + iteration * blockDim.x
        local_m = output // BN
        local_n = output % BN
        row = blockIdx.y * BM + local_m
        col = blockIdx.x * BN + local_n

    The mapping is deliberately simple and row-major. It still does not add
    boundary predicates, the GEMM K-loop, memory planning or actual CUDA source
    emission; those are later stages.
    """

    traversal: str
    output_order: str
    thread_axis: str
    block_m_axis: str
    block_n_axis: str


@dataclass(frozen=True)
class GEMMControlFlow:
    """Control-flow policy for the first naive GEMM kernel body.

    `output_traversal` describes how threads walk the logical output tile.
    `output_guard` records whether row/column bounds must be checked because
    M or N is not exactly covered by whole output tiles.

    `k_traversal` describes the reduction as an outer K-tile loop plus an
    inner BK loop. `k_tail_guard` records whether the final K tile requires
    an explicit `k < K` predicate.

    This still does not describe memory movement or the multiply-accumulate
    expression itself; v0.14 only makes the kernel control flow explicit.
    """

    output_traversal: str
    output_guard: bool
    k_traversal: str
    k_tail_guard: bool


@dataclass(frozen=True)
class GEMMEpilogue:
    """Output transformation applied after GEMM accumulation.

    v0.15 keeps this intentionally small: a GEMM may optionally add a bias
    vector and optionally apply one named activation. Keeping the epilogue in
    the lowering plan means backend codegen does not need to infer semantics
    from an operation name or strategy string.
    """

    bias: bool
    activation: str | None = None


@dataclass
class CUDAKernelPlan:
    """Minimal target-side representation of one planned CUDA kernel.

    `problem` describes what GEMM must be computed. `tile` describes how the
    problem is partitioned into logical output/reduction tiles. `schedule`
    records how many such tiles are needed to cover the full problem.
    `block_mapping` assigns a physical CUDA thread count to each output tile.
    `thread_mapping` defines how those CUDA indices correspond to row-major
    output coordinates. `control_flow` records whether output/K-tail
    predicates are required and how the K reduction is traversed. `epilogue`
    records the output transformation after accumulation.

    v0.15's backend uses these fields to emit a first naive global-memory CUDA
    kernel. Shared-memory/register tiling and vectorized memory movement remain
    intentionally deferred.
    """

    name: str
    source_op: str
    strategy: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    problem: GEMMProblem | None = None
    tile: GEMMTile | None = None
    schedule: GEMMSchedule | None = None
    block_mapping: GEMMBlockMapping | None = None
    thread_mapping: GEMMThreadMapping | None = None
    control_flow: GEMMControlFlow | None = None
    epilogue: GEMMEpilogue | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDALoweredModule:
    """CUDA-target lowering result composed of kernel plans."""

    source_ir: IRModule
    kernels: list[CUDAKernelPlan]