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
    A later execution-mapping stage will connect these logical tile counts to
    blockIdx/threadIdx and concrete dim3 launch dimensions.
    """

    grid_m: int
    grid_n: int
    k_tiles: int


@dataclass(frozen=True)
class GEMMBlockMapping:
    """Minimal physical CUDA thread-block mapping for one logical GEMM tile.

    `threads` is the number of CUDA threads assigned to one logical output
    tile. `warps` is the corresponding warp count for the current CUDA warp
    size. `outputs_per_thread` is the maximum number of logical C elements
    each thread must visit when the tile is traversed with a simple strided
    loop.

    This does not yet define threadIdx -> (local_m, local_n), warp tiles,
    register tiles, shared-memory loads or a concrete dim3 launch shape.
    """

    threads: int
    warps: int
    outputs_per_thread: int


@dataclass
class CUDAKernelPlan:
    """Minimal target-side representation of one planned CUDA kernel.

    `problem` describes what GEMM must be computed. `tile` describes how the
    problem is partitioned into logical output/reduction tiles. `schedule`
    records how many such tiles are needed to cover the full problem.
    `block_mapping` begins the physical CUDA mapping by assigning a fixed
    number of threads/warps to one logical output tile.

    Concrete thread coordinates, launch dim3 values and memory mapping are
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
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDALoweredModule:
    """CUDA-target lowering result composed of kernel plans."""

    source_ir: IRModule
    kernels: list[CUDAKernelPlan]