from __future__ import annotations

from ..registry import LoweringRegistry
from .ir import (
    CUDAKernelPlan,
    GEMMBlockMapping,
    GEMMProblem,
    GEMMSchedule,
    GEMMTile,
)


CUDA_LOWERINGS = LoweringRegistry()


# v0.12 keeps the fixed logical tile policy and introduces only a minimal
# physical thread-block mapping. Tile selection and detailed thread mapping
# are still intentionally deferred.
_DEFAULT_GEMM_TILE = GEMMTile(
    block_m=128,
    block_n=128,
    block_k=32,
)

_CUDA_WARP_SIZE = 32
_DEFAULT_GEMM_THREADS = 256


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor


def _gemm_problem(op) -> GEMMProblem:
    """Extract a 2D GEMM problem from GEMM-like IR operands.

    Expected operand layout:
        operand 0: X [M, K]
        operand 1: W [K, N]

    Fused epilogue operands may follow, but they do not change M/N/K.
    """

    if len(op.operands) < 2:
        raise ValueError(
            f"{op.name} lowering expects at least two operands"
        )

    x = op.operands[0]
    w = op.operands[1]

    x_shape = x.type.shape
    w_shape = w.type.shape

    if len(x_shape) != 2 or len(w_shape) != 2:
        raise ValueError(
            f"{op.name} lowering expects rank-2 GEMM operands, "
            f"got {x_shape} and {w_shape}"
        )

    m, k = x_shape
    w_k, n = w_shape

    if k != w_k:
        raise ValueError(
            f"{op.name} lowering GEMM shape mismatch: "
            f"{x_shape} @ {w_shape}"
        )

    if x.type.dtype != w.type.dtype:
        raise TypeError(
            f"{op.name} lowering GEMM dtype mismatch: "
            f"{x.type.dtype} vs {w.type.dtype}"
        )

    if op.results:
        expected_shape = (m, n)
        actual_shape = op.results[0].type.shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"{op.name} lowering result shape mismatch: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    return GEMMProblem(
        m=m,
        n=n,
        k=k,
        dtype=x.type.dtype,
    )


def _gemm_tile(problem: GEMMProblem) -> GEMMTile:
    """Return the explicit fixed GEMM tile policy.

    `problem` remains part of the interface so this function can later become
    a real target planner without changing lowering-rule call sites.
    """

    del problem
    return _DEFAULT_GEMM_TILE


def _gemm_schedule(
    problem: GEMMProblem,
    tile: GEMMTile,
) -> GEMMSchedule:
    """Compute the logical tile counts needed to cover a GEMM problem."""

    if tile.block_m <= 0 or tile.block_n <= 0 or tile.block_k <= 0:
        raise ValueError("GEMM tile dimensions must be positive")

    return GEMMSchedule(
        grid_m=_ceil_div(problem.m, tile.block_m),
        grid_n=_ceil_div(problem.n, tile.block_n),
        k_tiles=_ceil_div(problem.k, tile.block_k),
    )



def _gemm_block_mapping(tile: GEMMTile) -> GEMMBlockMapping:
    """Map one logical output tile onto a minimal CUDA thread block.

    v0.12 uses a fixed 256-thread policy. Threads conceptually traverse the
    BM*BN output elements with a 1D strided loop:

        output = threadIdx.x
        output += blockDim.x

    Exact local_m/local_n coordinate mapping is deferred to v0.13.
    """

    threads = _DEFAULT_GEMM_THREADS

    if threads <= 0:
        raise ValueError("GEMM block thread count must be positive")

    if threads % _CUDA_WARP_SIZE != 0:
        raise ValueError("GEMM block thread count must be warp-aligned")

    tile_outputs = tile.block_m * tile.block_n

    if tile_outputs <= 0:
        raise ValueError("GEMM output tile must contain at least one element")

    return GEMMBlockMapping(
        threads=threads,
        warps=threads // _CUDA_WARP_SIZE,
        outputs_per_thread=_ceil_div(tile_outputs, threads),
    )


def _kernel_plan(
    op,
    *,
    index: int,
    strategy: str,
    problem: GEMMProblem | None = None,
    tile: GEMMTile | None = None,
    schedule: GEMMSchedule | None = None,
    block_mapping: GEMMBlockMapping | None = None,
) -> CUDAKernelPlan:
    return CUDAKernelPlan(
        name=f"kernel_{index}_{op.name}",
        source_op=op.name,
        strategy=strategy,
        inputs=tuple(value.name for value in op.operands),
        outputs=tuple(value.name for value in op.results),
        problem=problem,
        tile=tile,
        schedule=schedule,
        block_mapping=block_mapping,
        attrs=dict(op.attrs),
    )


def _gemm_plan(op, *, index: int, strategy: str) -> CUDAKernelPlan:
    """Build the common problem -> tile -> schedule chain for GEMM-like ops."""

    problem = _gemm_problem(op)
    tile = _gemm_tile(problem)
    schedule = _gemm_schedule(problem, tile)
    block_mapping = _gemm_block_mapping(tile)

    return _kernel_plan(
        op,
        index=index,
        strategy=strategy,
        problem=problem,
        tile=tile,
        schedule=schedule,
        block_mapping=block_mapping,
    )


@CUDA_LOWERINGS.register("gemm")
def lower_gemm(op, *, index: int, context):
    del context
    return [
        _gemm_plan(
            op,
            index=index,
            strategy="gemm",
        )
    ]


@CUDA_LOWERINGS.register("bias_add")
def lower_bias_add(op, *, index: int, context):
    del context
    return [
        _kernel_plan(
            op,
            index=index,
            strategy="elementwise_bias_add",
        )
    ]


@CUDA_LOWERINGS.register("relu")
def lower_relu(op, *, index: int, context):
    del context
    return [
        _kernel_plan(
            op,
            index=index,
            strategy="elementwise_relu",
        )
    ]


@CUDA_LOWERINGS.register("fused_gemm_bias_relu")
def lower_fused_gemm_bias_relu(op, *, index: int, context):
    del context
    return [
        _gemm_plan(
            op,
            index=index,
            strategy="gemm_epilogue_bias_relu",
        )
    ]