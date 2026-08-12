from __future__ import annotations

from ..registry import LoweringRegistry
from .ir import (
    CUDAKernelPlan,
    GEMMBlockMapping,
    GEMMControlFlow,
    GEMMEpilogue,
    GEMMProblem,
    GEMMSchedule,
    GEMMThreadMapping,
    GEMMTile,
)


CUDA_LOWERINGS = LoweringRegistry()


# v0.14 keeps the fixed tile/thread/mapping policies. The new piece is an
# explicit control-flow plan: thread-strided output traversal, optional M/N
# boundary guarding, tiled K traversal and optional K-tail guarding.
_DEFAULT_GEMM_TILE = GEMMTile(
    block_m=128,
    block_n=128,
    block_k=32,
)

_CUDA_WARP_SIZE = 32
_DEFAULT_GEMM_THREADS = 256

_DEFAULT_GEMM_THREAD_MAPPING = GEMMThreadMapping(
    traversal="linear_strided",
    output_order="row_major",
    thread_axis="x",
    block_m_axis="y",
    block_n_axis="x",
)


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
    """Return the explicit fixed GEMM tile policy."""

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
    """Assign a fixed CUDA thread block to one logical GEMM output tile."""

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


def _gemm_thread_mapping(
    tile: GEMMTile,
    block_mapping: GEMMBlockMapping,
) -> GEMMThreadMapping:
    """Return the explicit v0.13 thread/block coordinate convention.

    The current mapping assumes a 1D CUDA block (`threadIdx.x`) and uses a
    row-major linear output index. Logical M tiles map to `blockIdx.y` while N
    tiles map to `blockIdx.x`.
    """

    expected_outputs = tile.block_m * tile.block_n
    mapped_capacity = (
        block_mapping.threads
        * block_mapping.outputs_per_thread
    )

    if mapped_capacity < expected_outputs:
        raise ValueError(
            "GEMM block mapping does not cover the logical output tile"
        )

    return _DEFAULT_GEMM_THREAD_MAPPING




def _gemm_control_flow(
    problem: GEMMProblem,
    tile: GEMMTile,
    thread_mapping: GEMMThreadMapping,
) -> GEMMControlFlow:
    """Derive the naive GEMM kernel control-flow requirements.

    Whole output tiles require no row/column predicate. Likewise, when K is
    an exact multiple of BK, every K iteration is fully in bounds and the
    inner `k < K` predicate can be omitted.
    """

    output_guard = (
        problem.m % tile.block_m != 0
        or problem.n % tile.block_n != 0
    )
    k_tail_guard = problem.k % tile.block_k != 0

    return GEMMControlFlow(
        output_traversal=thread_mapping.traversal,
        output_guard=output_guard,
        k_traversal="tile_then_inner",
        k_tail_guard=k_tail_guard,
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
    thread_mapping: GEMMThreadMapping | None = None,
    control_flow: GEMMControlFlow | None = None,
    epilogue: GEMMEpilogue | None = None,
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
        thread_mapping=thread_mapping,
        control_flow=control_flow,
        epilogue=epilogue,
        attrs=dict(op.attrs),
    )


def _gemm_plan(
    op,
    *,
    index: int,
    strategy: str,
    epilogue: GEMMEpilogue,
) -> CUDAKernelPlan:
    """Build the GEMM problem -> mapping -> epilogue lowering chain."""

    problem = _gemm_problem(op)
    tile = _gemm_tile(problem)
    schedule = _gemm_schedule(problem, tile)
    block_mapping = _gemm_block_mapping(tile)
    thread_mapping = _gemm_thread_mapping(tile, block_mapping)
    control_flow = _gemm_control_flow(
        problem,
        tile,
        thread_mapping,
    )

    if epilogue.bias and len(op.operands) < 3:
        raise ValueError(
            f"{op.name} lowering requires a bias operand for its epilogue"
        )

    if epilogue.activation not in (None, "relu"):
        raise NotImplementedError(
            f"unsupported GEMM epilogue activation: {epilogue.activation}"
        )

    return _kernel_plan(
        op,
        index=index,
        strategy=strategy,
        problem=problem,
        tile=tile,
        schedule=schedule,
        block_mapping=block_mapping,
        thread_mapping=thread_mapping,
        control_flow=control_flow,
        epilogue=epilogue,
    )


@CUDA_LOWERINGS.register("gemm")
def lower_gemm(op, *, index: int, context):
    del context
    return [
        _gemm_plan(
            op,
            index=index,
            strategy="gemm",
            epilogue=GEMMEpilogue(bias=False, activation=None),
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
            epilogue=GEMMEpilogue(bias=True, activation="relu"),
        )
    ]