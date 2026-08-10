from __future__ import annotations

from ..registry import LoweringRegistry
from .ir import CUDAKernelPlan, GEMMProblem


CUDA_LOWERINGS = LoweringRegistry()


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


def _kernel_plan(
    op,
    *,
    index: int,
    strategy: str,
    problem: GEMMProblem | None = None,
) -> CUDAKernelPlan:
    return CUDAKernelPlan(
        name=f"kernel_{index}_{op.name}",
        source_op=op.name,
        strategy=strategy,
        inputs=tuple(value.name for value in op.operands),
        outputs=tuple(value.name for value in op.results),
        problem=problem,
        attrs=dict(op.attrs),
    )


@CUDA_LOWERINGS.register("gemm")
def lower_gemm(op, *, index: int, context):
    return [
        _kernel_plan(
            op,
            index=index,
            strategy="gemm",
            problem=_gemm_problem(op),
        )
    ]


@CUDA_LOWERINGS.register("bias_add")
def lower_bias_add(op, *, index: int, context):
    return [
        _kernel_plan(
            op,
            index=index,
            strategy="elementwise_bias_add",
        )
    ]


@CUDA_LOWERINGS.register("relu")
def lower_relu(op, *, index: int, context):
    return [
        _kernel_plan(
            op,
            index=index,
            strategy="elementwise_relu",
        )
    ]


@CUDA_LOWERINGS.register("fused_gemm_bias_relu")
def lower_fused_gemm_bias_relu(op, *, index: int, context):
    return [
        _kernel_plan(
            op,
            index=index,
            strategy="gemm_epilogue_bias_relu",
            problem=_gemm_problem(op),
        )
    ]