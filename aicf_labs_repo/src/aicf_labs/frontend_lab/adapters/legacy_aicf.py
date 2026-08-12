from __future__ import annotations

from ...contracts import NumericalContract, TensorSpec, TransformationRecord, WorkloadSpec


def fused_gemm_bias_relu_workload(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str = "float32",
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> WorkloadSpec:
    """Contract adapter for the first v0.20 fused vertical slice.

    This intentionally accepts semantic shape information rather than a CUDA
    lowering object. A later adapter can extract the same fields directly from
    the legacy optimized IR without leaking that IR into Backend CUDA Lab.
    """

    return WorkloadSpec(
        kind="gemm_bias_relu",
        inputs=(
            TensorSpec((m, k), dtype=dtype, layout="row_major", name="x"),
            TensorSpec((k, n), dtype=dtype, layout="row_major", name="weight"),
            TensorSpec((n,), dtype=dtype, name="bias"),
        ),
        outputs=(
            TensorSpec((m, n), dtype=dtype, layout="row_major", name="output"),
        ),
        attributes={"activation": "relu"},
        numerical=NumericalContract(rtol=rtol, atol=atol),
    )


def gemm_bias_relu_fusion_record(*, selected: bool = True) -> TransformationRecord:
    return TransformationRecord(
        candidate="gemm_bias_relu",
        found=True,
        legal=True,
        selected=selected,
        reason=(
            "selected by baseline fusion policy; IR rewrite applied"
            if selected
            else "legal candidate retained without rewrite"
        ),
        before=("gemm", "bias_add", "relu"),
        after=("fused_gemm_bias_relu",) if selected else ("gemm", "bias_add", "relu"),
    )
