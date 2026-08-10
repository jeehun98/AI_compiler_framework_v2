from __future__ import annotations

from ..tensor import Parameter, Tensor, TensorSpec
from ...graph.builder import current_builder


TensorLike = Tensor | Parameter


def _as_tensor(value: TensorLike) -> Tensor:
    if isinstance(value, Tensor):
        if value.value is None:
            raise ValueError("Tensor is not bound to a graph Value")
        return value

    if isinstance(value, Parameter):
        builder = current_builder()

        if builder is None:
            raise RuntimeError("Parameter used outside graph capture")

        return builder.parameter(value)

    raise TypeError(
        f"expected Tensor or Parameter, got {type(value).__name__}"
    )


def _emit(
    op: str,
    inputs: list[TensorLike],
    *,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
) -> Tensor:
    builder = current_builder()

    if builder is None:
        raise RuntimeError("Frontend op called outside graph capture")

    if not inputs:
        raise ValueError("Frontend op must have at least one input")

    resolved = [_as_tensor(value) for value in inputs]

    out_shape = resolved[0].spec.shape if shape is None else shape
    out_dtype = resolved[0].spec.dtype if dtype is None else dtype

    out_spec = TensorSpec(
        shape=out_shape,
        dtype=out_dtype,
    )

    return builder.emit(op, resolved, out_spec)


def gemm(x: TensorLike, w: TensorLike) -> Tensor:
    x_tensor = _as_tensor(x)
    w_tensor = _as_tensor(w)

    if len(x_tensor.spec.shape) != 2:
        raise ValueError(
            f"gemm expects rank-2 x, got {x_tensor.spec.shape}"
        )

    if len(w_tensor.spec.shape) != 2:
        raise ValueError(
            f"gemm expects rank-2 w, got {w_tensor.spec.shape}"
        )

    if x_tensor.spec.shape[1] != w_tensor.spec.shape[0]:
        raise ValueError(
            f"gemm shape mismatch: "
            f"{x_tensor.spec.shape} @ {w_tensor.spec.shape}"
        )

    if x_tensor.spec.dtype != w_tensor.spec.dtype:
        raise TypeError(
            f"gemm dtype mismatch: "
            f"{x_tensor.spec.dtype} vs {w_tensor.spec.dtype}"
        )

    m = x_tensor.spec.shape[0]
    n = w_tensor.spec.shape[1]

    return _emit(
        "gemm",
        [x_tensor, w_tensor],
        shape=(m, n),
        dtype=x_tensor.spec.dtype,
    )


def bias_add(x: TensorLike, bias: TensorLike) -> Tensor:
    x_tensor = _as_tensor(x)
    bias_tensor = _as_tensor(bias)

    if len(x_tensor.spec.shape) < 1:
        raise ValueError("bias_add expects x with rank >= 1")

    if len(bias_tensor.spec.shape) != 1:
        raise ValueError(
            "bias_add currently expects a rank-1 bias"
        )

    if x_tensor.spec.shape[-1] != bias_tensor.spec.shape[0]:
        raise ValueError(
            f"bias shape mismatch: "
            f"{x_tensor.spec.shape} + {bias_tensor.spec.shape}"
        )

    if x_tensor.spec.dtype != bias_tensor.spec.dtype:
        raise TypeError(
            f"bias_add dtype mismatch: "
            f"{x_tensor.spec.dtype} vs {bias_tensor.spec.dtype}"
        )

    return _emit("bias_add", [x_tensor, bias_tensor])


def relu(x: TensorLike) -> Tensor:
    return _emit("relu", [x])