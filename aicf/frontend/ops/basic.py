from __future__ import annotations
from ..tensor import Tensor, TensorSpec, Parameter
from ...graph.builder import current_builder

TensorLike = Tensor | Parameter


def _as_tensor(value: TensorLike) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Parameter):
        builder = current_builder()
        if builder is None:
            raise RuntimeError("Parameter used outside graph capture")
        return builder.parameter(value)
    raise TypeError(f"expected Tensor or Parameter, got {type(value).__name__}")


def _emit(op: str, inputs: list[TensorLike], *, shape=None, dtype=None) -> Tensor:
    builder = current_builder()
    if builder is None:
        raise RuntimeError("Frontend op called outside graph capture")

    resolved = [_as_tensor(value) for value in inputs]
    out_spec = TensorSpec(
        shape=shape or resolved[0].spec.shape,
        dtype=dtype or resolved[0].spec.dtype,
    )
    return builder.emit(op, resolved, out_spec)


def gemm(x: TensorLike, w: TensorLike) -> Tensor:
    x_tensor = _as_tensor(x)
    w_tensor = _as_tensor(w)

    # Minimal validation; full shape inference is intentionally left for later.
    if len(x_tensor.spec.shape) != 2 or len(w_tensor.spec.shape) != 2:
        raise ValueError("gemm currently expects rank-2 tensors")
    if x_tensor.spec.shape[1] != w_tensor.spec.shape[0]:
        raise ValueError(
            f"gemm shape mismatch: {x_tensor.spec.shape} @ {w_tensor.spec.shape}"
        )

    m = x_tensor.spec.shape[0]
    n = w_tensor.spec.shape[1]
    return _emit("gemm", [x_tensor, w_tensor], shape=(m, n))


def bias_add(x: TensorLike, bias: TensorLike) -> Tensor:
    x_tensor = _as_tensor(x)
    bias_tensor = _as_tensor(bias)
    if len(bias_tensor.spec.shape) != 1:
        raise ValueError("bias_add currently expects a rank-1 bias")
    if x_tensor.spec.shape[-1] != bias_tensor.spec.shape[0]:
        raise ValueError(
            f"bias shape mismatch: {x_tensor.spec.shape} + {bias_tensor.spec.shape}"
        )
    return _emit("bias_add", [x_tensor, bias_tensor])


def relu(x: TensorLike) -> Tensor:
    return _emit("relu", [x])
