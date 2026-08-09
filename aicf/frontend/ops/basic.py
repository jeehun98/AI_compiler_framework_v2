from __future__ import annotations
from ..tensor import Tensor, TensorSpec
from ...graph.builder import current_builder


def _emit(op: str, inputs: list[Tensor], *, shape=None, dtype=None) -> Tensor:
    builder = current_builder()
    if builder is None:
        raise RuntimeError("Frontend op called outside graph capture")
    out_spec = TensorSpec(
        shape=shape or inputs[0].spec.shape,
        dtype=dtype or inputs[0].spec.dtype,
    )
    return builder.emit(op, inputs, out_spec)


def gemm(x: Tensor, w: Tensor) -> Tensor:
    # TODO: real shape inference and validation.
    m = x.spec.shape[0]
    n = w.spec.shape[-1]
    return _emit("gemm", [x, w], shape=(m, n))


def bias_add(x: Tensor, bias: Tensor) -> Tensor:
    return _emit("bias_add", [x, bias])


def relu(x: Tensor) -> Tensor:
    return _emit("relu", [x])
