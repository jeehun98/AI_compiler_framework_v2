from .base import OperatorDefinition
from .registry import register


def _shape(inputs, attrs):
    a, b = inputs
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"gemm shape mismatch: {a.shape} x {b.shape}")
    return (a.shape[0], b.shape[1])


GEMM = register(OperatorDefinition("gemm", _shape))


def gemm(a, b):
    return GEMM.emit((a, b))
