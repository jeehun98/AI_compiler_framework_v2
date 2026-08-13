from .base import OperatorDefinition
from .registry import register


def _shape(inputs, attrs):
    x, bias = inputs
    if len(bias.shape) != 1 or not x.shape or x.shape[-1] != bias.shape[0]:
        raise ValueError(f"bias_add shape mismatch: {x.shape} + {bias.shape}")
    return x.shape


BIAS_ADD = register(OperatorDefinition("bias_add", _shape))


def bias_add(x, bias):
    return BIAS_ADD.emit((x, bias))
