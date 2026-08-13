from .base import OperatorDefinition
from .registry import register


def _shape(inputs, attrs):
    return inputs[0].shape


RELU = register(OperatorDefinition("relu", _shape))


def relu(x):
    return RELU.emit((x,))
