"""Activation layer declarations."""

from dataclasses import dataclass

from ..layer import Layer
from ..operators import ReluOperator


@dataclass(frozen=True, init=False)
class ReLU(Layer):
    """A ReLU layer containing one ReLU operator."""

    def __init__(self) -> None:
        super().__init__((ReluOperator(),))
