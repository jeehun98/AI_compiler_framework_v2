"""Activation layer declarations."""

from dataclasses import dataclass

from ..layer import Layer
from ..operators import ReluOperator


@dataclass(frozen=True, init=False)
class ReLU(Layer):
    """A ReLU layer containing one ReLU operator."""

    def __init__(self) -> None:
        super().__init__((ReluOperator(),))

    def forward(self, value: object) -> object:
        """Execute through the layer's existing primitive operator."""

        return self.operators[0](value)

    def __call__(self, value: object) -> object:
        return self.forward(value)
