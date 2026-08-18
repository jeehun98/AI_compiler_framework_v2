"""Shape-oriented layer declarations."""

from dataclasses import dataclass

from ..layer import Layer
from ..operators import ReshapeOperator


@dataclass(frozen=True, init=False)
class Flatten(Layer):
    """Flatten dimensions starting at ``start_dim`` using a reshape operator."""

    start_dim: int

    def __init__(self, start_dim: int = 1) -> None:
        if isinstance(start_dim, bool) or not isinstance(start_dim, int):
            raise TypeError("start_dim must be an integer")
        super().__init__((ReshapeOperator(),))
        object.__setattr__(self, "start_dim", start_dim)
