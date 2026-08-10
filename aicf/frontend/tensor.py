from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("shape must be a tuple")

        for dimension in self.shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool):
                raise TypeError("shape dimensions must be integers")
            if dimension < 0:
                raise ValueError("shape dimensions must be non-negative")

        if not isinstance(self.dtype, str) or not self.dtype:
            raise TypeError("dtype must be a non-empty string")


@dataclass
class Tensor:
    """Symbolic tensor associated with a graph-level Value."""

    spec: TensorSpec
    value: object | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype


@dataclass(eq=False)
class Parameter:
    """Model-owned symbolic tensor state."""

    spec: TensorSpec
    requires_grad: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a bool")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype