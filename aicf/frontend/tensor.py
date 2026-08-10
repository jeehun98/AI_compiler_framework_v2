from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    name: Optional[str] = None


@dataclass
class Tensor:
    """Symbolic frontend tensor used while graph capture is active.

    `value` points to the graph-level Value that represents this tensor.
    Actual storage/device data is intentionally not implemented yet.
    """

    spec: TensorSpec
    value: object | None = None


@dataclass(eq=False)
class Parameter:
    """Model-owned tensor state.

    This is intentionally lightweight. A future implementation can make
    Parameter a real Tensor subclass with storage, initialization and gradients.
    """

    spec: TensorSpec
    requires_grad: bool = True

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> str:
        return self.spec.dtype
