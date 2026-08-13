from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph.value import Value


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    dtype: str = "float32"
    name: str | None = None
    _value: "Value | None" = None

    def __init__(self, shape, dtype: str = "float32", name: str | None = None, _value=None):
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_value", _value)

    @property
    def value(self):
        return self._value

    def _bind(self, value: "Value") -> "Tensor":
        return replace(self, _value=value)

    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        from .ops import add

        return add(self, other)

    def __radd__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        from .ops import add

        return add(other, self)
