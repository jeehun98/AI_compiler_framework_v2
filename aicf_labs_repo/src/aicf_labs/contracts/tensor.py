from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    layout: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.shape or any(int(dim) <= 0 for dim in self.shape):
            raise ValueError(f"tensor shape must contain positive dimensions: {self.shape}")
