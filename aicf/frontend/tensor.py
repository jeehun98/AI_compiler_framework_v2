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
    spec: TensorSpec
    producer: object | None = None
