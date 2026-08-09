from __future__ import annotations
from dataclasses import dataclass
from ..frontend.tensor import TensorSpec


@dataclass
class Value:
    id: int
    spec: TensorSpec
    name: str | None = None
