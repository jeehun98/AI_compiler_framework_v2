from __future__ import annotations
from dataclasses import dataclass
from .type import TensorType


@dataclass
class IRValue:
    name: str
    type: TensorType
