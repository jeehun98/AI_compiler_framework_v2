from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .value import IRValue


@dataclass
class Operation:
    name: str
    operands: list[IRValue]
    results: list[IRValue]
    attrs: dict[str, Any] = field(default_factory=dict)
