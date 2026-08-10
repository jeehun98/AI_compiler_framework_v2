from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .value import Value


@dataclass(eq=False)
class Node:
    id: int
    op: str
    inputs: list[Value]
    outputs: list[Value]
    attrs: dict[str, Any] = field(default_factory=dict)