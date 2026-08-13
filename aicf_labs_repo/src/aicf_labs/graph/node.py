from __future__ import annotations

from dataclasses import dataclass, field

from .value import Value


@dataclass(frozen=True)
class Node:
    id: int
    op: str
    inputs: tuple[Value, ...]
    output: Value
    attrs: dict[str, object] = field(default_factory=dict)
