from __future__ import annotations
from dataclasses import dataclass, field
from .node import Node
from .value import Value


@dataclass
class Graph:
    inputs: list[Value] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
