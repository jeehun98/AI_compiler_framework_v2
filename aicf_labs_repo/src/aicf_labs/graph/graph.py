from __future__ import annotations

from dataclasses import dataclass, field

from .node import Node
from .value import Value


@dataclass
class Graph:
    inputs: list[Value] = field(default_factory=list)
    parameters: list[Value] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)

    def operator_names(self) -> tuple[str, ...]:
        return tuple(node.op for node in self.nodes)

    def dump(self) -> str:
        lines: list[str] = []
        for value in self.inputs:
            lines.append(f"%{value.id} = input {value.shape} {value.dtype}")
        for value in self.parameters:
            lines.append(f"%{value.id} = parameter {value.name} {value.shape} {value.dtype}")
        for node in self.nodes:
            args = ", ".join(f"%{v.id}" for v in node.inputs)
            lines.append(f"%{node.output.id} = {node.op}({args})")
        return "\n".join(lines)
