from __future__ import annotations
from dataclasses import dataclass, field
from .operation import Operation
from .value import IRValue


@dataclass
class IRModule:
    inputs: list[IRValue] = field(default_factory=list)
    parameters: list[IRValue] = field(default_factory=list)
    ops: list[Operation] = field(default_factory=list)
    outputs: list[IRValue] = field(default_factory=list)
