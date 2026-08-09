from __future__ import annotations
from abc import ABC, abstractmethod
from ..context import CompileContext
from ..ir.module import IRModule


class Pass(ABC):
    name = "pass"

    @abstractmethod
    def run(self, module: IRModule, context: CompileContext) -> IRModule:
        raise NotImplementedError
