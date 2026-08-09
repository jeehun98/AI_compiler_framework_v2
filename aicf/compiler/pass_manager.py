from __future__ import annotations
from .pass_base import Pass
from ..context import CompileContext
from ..ir.module import IRModule
from ..diagnostics.events import emit


class PassManager:
    def __init__(self, passes: list[Pass] | None = None):
        self.passes = passes or []

    def add(self, p: Pass):
        self.passes.append(p)

    def run(self, module: IRModule, context: CompileContext) -> IRModule:
        for p in self.passes:
            emit(context, "pass.before", {"pass": p.name, "module": module})
            module = p.run(module, context)
            emit(context, "pass.after", {"pass": p.name, "module": module})
        return module
