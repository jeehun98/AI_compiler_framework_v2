from __future__ import annotations

from .pass_base import Pass
from ..context import CompileContext
from ..ir.module import IRModule
from ..ir.verifier import verify_ir
from ..diagnostics.events import emit


class PassManager:
    def __init__(self, passes: list[Pass] | None = None):
        self.passes = passes or []

    def add(self, p: Pass):
        self.passes.append(p)

    def run(self, module: IRModule, context: CompileContext) -> IRModule:
        # Establish the invariant before the first pass sees the module.
        verify_ir(module)

        for p in self.passes:
            emit(context, "pass.before", {"pass": p.name, "module": module})
            module = p.run(module, context)

            # Every pass must leave valid IR behind, even if it did not mutate.
            verify_ir(module)

            emit(context, "pass.after", {"pass": p.name, "module": module})

        return module