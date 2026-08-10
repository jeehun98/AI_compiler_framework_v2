from __future__ import annotations

from ...ir.module import IRModule
from .use_def import UseDefAnalysis, build_use_def


class AnalysisManager:
    """Own and cache analyses derived from mutable IR.

    Analyses are keyed by IRModule identity. Because passes currently mutate an
    IRModule in place, any transforming pass must invalidate cached analyses
    after changing the module.
    """

    def __init__(self) -> None:
        self._use_def: dict[int, UseDefAnalysis] = {}

    def use_def(self, module: IRModule) -> UseDefAnalysis:
        key = id(module)
        analysis = self._use_def.get(key)
        if analysis is None:
            analysis = build_use_def(module)
            self._use_def[key] = analysis
        return analysis

    def invalidate(self, module: IRModule) -> None:
        self._use_def.pop(id(module), None)

    def invalidate_all(self) -> None:
        self._use_def.clear()