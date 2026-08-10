from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .compiler.analysis.manager import AnalysisManager


@dataclass
class CompileContext:
    target: str = "cuda"
    diagnostics: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    analyses: AnalysisManager = field(default_factory=AnalysisManager)