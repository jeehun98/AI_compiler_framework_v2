from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompileContext:
    target: str = "cuda"
    diagnostics: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)