from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImplementationSpec:
    name: str
    backend: str
    source: str
    configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("implementation name must not be empty")
        if not self.backend:
            raise ValueError("implementation backend must not be empty")
