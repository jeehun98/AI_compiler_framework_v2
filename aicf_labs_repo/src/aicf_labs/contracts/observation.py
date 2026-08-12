from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaticObservation:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeObservation:
    latency_us: float | None = None
    gflops: float | None = None
    values: dict[str, Any] = field(default_factory=dict)
