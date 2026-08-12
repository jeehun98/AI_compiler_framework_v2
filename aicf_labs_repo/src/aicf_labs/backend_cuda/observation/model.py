from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..kernels import KernelSpec


@dataclass(frozen=True)
class LayerObservation:
    layer: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelObservation:
    kernel: KernelSpec
    source: LayerObservation | None = None
    ast: LayerObservation | None = None
    ptx: LayerObservation | None = None
    sass: LayerObservation | None = None
    runtime: LayerObservation | None = None
