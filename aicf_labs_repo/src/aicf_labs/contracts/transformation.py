from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformationRecord:
    candidate: str
    found: bool
    legal: bool
    selected: bool
    reason: str
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
