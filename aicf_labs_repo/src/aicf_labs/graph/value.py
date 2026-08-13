from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Value:
    id: int
    shape: tuple[int, ...]
    dtype: str
    name: str
    kind: str = "temporary"

    def __str__(self) -> str:
        return f"%{self.id}:{self.name}{self.shape}"
