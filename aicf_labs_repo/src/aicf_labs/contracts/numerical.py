from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalContract:
    rtol: float = 1e-5
    atol: float = 1e-6

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("numerical tolerances must be non-negative")
