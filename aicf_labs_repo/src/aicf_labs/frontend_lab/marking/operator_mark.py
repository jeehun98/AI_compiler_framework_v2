from __future__ import annotations

from dataclasses import dataclass

from .mask import OpMask, mask_names


@dataclass(frozen=True)
class OperatorMark:
    """Binary property marking for one mathematical operator."""

    operator: str
    mask: OpMask
    note: str = ""

    def has(self, required: OpMask) -> bool:
        return (self.mask & required) == required

    @property
    def properties(self) -> tuple[str, ...]:
        return mask_names(self.mask)
