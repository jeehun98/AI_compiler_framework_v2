"""Small positive feature masks for cheap operator candidate screening.

A mask is an index, not a proof of transformation legality. Missing flags mean
only that a feature is not indexed; they do not prove the opposite property.
"""

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Iterable


class OperatorMask(IntFlag):
    """Features currently consumed by model or rewrite candidate matching."""

    NONE = 0
    ELEMENTWISE = auto()
    REDUCTION = auto()
    COMMUTATIVE = auto()
    PURE = auto()
    SHAPE_PRESERVING = auto()
    PERMUTATION = auto()
    BROADCAST = auto()

    def matches(self, required: "OperatorMask") -> bool:
        """Return whether this mask contains every required screening feature."""

        if not isinstance(required, OperatorMask):
            raise TypeError("required must be an OperatorMask")
        return (self & required) == required


@dataclass(frozen=True)
class MaskSummary:
    """Two cheap aggregates for a sequence of operators, not a region IR."""

    common: OperatorMask
    present: OperatorMask

    def matches(
        self,
        required: OperatorMask,
        forbidden: OperatorMask = OperatorMask.NONE,
    ) -> bool:
        """Screen a region by shared requirements and any forbidden feature."""

        if not isinstance(required, OperatorMask):
            raise TypeError("required must be an OperatorMask")
        if not isinstance(forbidden, OperatorMask):
            raise TypeError("forbidden must be an OperatorMask")
        return self.common.matches(required) and not (self.present & forbidden)


def summarize_masks(masks: Iterable[OperatorMask]) -> MaskSummary:
    """Summarize common (AND) and present (OR) features in one pass."""

    common: OperatorMask | None = None
    present = OperatorMask.NONE
    for mask in masks:
        if not isinstance(mask, OperatorMask):
            raise TypeError("masks must contain only OperatorMask values")
        common = mask if common is None else common & mask
        present |= mask
    return MaskSummary(common or OperatorMask.NONE, present)


__all__ = ("MaskSummary", "OperatorMask", "summarize_masks")
