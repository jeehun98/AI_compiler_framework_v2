from __future__ import annotations

from dataclasses import dataclass

from ..marking import OpMask, OperatorMarkRegistry, mask_names


@dataclass(frozen=True)
class MaskTraceStep:
    operator: str
    operator_mask: OpMask
    surviving_mask: OpMask
    removed_mask: OpMask

    @property
    def surviving_properties(self) -> tuple[str, ...]:
        return mask_names(self.surviving_mask)

    @property
    def removed_properties(self) -> tuple[str, ...]:
        return mask_names(self.removed_mask)


@dataclass(frozen=True)
class MaskPropagationResult:
    operators: tuple[str, ...]
    common_mask: OpMask
    steps: tuple[MaskTraceStep, ...]

    def preserves(self, required: OpMask) -> bool:
        return (self.common_mask & required) == required


def propagate_common_mask(
    operators: tuple[str, ...] | list[str],
    registry: OperatorMarkRegistry,
) -> MaskPropagationResult:
    """Intersect advertised properties across an operator chain.

    This is a cheap candidate-screening primitive. It does not prove semantic
    equivalence or final fusion legality.
    """

    names = tuple(operators)
    if not names:
        raise ValueError("operator chain must not be empty")

    current = registry.get(names[0]).mask
    steps: list[MaskTraceStep] = [
        MaskTraceStep(names[0], current, current, OpMask.NONE)
    ]

    for name in names[1:]:
        operator_mask = registry.get(name).mask
        next_mask = current & operator_mask
        removed = current & ~next_mask
        steps.append(MaskTraceStep(name, operator_mask, next_mask, removed))
        current = next_mask

    return MaskPropagationResult(names, current, tuple(steps))
