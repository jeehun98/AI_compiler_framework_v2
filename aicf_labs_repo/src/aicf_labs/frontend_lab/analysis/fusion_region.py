from __future__ import annotations

from dataclasses import dataclass

from ..marking import OpMask, OperatorMarkRegistry
from .propagation import MaskPropagationResult, propagate_common_mask


@dataclass(frozen=True)
class FusionScreen:
    name: str
    required_mask: OpMask

    def evaluate(
        self,
        operators: tuple[str, ...] | list[str],
        registry: OperatorMarkRegistry,
    ) -> MaskPropagationResult:
        return propagate_common_mask(operators, registry)

    def candidate_survives(self, result: MaskPropagationResult) -> bool:
        return result.preserves(self.required_mask)


GENERIC_ELEMENTWISE_SCREEN = FusionScreen(
    name="generic_elementwise",
    required_mask=(
        OpMask.SIDE_EFFECT_FREE
        | OpMask.ELEMENT_LOCAL
        | OpMask.MATERIALIZATION_OPTIONAL
        | OpMask.FUSION_FRIENDLY
    ),
)
