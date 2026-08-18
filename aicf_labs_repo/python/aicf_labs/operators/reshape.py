"""Reshape operator semantics."""

from dataclasses import dataclass

from ..masks import Monotonicity, OperatorMask, State
from ..operator import Operator


@dataclass(frozen=True, init=False)
class ReshapeOperator(Operator):
    """Declarative reshape with materialization and rank details unresolved."""

    def __init__(self) -> None:
        super().__init__(
            name="reshape",
            expression="output = reshape(input)",
            category="layout_transform",
            arity=1,
            mask=OperatorMask(
                elementwise=State.NO,
                reduction=State.NO,
                shape_preserving=State.NO,
                rank_preserving=State.UNKNOWN,
                element_independent=State.YES,
                linear=State.YES,
                idempotent=State.UNKNOWN,
                zero_preserving=State.YES,
                invertible=State.UNKNOWN,
                monotonicity=Monotonicity.NOT_APPLICABLE,
                producer_fusible=State.UNKNOWN,
                consumer_fusible=State.UNKNOWN,
                epilogue_fusible=State.UNKNOWN,
                requires_materialization=State.UNKNOWN,
                requires_global_sync=State.NO,
            ),
        )
