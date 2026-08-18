"""Elementwise addition operator semantics."""

from dataclasses import dataclass

from ..masks import Monotonicity, OperatorMask, State
from ..operator import Operator


@dataclass(frozen=True, init=False)
class AddOperator(Operator):
    """Declarative addition, including broadcast forms such as bias add."""

    def __init__(self) -> None:
        super().__init__(
            name="add",
            expression="y = x + b",
            category="elementwise",
            arity=2,
            mask=OperatorMask(
                elementwise=State.YES,
                reduction=State.NO,
                shape_preserving=State.UNKNOWN,
                rank_preserving=State.UNKNOWN,
                element_independent=State.YES,
                linear=State.NO,
                idempotent=State.NO,
                zero_preserving=State.YES,
                invertible=State.UNKNOWN,
                monotonicity=Monotonicity.NONDECREASING,
                producer_fusible=State.YES,
                consumer_fusible=State.YES,
                epilogue_fusible=State.YES,
                requires_materialization=State.NO,
                requires_global_sync=State.NO,
            ),
        )
