"""Matrix multiplication operator semantics."""

from dataclasses import dataclass

from ..masks import Monotonicity, OperatorMask, State
from ..operator import Operator


@dataclass(frozen=True, init=False)
class MatMulOperator(Operator):
    """Declarative matrix multiplication with no selected implementation."""

    def __init__(self) -> None:
        super().__init__(
            name="matmul",
            expression="y = xW",
            category="matrix_operation",
            arity=2,
            mask=OperatorMask(
                elementwise=State.NO,
                reduction=State.NO,
                shape_preserving=State.NO,
                rank_preserving=State.UNKNOWN,
                element_independent=State.NO,
                linear=State.NO,
                idempotent=State.NO,
                zero_preserving=State.YES,
                invertible=State.UNKNOWN,
                monotonicity=Monotonicity.CONDITIONAL,
                producer_fusible=State.UNKNOWN,
                consumer_fusible=State.UNKNOWN,
                epilogue_fusible=State.UNKNOWN,
                requires_materialization=State.UNKNOWN,
                requires_global_sync=State.UNKNOWN,
            ),
        )
