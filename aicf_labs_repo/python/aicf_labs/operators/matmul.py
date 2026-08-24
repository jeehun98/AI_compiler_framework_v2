"""Matrix multiplication operator semantics."""

from dataclasses import dataclass

from ..masks import OperatorMask
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
            mask=OperatorMask.PURE,
        )
