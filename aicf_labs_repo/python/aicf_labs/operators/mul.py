"""Elementwise multiplication operator semantics."""

from dataclasses import dataclass

from ..masks import OperatorMask
from ..operator import Operator


@dataclass(frozen=True, init=False)
class MulOperator(Operator):
    """Declarative elementwise multiplication with scalar broadcasting."""

    def __init__(self) -> None:
        super().__init__(
            name="mul",
            expression="y = x * b",
            category="elementwise",
            arity=2,
            mask=(
                OperatorMask.ELEMENTWISE
                | OperatorMask.COMMUTATIVE
                | OperatorMask.PURE
                | OperatorMask.BROADCAST
            ),
        )
