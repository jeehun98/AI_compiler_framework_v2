"""Reshape operator semantics."""

from dataclasses import dataclass

from ..masks import OperatorMask
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
            mask=OperatorMask.PURE,
        )
