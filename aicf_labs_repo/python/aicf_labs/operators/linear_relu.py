"""Logical Linear-plus-ReLU operator produced by frontend rewriting."""

from dataclasses import dataclass

from ..masks import OperatorMask
from ..operator import Operator


@dataclass(frozen=True, init=False)
class LinearReluOperator(Operator):
    """A backend-unbound logical composition of MatMul, bias Add, and ReLU."""

    def __init__(self) -> None:
        super().__init__(
            name="linearRelu",
            expression="y = relu(xW + b)",
            category="composite",
            arity=3,
            mask=OperatorMask.PURE,
        )
