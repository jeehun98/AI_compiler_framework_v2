"""Operator semantic base type."""

from dataclasses import dataclass

from .implementation import Implementation
from .masks import OperatorMask


@dataclass(frozen=True)
class Operator:
    """An immutable semantic computation unit used inside a layer."""

    name: str
    expression: str
    category: str
    arity: int
    mask: OperatorMask
    implementations: tuple[Implementation, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Operator.name must not be empty")
        if not self.expression:
            raise ValueError("Operator.expression must not be empty")
        if not self.category:
            raise ValueError("Operator.category must not be empty")
        if isinstance(self.arity, bool) or not isinstance(self.arity, int):
            raise TypeError("Operator.arity must be an integer")
        if self.arity < 1:
            raise ValueError("Operator.arity must be positive")
        if not isinstance(self.mask, OperatorMask):
            raise TypeError("Operator.mask must be an OperatorMask")
        if not isinstance(self.implementations, tuple):
            raise TypeError("Operator.implementations must be a tuple")
        if not all(
            isinstance(implementation, Implementation)
            for implementation in self.implementations
        ):
            raise TypeError(
                "Operator.implementations must contain only Implementation objects"
            )
