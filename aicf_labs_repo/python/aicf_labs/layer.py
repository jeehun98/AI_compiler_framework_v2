"""User-facing layer base type."""

from dataclasses import dataclass, field

from .operator import Operator


@dataclass(frozen=True)
class Layer:
    """A model declaration unit composed of one or more operators."""

    _operators: tuple[Operator, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._operators, tuple):
            raise TypeError("Layer operators must be stored as a tuple")
        if not self._operators:
            raise ValueError("Layer must contain at least one Operator")
        if not all(isinstance(operator, Operator) for operator in self._operators):
            raise TypeError("Layer operators must contain only Operator objects")

    @property
    def operators(self) -> tuple[Operator, ...]:
        """Return the layer's computation operators in declaration order."""

        return self._operators
