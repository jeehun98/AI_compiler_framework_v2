"""Linear layer declaration."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from ..layer import Layer
from ..operators import AddOperator, MatMulOperator


def _positive_feature_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


_Scalar = int | float
_Matrix = tuple[tuple[_Scalar, ...], ...]
_Vector = tuple[_Scalar, ...]


def _finite_number(value: object, name: str) -> _Scalar:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain only numeric values")
    if not math.isfinite(value):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _weight_value(
    value: Sequence[Sequence[_Scalar]],
    in_features: int,
    out_features: int,
) -> _Matrix:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != in_features
    ):
        raise ValueError(f"weight must have shape ({in_features}, {out_features})")
    rows: list[tuple[_Scalar, ...]] = []
    for row in value:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != out_features
        ):
            raise ValueError(f"weight must have shape ({in_features}, {out_features})")
        rows.append(tuple(_finite_number(item, "weight") for item in row))
    return tuple(rows)


def _bias_value(value: Sequence[_Scalar], out_features: int) -> _Vector:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != out_features
    ):
        raise ValueError(f"bias_value must have shape ({out_features},)")
    return tuple(_finite_number(item, "bias_value") for item in value)


@dataclass(frozen=True, init=False)
class Linear(Layer):
    """A MatMul/Add declaration, executable when explicit parameters are present."""

    in_features: int
    out_features: int
    bias: bool
    weight: _Matrix | None = field(repr=False)
    bias_value: _Vector | None = field(repr=False)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        weight: Sequence[Sequence[_Scalar]] | None = None,
        bias_value: Sequence[_Scalar] | None = None,
    ) -> None:
        checked_in_features = _positive_feature_count(in_features, "in_features")
        checked_out_features = _positive_feature_count(
            out_features, "out_features"
        )
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        if not bias and bias_value is not None:
            raise ValueError("bias_value requires bias=True")

        checked_weight = (
            _weight_value(weight, checked_in_features, checked_out_features)
            if weight is not None
            else None
        )
        checked_bias_value = (
            _bias_value(bias_value, checked_out_features)
            if bias_value is not None
            else None
        )

        operators = (MatMulOperator(),)
        if bias:
            operators += (AddOperator(),)

        super().__init__(operators)
        object.__setattr__(self, "in_features", checked_in_features)
        object.__setattr__(self, "out_features", checked_out_features)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "weight", checked_weight)
        object.__setattr__(self, "bias_value", checked_bias_value)

    def forward(self, value: object) -> object:
        """Execute through the same primitive operators exposed as metadata."""

        if self.weight is None:
            raise RuntimeError("Linear execution requires an explicit weight")
        result = self.operators[0](value, self.weight)
        if not self.bias:
            return result
        if self.bias_value is None:
            raise RuntimeError("Linear execution requires an explicit bias_value")
        return self.operators[1](result, self.bias_value)

    def __call__(self, value: object) -> object:
        return self.forward(value)
