"""Linear layer declaration."""

from dataclasses import dataclass

from ..layer import Layer
from ..operators import AddOperator, MatMulOperator


def _positive_feature_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, init=False)
class Linear(Layer):
    """A matrix multiplication followed by an optional bias addition."""

    in_features: int
    out_features: int
    bias: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        checked_in_features = _positive_feature_count(in_features, "in_features")
        checked_out_features = _positive_feature_count(
            out_features, "out_features"
        )
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")

        operators = (MatMulOperator(),)
        if bias:
            operators += (AddOperator(),)

        super().__init__(operators)
        object.__setattr__(self, "in_features", checked_in_features)
        object.__setattr__(self, "out_features", checked_out_features)
        object.__setattr__(self, "bias", bias)
