from __future__ import annotations
from .module import Module
from ..frontend.tensor import Parameter, TensorSpec
from ..frontend.ops import gemm, bias_add, relu


class Linear(Module):
    """Minimal Linear layer: y = x @ W + b.

    Weight layout is [in_features, out_features] in this skeleton so it maps
    directly onto the current GEMM frontend op.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        dtype: str = "float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            TensorSpec((in_features, out_features), dtype=dtype, name="weight")
        )
        self.bias = (
            Parameter(TensorSpec((out_features,), dtype=dtype, name="bias"))
            if bias
            else None
        )

    def forward(self, x):
        y = gemm(x, self.weight)
        if self.bias is not None:
            y = bias_add(y, self.bias)
        return y

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)
