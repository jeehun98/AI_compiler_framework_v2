from __future__ import annotations

from ... import ops
from ..module import Module
from ..parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: str = "float32"):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((in_features, out_features), dtype=dtype, name="weight")
        self.bias = Parameter((out_features,), dtype=dtype, name="bias") if bias else None

    def forward(self, x):
        y = ops.gemm(x, self.weight)
        if self.bias is not None:
            y = ops.bias_add(y, self.bias)
        return y
