from __future__ import annotations

from ..tensor import Tensor


class Parameter(Tensor):
    is_parameter = True

    def __init__(self, shape, dtype: str = "float32", name: str | None = None):
        super().__init__(shape, dtype=dtype, name=name)
