from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..tensor import Tensor
from ..graph import current_builder


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    infer_shape: Callable[[tuple[Tensor, ...], dict[str, object]], tuple[int, ...]]

    def emit(self, inputs: tuple[Tensor, ...], **attrs) -> Tensor:
        shape = self.infer_shape(inputs, attrs)
        dtype = inputs[0].dtype
        builder = current_builder()
        if builder is None:
            return Tensor(shape, dtype=dtype, name=self.name)
        return builder.emit(self.name, inputs, shape, dtype, attrs)
