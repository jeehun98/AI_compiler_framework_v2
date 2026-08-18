"""Ordered layer-based model declaration."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import overload

from .layer import Layer
from .model import Model
from .operator import Operator


@dataclass(frozen=True, init=False, repr=False, slots=True)
class Sequential(Model):
    """Store layers in declaration order and expose structural inspection."""

    _layers: tuple[Layer, ...] = field(repr=False)

    def __init__(self, *layers: Layer) -> None:
        for index, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(
                    "Sequential accepts only Layer instances; "
                    f"item {index} is {type(layer).__name__}"
                )
        object.__setattr__(self, "_layers", tuple(layers))

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers)

    @overload
    def __getitem__(self, index: int) -> Layer: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Layer, ...]: ...

    def __getitem__(self, index: int | slice) -> Layer | tuple[Layer, ...]:
        return self._layers[index]

    def operators(self) -> tuple[Operator, ...]:
        """Flatten layer-owned operators for read-only inspection."""

        return tuple(
            operator
            for layer in self._layers
            for operator in layer.operators
        )

    def summary(self) -> str:
        if not self._layers:
            return "Sequential()"

        operator_name_width = max(
            len(operator.name) for operator in self.operators()
        )
        lines = ["Sequential("]
        for layer_index, layer in enumerate(self._layers):
            lines.append(f"  [{layer_index}] {layer!r}")
            for operator_index, operator in enumerate(layer.operators):
                branch = "└─" if operator_index == len(layer.operators) - 1 else "├─"
                implementation_names = ", ".join(
                    implementation.name
                    for implementation in operator.implementations
                )
                implementation_suffix = (
                    f" [{implementation_names}]" if implementation_names else ""
                )
                lines.append(
                    "      "
                    f"{branch} {operator.name:<{operator_name_width}} : "
                    f"{operator.expression}{implementation_suffix}"
                )
        lines.append(")")
        return "\n".join(lines)
