from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .numerical import NumericalContract
from .tensor import TensorSpec


@dataclass(frozen=True)
class WorkloadSpec:
    """Implementation-independent computation contract."""

    kind: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    numerical: NumericalContract = field(default_factory=NumericalContract)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("workload kind must not be empty")
        if not self.inputs:
            raise ValueError("workload requires at least one input")
        if not self.outputs:
            raise ValueError("workload requires at least one output")

    @property
    def key(self) -> str:
        shapes = ",".join("x".join(map(str, spec.shape)) for spec in self.inputs)
        return f"{self.kind}:{shapes}"
