from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar

from .graph import Graph
from .node import Node
from .value import Value
from ..frontend.tensor import Tensor, TensorSpec, Parameter

_CURRENT: ContextVar[GraphBuilder | None] = ContextVar("aicf_graph_builder", default=None)


def current_builder():
    return _CURRENT.get()


class GraphBuilder:
    def __init__(self):
        self.graph = Graph()
        self._next_value = 0
        self._next_node = 0
        self._parameter_names: dict[int, str] = {}
        self._parameter_values: dict[int, Tensor] = {}

    def bind_parameters(self, named_parameters) -> None:
        """Register model parameters for this capture session.

        Parameters are materialized as graph Values up front so model state is
        structurally separate from user inputs and temporary values.
        """
        items = list(named_parameters)
        self._parameter_names = {id(parameter): name for name, parameter in items}
        for _, parameter in items:
            self.parameter(parameter)

    def _value(self, spec: TensorSpec, name=None) -> Value:
        value = Value(self._next_value, spec, name)
        self._next_value += 1
        return value

    def input(self, spec: TensorSpec) -> Tensor:
        value = self._value(spec, spec.name)
        self.graph.inputs.append(value)
        return Tensor(spec, value=value)

    def parameter(self, parameter: Parameter) -> Tensor:
        key = id(parameter)
        cached = self._parameter_values.get(key)
        if cached is not None:
            return cached

        name = self._parameter_names.get(key, parameter.spec.name)
        value = self._value(parameter.spec, name=name)
        tensor = Tensor(parameter.spec, value=value)
        self.graph.parameters.append(value)
        self._parameter_values[key] = tensor
        return tensor

    def emit(
        self,
        op: str,
        inputs: list[Tensor],
        out_spec: TensorSpec,
    ) -> Tensor:
        in_values = [tensor.value for tensor in inputs]

        if any(value is None for value in in_values):
            raise RuntimeError(
                "cannot emit graph op from an unbound Tensor"
            )

        out = self._value(out_spec)

        node = Node(
            self._next_node,
            op,
            in_values,
            [out],
        )

        self._next_node += 1

        # Build use-def relationships.
        out.producer = node

        for value in in_values:
            value.users.append(node)

        self.graph.nodes.append(node)

        return Tensor(
            out_spec,
            value=out,
        )

    def output(self, tensor: Tensor):
        if tensor.value is None:
            raise RuntimeError("cannot mark an unbound Tensor as graph output")
        self.graph.outputs.append(tensor.value)


@contextmanager
def capture_graph():
    builder = GraphBuilder()
    token = _CURRENT.set(builder)
    try:
        yield builder
    finally:
        _CURRENT.reset(token)
