from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from .graph import Graph
from .node import Node
from .value import Value
from ..frontend.tensor import Tensor, TensorSpec

_CURRENT: ContextVar[GraphBuilder | None] = ContextVar("aicf_graph_builder", default=None)


def current_builder():
    return _CURRENT.get()


class GraphBuilder:
    def __init__(self):
        self.graph = Graph()
        self._next_value = 0
        self._next_node = 0

    def _value(self, spec: TensorSpec, name=None) -> Value:
        v = Value(self._next_value, spec, name)
        self._next_value += 1
        return v

    def input(self, spec: TensorSpec) -> Tensor:
        value = self._value(spec, spec.name)
        self.graph.inputs.append(value)
        return Tensor(spec, producer=value)

    def emit(self, op: str, inputs: list[Tensor], out_spec: TensorSpec) -> Tensor:
        in_values = [t.producer for t in inputs]
        out = self._value(out_spec)
        node = Node(self._next_node, op, in_values, [out])
        self._next_node += 1
        self.graph.nodes.append(node)
        return Tensor(out_spec, producer=out)

    def output(self, tensor: Tensor):
        self.graph.outputs.append(tensor.producer)


@contextmanager
def capture_graph():
    builder = GraphBuilder()
    token = _CURRENT.set(builder)
    try:
        yield builder
    finally:
        _CURRENT.reset(token)
