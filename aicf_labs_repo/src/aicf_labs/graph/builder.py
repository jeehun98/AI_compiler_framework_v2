from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager

from ..tensor import Tensor
from .graph import Graph
from .node import Node
from .value import Value

_CURRENT: ContextVar["GraphBuilder | None"] = ContextVar("aicf_graph_builder", default=None)


class GraphBuilder:
    def __init__(self):
        self.graph = Graph()
        self._next_value = 0
        self._next_node = 0
        self._bindings: dict[int, Value] = {}

    def _new_value(self, tensor: Tensor, name: str, kind: str) -> Value:
        value = Value(self._next_value, tensor.shape, tensor.dtype, name, kind)
        self._next_value += 1
        return value

    def value_for(self, tensor: Tensor) -> Value:
        if tensor.value is not None:
            return tensor.value
        key = id(tensor)
        if key in self._bindings:
            return self._bindings[key]
        kind = "parameter" if getattr(tensor, "is_parameter", False) else "input"
        name = tensor.name or kind
        value = self._new_value(tensor, name, kind)
        self._bindings[key] = value
        (self.graph.parameters if kind == "parameter" else self.graph.inputs).append(value)
        return value

    def emit(self, op: str, inputs: tuple[Tensor, ...], shape, dtype: str, attrs=None) -> Tensor:
        input_values = tuple(self.value_for(t) for t in inputs)
        out_proto = Tensor(shape, dtype=dtype, name=op)
        output = self._new_value(out_proto, op, "temporary")
        node = Node(self._next_node, op, input_values, output, dict(attrs or {}))
        self._next_node += 1
        self.graph.nodes.append(node)
        return out_proto._bind(output)

    def finish(self, output: Tensor) -> Graph:
        self.graph.outputs = [self.value_for(output)]
        return self.graph

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)


def current_builder() -> GraphBuilder | None:
    return _CURRENT.get()


def capture(model, x: Tensor) -> Graph:
    builder = GraphBuilder()
    with builder.activate():
        y = model(x)
    return builder.finish(y)
