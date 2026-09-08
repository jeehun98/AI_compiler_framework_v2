"""User layers only construct operators and edges; no execution or rules."""

from dataclasses import dataclass
from .operators import Graph, shape_of


@dataclass
class Linear:
    weight: list
    bias: list

    def build(self, graph: Graph, value: str) -> str:
        weight = graph.add("constant", value=self.weight)
        bias = graph.add("constant", value=self.bias)
        product = graph.add("matmul", value, weight)
        if graph.node(bias).shape != (graph.node(product).shape[1],):
            raise ValueError("Linear bias must have shape [out_features]")
        return graph.add("add", product, bias)

    def describe(self):
        return {"type": "Linear", "weight_shape": shape_of(self.weight), "bias_shape": shape_of(self.bias)}


class ReLU:
    def build(self, graph: Graph, value: str) -> str:
        return graph.add("relu", value)

    def describe(self):
        return {"type": "ReLU"}


class Sequential:
    def __init__(self, *layers: Linear | ReLU):
        if not layers or any(not isinstance(layer, (Linear, ReLU)) for layer in layers):
            raise ValueError("Sequential requires Linear/ReLU layers")
        self.layers = layers

    def build(self, input_shape: tuple[int, ...]) -> Graph:
        graph = Graph()
        value = graph.add("input", name="x", shape=input_shape)
        for layer in self.layers:
            value = layer.build(graph, value)
        graph.output = value
        graph.validate()
        return graph

    def describe(self):
        return [layer.describe() for layer in self.layers]
