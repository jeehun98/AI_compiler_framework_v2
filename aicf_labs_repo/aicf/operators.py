"""The only operator/property catalog, plus an ordered, single-output graph."""

from copy import deepcopy
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Property:
    kind: str
    argument: int | None = None
    condition: str | None = None


@dataclass(frozen=True)
class Semantics:
    arity: int
    expression: str
    properties: tuple[Property, ...] = ()


PURE = Property("PURE")
ELEMENTWISE = Property("ELEMENTWISE")
# Mathematical real-valued contracts. No dtype, kernel or observed facts here.
SEMANTICS = {
    "input": Semantics(0, "external value"),
    "constant": Semantics(0, "literal value"),
    "matmul": Semantics(2, "Y[i,j] = sum_k X[i,k] W[k,j]", (
        PURE, Property("LINEAR", 0, "argument 1 fixed"),
        Property("LINEAR", 1, "argument 0 fixed"))),
    "add": Semantics(2, "Y = A + B (scalar, equal shape or matrix/vector bias)", (
        PURE, ELEMENTWISE, Property("ASSOCIATIVE"))),
    "mul": Semantics(2, "Y = A * B (scalar, equal shape or matrix/vector broadcast)", (
        PURE, ELEMENTWISE, Property("ASSOCIATIVE"))),
    "relu": Semantics(1, "Y[i] = max(0, X[i])", (
        PURE, ELEMENTWISE, Property("POSITIVE_HOMOGENEOUS", 0, "scalar >= 0"))),
    "reduce_sum": Semantics(1, "Y = sum_i X[i] (all axes)", (
        PURE, Property("REDUCTION"), Property("LINEAR", 0))),
    "linear_relu": Semantics(3, "Y = ReLU(MatMul(X, W) + bias)", (PURE,)),
}


def shape_of(value) -> tuple[int, ...]:
    """Finite scalars and nonempty rectangular lists; no Tensor runtime."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Reference values must be finite")
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a finite scalar or nonempty rectangular list")
    shapes = [shape_of(item) for item in value]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("Ragged values are unsupported")
    return (len(value), *shapes[0])


def output_shape(kind, shapes, attributes) -> tuple[int, ...]:
    if kind == "input":
        shape = tuple(attributes["shape"])
        if any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Input shape must contain positive integer dimensions")
        if not isinstance(attributes.get("name"), str) or not attributes["name"]:
            raise ValueError("Input requires a name")
        return shape
    if kind == "constant":
        return shape_of(attributes["value"])
    if kind in {"matmul", "linear_relu"}:
        a, b = shapes[:2]
        if len(a) != 2 or len(b) != 2 or a[1] != b[0]:
            raise ValueError("MatMul requires [M,K] and [K,N]")
        if kind == "linear_relu" and shapes[2] != (b[1],):
            raise ValueError("LinearReLU requires a [N] bias")
        return (a[0], b[1])
    if kind in {"add", "mul"}:
        a, b = shapes
        if a == b or b == ():
            return a
        if a == ():
            return b
        if len(a) == 2 and b == (a[1],):
            return a
        if len(b) == 2 and a == (b[1],):
            return b
        raise ValueError("Only equal shape, scalar or matrix/vector broadcast is supported")
    if kind == "relu":
        return shapes[0]
    if kind == "reduce_sum":
        if attributes.get("axis", "all") != "all":
            raise ValueError("Only all-axis ReduceSum is retained")
        return ()
    raise ValueError(f"Unknown operator: {kind}")


@dataclass(frozen=True)
class Operator:
    id: str
    type: str
    inputs: tuple[str, ...]
    attributes: dict
    shape: tuple[int, ...]

    @property
    def outputs(self):
        return (self.id,)

    @property
    def semantics(self):
        return SEMANTICS[self.type]


@dataclass
class Graph:
    nodes: list[Operator] = field(default_factory=list)
    output: str = ""

    def node(self, node_id: str) -> Operator:
        return next(node for node in self.nodes if node.id == node_id)

    def fresh_id(self, kind: str) -> str:
        used = {node.id for node in self.nodes}
        index = 0
        while f"{kind}.{index}" in used:
            index += 1
        return f"{kind}.{index}"

    def add(self, kind: str, *inputs: str, **attributes) -> str:
        if kind not in SEMANTICS or len(inputs) != SEMANTICS[kind].arity:
            raise ValueError(f"Unknown operator or wrong arity: {kind}")
        shape = output_shape(kind, [self.node(item).shape for item in inputs], attributes)
        node = Operator(self.fresh_id(kind), kind, inputs, deepcopy(attributes), shape)
        self.nodes.append(node)
        self.output = node.id
        return node.id

    def validate(self):
        seen = {}
        input_names = set()
        for node in self.nodes:
            if node.id in seen or node.type not in SEMANTICS:
                raise ValueError("Duplicate node ID or unknown operator")
            if len(node.inputs) != node.semantics.arity or any(item not in seen for item in node.inputs):
                raise ValueError("Graph must be topologically ordered with valid input arity")
            inferred = output_shape(node.type, [seen[item].shape for item in node.inputs], node.attributes)
            if node.shape != inferred:
                raise ValueError("Stored shape differs from operator shape")
            if node.type == "input":
                name = node.attributes["name"]
                if name in input_names:
                    raise ValueError("Input names must be unique")
                input_names.add(name)
            seen[node.id] = node
        if self.output not in seen:
            raise ValueError("Graph output is missing")

    def use_count(self, node_id):
        return sum(node.inputs.count(node_id) for node in self.nodes)
