"""Two internal rules, reference execution, and explicit CUDA selection."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from .cuda import lookup
from .operators import Graph, Operator, shape_of


@dataclass(frozen=True)
class Decision:
    rule: str
    status: str  # ACCEPT / REJECT / UNKNOWN
    nodes: tuple[str, ...]
    reason: str
    domain: str = "real"


def fuse_linear_relu(source: Graph, domain="real") -> tuple[Graph, list[Decision]]:
    """Match MatMul → bias Add → ReLU, check private intermediates, replace."""
    source.validate()
    graph, decisions = deepcopy(source), []
    for original in list(source.nodes):
        activation = graph.node(original.id)
        if activation.type != "relu":
            continue
        add = graph.node(activation.inputs[0])
        if add.type != "add":
            continue
        operands = [graph.node(item) for item in add.inputs]
        product = next((node for node in operands if node.type == "matmul"), None)
        if product is None:
            continue
        bias = operands[1] if operands[0].id == product.id else operands[0]
        nodes = (product.id, add.id, activation.id)
        status, reason = "ACCEPT", "Private MatMul and [N] bias Add compose with ReLU."
        if domain != "real":
            status, reason = "UNKNOWN", "No execution-domain equivalence proof is available."
        elif bias.type != "constant":
            status, reason = "UNKNOWN", "This rule requires a declared constant bias."
        elif bias.shape != (product.shape[1],):
            status, reason = "REJECT", "Bias must have shape [N]."
        elif any(graph.use_count(node.id) != 1 or graph.output == node.id for node in (product, add)):
            status, reason = "REJECT", "MatMul/Add intermediates must have one use and not be graph outputs."
        elif not all(any(prop.kind == "PURE" for prop in node.semantics.properties)
                     for node in (product, add, activation)):
            status, reason = "UNKNOWN", "Required PURE semantics are not declared."
        decisions.append(Decision("linear_relu", status, nodes, reason, domain))
        if status != "ACCEPT":
            continue
        fused = Operator(graph.fresh_id("linear_relu"), "linear_relu",
                         (*product.inputs, bias.id), {}, activation.shape)
        rewritten = []
        for node in graph.nodes:
            if node.id == activation.id:
                rewritten.append(fused)
            elif node.id not in (product.id, add.id):
                rewritten.append(replace(node, inputs=tuple(
                    fused.id if item == activation.id else item for item in node.inputs)))
        graph = Graph(rewritten, fused.id if graph.output == activation.id else graph.output)
        graph.validate()
    return graph, decisions


def propagate_positive_scale(source: Graph, domain="real") -> tuple[Graph, list[Decision]]:
    """Move a scalar through a unary operator with nonnegative homogeneity."""
    source.validate()
    graph, decisions = deepcopy(source), []
    for original in list(source.nodes):
        target = graph.node(original.id)
        if len(target.inputs) != 1:
            continue
        scale = graph.node(target.inputs[0])
        if scale.type != "mul":
            continue
        operands = [graph.node(item) for item in scale.inputs]
        alpha = next((node for node in operands if node.type == "constant" and node.shape == ()), None)
        if alpha is None:
            continue
        value = operands[1] if operands[0].id == alpha.id else operands[0]
        claim = next((prop for prop in target.semantics.properties
                      if prop.kind == "POSITIVE_HOMOGENEOUS" and prop.argument == 0), None)
        status, reason = "ACCEPT", "Nonnegative homogeneity permits scale movement over reals."
        if claim is None or claim.condition != "scalar >= 0":
            status, reason = "UNKNOWN", "Supported scoped homogeneity is not declared."
        elif alpha.attributes["value"] < 0:
            status, reason = "REJECT", "The declared scalar >= 0 condition is false."
        elif domain != "real":
            status, reason = "UNKNOWN", "Real homogeneity does not establish IEEE or tolerance equivalence."
        elif graph.use_count(scale.id) != 1 or graph.output == scale.id:
            status, reason = "REJECT", "Scale must have one use and not be a graph output."
        decisions.append(Decision("positive_scale", status, (scale.id, target.id), reason, domain))
        if status != "ACCEPT":
            continue
        rewritten = []
        for node in graph.nodes:
            if node.id == scale.id:
                continue
            if node.id == target.id:
                rewritten.extend([replace(target, inputs=(value.id,)),
                                  replace(scale, inputs=(target.id, alpha.id), shape=target.shape)])
            else:
                rewritten.append(replace(node, inputs=tuple(
                    scale.id if item == target.id else item for item in node.inputs)))
        graph = Graph(rewritten, scale.id if graph.output == target.id else graph.output)
        graph.validate()
    return graph, decisions


def transform(graph: Graph, rule="linear_relu", domain="real"):
    if rule == "linear_relu":
        return fuse_linear_relu(graph, domain)
    if rule == "positive_scale":
        return propagate_positive_scale(graph, domain)
    raise ValueError(f"Unknown backend rule: {rule}")


def _map(value, function):
    return [_map(item, function) for item in value] if isinstance(value, list) else function(value)


def _binary(a, b, function):
    if not isinstance(a, list):
        return _map(b, lambda value: function(a, value))
    if not isinstance(b, list):
        return _map(a, lambda value: function(value, b))
    if len(shape_of(a)) == 2 and len(shape_of(b)) == 1:
        return [_binary(row, b, function) for row in a]
    if len(shape_of(b)) == 2 and len(shape_of(a)) == 1:
        return [_binary(a, row, function) for row in b]
    return [_binary(x, y, function) for x, y in zip(a, b)]


def _matmul(a, b):
    return [[sum(x * b[k][j] for k, x in enumerate(row))
             for j in range(len(b[0]))] for row in a]


def _sum(value):
    return sum(_sum(item) for item in value) if isinstance(value, list) else value


def _evaluate(graph: Graph, inputs: dict) -> dict:
    graph.validate()
    values = {}
    for node in graph.nodes:
        args = [values[item] for item in node.inputs]
        if node.type == "input":
            value = inputs[node.attributes["name"]]
        elif node.type == "constant":
            value = node.attributes["value"]
        elif node.type == "matmul":
            value = _matmul(*args)
        elif node.type == "add":
            value = _binary(*args, lambda a, b: a + b)
        elif node.type == "mul":
            value = _binary(*args, lambda a, b: a * b)
        elif node.type == "relu":
            value = _map(args[0], lambda x: max(0, x))
        elif node.type == "reduce_sum":
            value = _sum(args[0])
        elif node.type == "linear_relu":
            value = _map(_binary(_matmul(args[0], args[1]), args[2], lambda a, b: a + b), lambda x: max(0, x))
        else:
            raise ValueError(f"No reference execution for {node.type}")
        if shape_of(value) != node.shape:
            raise ValueError(f"Value shape does not match {node.id}")
        values[node.id] = deepcopy(value)
    return values


def execute(graph: Graph, inputs: dict):
    return _evaluate(graph, inputs)[graph.output]


def select_implementations(graph: Graph):
    graph.validate()
    return [(node, lookup(node.type)) for node in graph.nodes if node.type not in {"input", "constant"}]


def _graph_report(graph: Graph):
    return {"output": graph.output, "nodes": [
        {**asdict(node), "outputs": node.outputs, "semantics": asdict(node.semantics)}
        for node in graph.nodes]}


def run(model, value) -> dict:
    original = model.build(shape_of(value))
    transformed, decisions = transform(original)
    before = execute(original, {"x": value})
    values = _evaluate(transformed, {"x": value})
    after = values[transformed.output]
    selections = []
    for node, implementation in select_implementations(transformed):
        selections.append({"node_id": node.id, "implementation_id": implementation.id,
                           "launch": implementation.launch(tuple(values[item] for item in node.inputs), node.attributes)})
    return {"schema_version": 1, "model": model.describe(),
            "original": _graph_report(original), "transformed": _graph_report(transformed),
            "decisions": [asdict(decision) for decision in decisions],
            "execution": {"backend": "reference", "original_output": before, "transformed_output": after,
                          "equal": before == after, "scope": "Numeric equality on these reference inputs only; not a GPU or bitwise proof."},
            "cuda": selections}
