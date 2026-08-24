"""Minimal model-execution tracing, graph rewriting, and reference execution."""

from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
import re

from .masks import OperatorMask
from .model import Model
from .operator import Operator
from .operators import LinearReluOperator


ReferenceValue = int | float | list["ReferenceValue"] | tuple["ReferenceValue", ...]


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _shape_of(value: ReferenceValue) -> tuple[int, ...]:
    if not _is_sequence(value):
        return ()
    items = tuple(value)
    if not items:
        return (0,)
    child_shape = _shape_of(items[0])
    if any(_shape_of(item) != child_shape for item in items[1:]):
        raise ValueError("reference values must have a rectangular shape")
    return (len(items), *child_shape)


def _dtype_of(value: ReferenceValue) -> str:
    if _is_sequence(value):
        items = tuple(value)
        return _dtype_of(items[0]) if items else "unknown"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    raise TypeError(f"unsupported reference value type: {type(value).__name__}")


def _freeze_reference_value(value: object) -> ReferenceValue:
    """Validate and detach a literal value before storing it in a graph."""

    if isinstance(value, bool):
        raise TypeError("graph constants must contain only numeric values")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("graph constants must be finite")
        return value
    if _is_sequence(value):
        frozen = tuple(_freeze_reference_value(item) for item in value)
        _shape_of(frozen)
        return frozen
    raise TypeError("graph constants must be numeric scalars or rectangular tensors")


def _elementwise_binary(
    left: ReferenceValue,
    right: ReferenceValue,
    operation: Callable[[int | float, int | float], int | float],
) -> ReferenceValue:
    if _is_sequence(left) and _is_sequence(right):
        left_items = tuple(left)
        right_items = tuple(right)
        left_shape = _shape_of(left)
        right_shape = _shape_of(right)
        if (
            len(left_shape) == 2
            and len(right_shape) == 1
            and left_shape[1] == right_shape[0]
        ):
            return [
                _elementwise_binary(left_item, right_items, operation)
                for left_item in left_items
            ]
        if (
            len(right_shape) == 2
            and len(left_shape) == 1
            and right_shape[1] == left_shape[0]
        ):
            return [
                _elementwise_binary(left_items, right_item, operation)
                for right_item in right_items
            ]
        if len(left_items) != len(right_items):
            raise ValueError("elementwise operands must have equal shapes")
        return [
            _elementwise_binary(left_item, right_item, operation)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        ]
    if _is_sequence(left):
        return [_elementwise_binary(item, right, operation) for item in left]
    if _is_sequence(right):
        return [_elementwise_binary(left, item, operation) for item in right]
    return operation(left, right)


def _elementwise_unary(
    value: ReferenceValue,
    operation: Callable[[int | float], int | float],
) -> ReferenceValue:
    if _is_sequence(value):
        return [_elementwise_unary(item, operation) for item in value]
    return operation(value)


def _matmul(left: ReferenceValue, right: ReferenceValue) -> ReferenceValue:
    if not (_is_sequence(left) and _is_sequence(right)):
        raise TypeError("matmul expects two rank-2 reference values")
    left_items = tuple(left)
    right_items = tuple(right)
    if (
        not left_items
        or not right_items
        or any(not _is_sequence(row) for row in (*left_items, *right_items))
    ):
        raise TypeError("matmul expects two non-empty rank-2 reference values")
    left_rows = tuple(tuple(row) for row in left_items)
    right_rows = tuple(tuple(row) for row in right_items)
    inner_size = len(left_rows[0])
    if any(len(row) != inner_size for row in left_rows):
        raise ValueError("left matmul operand must be rectangular")
    if len(right_rows) != inner_size:
        raise ValueError("matmul inner dimensions must match")
    column_count = len(right_rows[0])
    if any(len(row) != column_count for row in right_rows):
        raise ValueError("right matmul operand must be rectangular")
    return [
        [
            sum(
                left_rows[row][inner] * right_rows[inner][column]
                for inner in range(inner_size)
            )
            for column in range(column_count)
        ]
        for row in range(len(left_rows))
    ]


def _sum_all(value: ReferenceValue) -> int | float:
    if _is_sequence(value):
        return sum(_sum_all(item) for item in value)
    return value


def _evaluate_operator(
    name: str,
    inputs: tuple[ReferenceValue, ...],
) -> ReferenceValue:
    if name == "add":
        return _elementwise_binary(inputs[0], inputs[1], lambda left, right: left + right)
    if name == "mul":
        return _elementwise_binary(inputs[0], inputs[1], lambda left, right: left * right)
    if name == "matmul":
        return _matmul(inputs[0], inputs[1])
    if name == "relu":
        return _elementwise_unary(inputs[0], lambda value: max(value, 0))
    if name == "linearRelu":
        product = _matmul(inputs[0], inputs[1])
        biased = _elementwise_binary(product, inputs[2], lambda left, right: left + right)
        return _elementwise_unary(biased, lambda value: max(value, 0))
    if name == "transpose":
        if not _is_sequence(inputs[0]):
            raise TypeError("transpose expects a rank-2 reference value")
        rows = tuple(tuple(row) for row in inputs[0] if _is_sequence(row))
        if len(rows) != len(inputs[0]):
            raise TypeError("transpose expects a rank-2 reference value")
        return [list(column) for column in zip(*rows, strict=True)]
    if name == "reduceSum":
        return _sum_all(inputs[0])
    raise NotImplementedError(f"reference execution does not support '{name}'")


@dataclass(frozen=True)
class GraphNode:
    """One single-output operation; its node ID is also its output value ID."""

    id: str
    operator_id: str
    input_ids: tuple[str, ...]
    mask: OperatorMask
    shape: tuple[int, ...]
    dtype: str
    constant_value: ReferenceValue | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class ExecutionGraph:
    """Canonical frontend graph for the current single-output operator subset."""

    id: str
    name: str
    nodes: tuple[GraphNode, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("ExecutionGraph node IDs must be unique")
        seen: set[str] = set()
        for node in self.nodes:
            if any(input_id not in seen for input_id in node.input_ids):
                raise ValueError("ExecutionGraph nodes must be in execution order")
            seen.add(node.id)
        if not self.outputs or any(output not in seen for output in self.outputs):
            raise ValueError("ExecutionGraph outputs must reference graph nodes")

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes if node.operator_id == "input")

    @property
    def operator_sequence(self) -> tuple[str, ...]:
        return tuple(
            node.operator_id
            for node in self.nodes
            if node.operator_id not in {"input", "constant"}
        )

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def to_web_graph(self) -> dict[str, object]:
        """Serialize to the existing TypeScript ``Graph`` schema."""

        web_nodes: list[dict[str, object]] = []
        web_edges: list[dict[str, object]] = []
        edge_index = 0
        for node in self.nodes:
            if node.operator_id == "input":
                parameters: dict[str, object] = {
                    "symbol": node.symbol or node.id,
                    "shape": list(node.shape),
                }
            elif node.operator_id == "constant":
                if _is_sequence(node.constant_value):
                    raise NotImplementedError(
                        "the Web Graph schema currently supports scalar constants only"
                    )
                parameters = {"value": node.constant_value}
            else:
                parameters = {}
            web_nodes.append(
                {"id": node.id, "operatorId": node.operator_id, "parameters": parameters}
            )
            for input_index, input_id in enumerate(node.input_ids):
                web_edges.append(
                    {
                        "id": f"edge.{edge_index}",
                        "sourceNodeId": input_id,
                        "sourcePort": "out",
                        "targetNodeId": node.id,
                        "targetPort": f"in-{input_index}",
                    }
                )
                edge_index += 1
        return {
            "id": self.id,
            "name": self.name,
            "nodes": web_nodes,
            "edges": web_edges,
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True)
class _TraceValue:
    data: ReferenceValue
    node_id: str
    tracer: "_OperationTracer"


class _OperationTracer:
    def __init__(self, graph_id: str, name: str) -> None:
        self.graph_id = graph_id
        self.name = name
        self.nodes: list[GraphNode] = []
        self.operator_counts: dict[str, int] = {}
        self.constant_count = 0

    def add_input(
        self,
        value: ReferenceValue,
        index: int,
        symbol: str,
    ) -> _TraceValue:
        node_id = f"input.{index}"
        self.nodes.append(
            GraphNode(
                id=node_id,
                operator_id="input",
                input_ids=(),
                mask=OperatorMask.NONE,
                shape=_shape_of(value),
                dtype=_dtype_of(value),
                symbol=symbol,
            )
        )
        return _TraceValue(value, node_id, self)

    def as_trace_value(self, value: object) -> _TraceValue:
        if isinstance(value, _TraceValue):
            if value.tracer is not self:
                raise ValueError("traced values cannot cross tracing contexts")
            return value
        constant = _freeze_reference_value(value)
        node_id = f"constant.{self.constant_count}"
        self.constant_count += 1
        self.nodes.append(
            GraphNode(
                id=node_id,
                operator_id="constant",
                input_ids=(),
                mask=OperatorMask.NONE,
                shape=_shape_of(constant),
                dtype=_dtype_of(constant),
                constant_value=constant,
            )
        )
        return _TraceValue(constant, node_id, self)

    def invoke(self, operator: Operator, inputs: tuple[object, ...]) -> _TraceValue:
        traced_inputs = tuple(self.as_trace_value(value) for value in inputs)
        result = _evaluate_operator(
            operator.name,
            tuple(value.data for value in traced_inputs),
        )
        operator_index = self.operator_counts.get(operator.name, 0)
        self.operator_counts[operator.name] = operator_index + 1
        node_id = f"{operator.name}.{operator_index}"
        self.nodes.append(
            GraphNode(
                id=node_id,
                operator_id=operator.name,
                input_ids=tuple(value.node_id for value in traced_inputs),
                mask=operator.mask,
                shape=_shape_of(result),
                dtype=_dtype_of(result),
            )
        )
        return _TraceValue(result, node_id, self)

    def finish(self, outputs: object) -> ExecutionGraph:
        output_values = outputs if isinstance(outputs, tuple) else (outputs,)
        if not output_values or not all(
            isinstance(value, _TraceValue) and value.tracer is self
            for value in output_values
        ):
            raise TypeError("model outputs must be values produced inside the trace")
        return ExecutionGraph(
            id=self.graph_id,
            name=self.name,
            nodes=tuple(self.nodes),
            outputs=tuple(value.node_id for value in output_values),
        )


_ACTIVE_TRACER: ContextVar[_OperationTracer | None] = ContextVar(
    "aicf_operation_tracer",
    default=None,
)


def invoke_operator(operator: Operator, inputs: tuple[object, ...]) -> object:
    """Execute an operator and record it when a model trace is active."""

    if len(inputs) != operator.arity:
        raise TypeError(f"{operator.name} expects {operator.arity} inputs, got {len(inputs)}")
    tracer = _ACTIVE_TRACER.get()
    if tracer is not None:
        return tracer.invoke(operator, inputs)
    return _evaluate_operator(operator.name, inputs)  # type: ignore[arg-type]


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def trace_model(
    model: Model,
    *inputs: ReferenceValue,
    input_names: Sequence[str] | None = None,
) -> ExecutionGraph:
    """Run ``model`` once and directly build its operation execution graph."""

    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    names = tuple(
        (f"input{index}" for index in range(len(inputs)))
        if input_names is None
        else input_names
    )
    if len(names) != len(inputs) or any(not name for name in names):
        raise ValueError("input_names must provide one non-empty name per input")
    model_name = type(model).__name__
    tracer = _OperationTracer(f"graph.{_snake_case(model_name)}", model_name)
    traced_inputs = tuple(
        tracer.add_input(value, index, names[index])
        for index, value in enumerate(inputs)
    )
    token = _ACTIVE_TRACER.set(tracer)
    try:
        outputs = model(*traced_inputs)
    finally:
        _ACTIVE_TRACER.reset(token)
    return tracer.finish(outputs)


def execute_graph(
    graph: ExecutionGraph,
    inputs: Sequence[ReferenceValue],
) -> ReferenceValue | tuple[ReferenceValue, ...]:
    """Execute the supported graph subset in recorded topological order."""

    if len(inputs) != len(graph.inputs):
        raise ValueError(f"graph expects {len(graph.inputs)} inputs, got {len(inputs)}")
    input_values = dict(zip(graph.inputs, inputs, strict=True))
    values: dict[str, ReferenceValue] = {}
    for node in graph.nodes:
        if node.operator_id == "input":
            values[node.id] = input_values[node.id]
        elif node.operator_id == "constant":
            if node.constant_value is None:
                raise ValueError(f"constant node '{node.id}' has no value")
            values[node.id] = node.constant_value
        else:
            values[node.id] = _evaluate_operator(
                node.operator_id,
                tuple(values[input_id] for input_id in node.input_ids),
            )
    outputs = tuple(values[output_id] for output_id in graph.outputs)
    return outputs[0] if len(outputs) == 1 else outputs


def reference_values_close(
    left: object,
    right: object,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """Compare nested finite reference values with explicit tolerances."""

    if _is_sequence(left) and _is_sequence(right):
        left_items = tuple(left)
        right_items = tuple(right)
        return len(left_items) == len(right_items) and all(
            reference_values_close(
                left_item,
                right_item,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
    return left == right


@dataclass(frozen=True)
class FrontendRewriteMetrics:
    rule_id: str
    nodes_scanned: int
    mask_accepted: int
    condition_accepted: int
    rewrites_applied: int

    @property
    def candidate_reduction(self) -> float:
        if self.nodes_scanned == 0:
            return 0.0
        return 1.0 - self.mask_accepted / self.nodes_scanned


@dataclass(frozen=True)
class GraphSimplification:
    graph: ExecutionGraph
    metrics: tuple[FrontendRewriteMetrics, ...]
    node_count_before: int
    node_count_after: int
    removed_nodes: tuple[str, ...]
    rewrite_count: int


@dataclass(frozen=True)
class PatternCandidate:
    """A structurally discovered opportunity, with no legality or apply behavior."""

    pattern_id: str
    node_ids: tuple[str, ...]
    root_id: str


@dataclass(frozen=True)
class PatternSearchResult:
    """Candidate list and the minimum useful screening counts."""

    candidates: tuple[PatternCandidate, ...]
    nodes_scanned: int
    mask_accepted: int

    @property
    def structural_matches(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class LegalityResult:
    """Minimal decision for one structural candidate; it cannot rewrite a graph."""

    candidate: PatternCandidate
    legal: bool
    reasons: tuple[str, ...]


def find_linear_relu_candidates(graph: ExecutionGraph) -> PatternSearchResult:
    """Find MatMul -> Add -> ReLU structure without proving or rewriting it."""

    required = (
        OperatorMask.ELEMENTWISE
        | OperatorMask.PURE
        | OperatorMask.SHAPE_PRESERVING
    )
    screened = tuple(node for node in graph.nodes if node.mask.matches(required))
    candidates: list[PatternCandidate] = []
    for relu in screened:
        if relu.operator_id != "relu" or len(relu.input_ids) != 1:
            continue
        add = graph.node(relu.input_ids[0])
        if add.operator_id != "add":
            continue
        for input_id in add.input_ids:
            matmul = graph.node(input_id)
            if matmul.operator_id == "matmul":
                candidates.append(
                    PatternCandidate(
                        pattern_id="linear-relu-fusion",
                        node_ids=(matmul.id, add.id, relu.id),
                        root_id=relu.id,
                    )
                )
    return PatternSearchResult(
        candidates=tuple(candidates),
        nodes_scanned=len(graph.nodes),
        mask_accepted=len(screened),
    )


def _consumer_ids(graph: ExecutionGraph, node_id: str) -> set[str]:
    return {
        node.id
        for node in graph.nodes
        if node_id in node.input_ids
    }


def check_linear_relu_legality(
    graph: ExecutionGraph,
    candidate: PatternCandidate,
) -> LegalityResult:
    """Check only bias-add shape and single-consumer requirements."""

    if candidate.pattern_id != "linear-relu-fusion" or len(candidate.node_ids) != 3:
        raise ValueError("candidate must be a linear-relu-fusion structural match")
    matmul = graph.node(candidate.node_ids[0])
    add = graph.node(candidate.node_ids[1])
    relu = graph.node(candidate.node_ids[2])
    if (
        matmul.operator_id != "matmul"
        or add.operator_id != "add"
        or relu.operator_id != "relu"
        or candidate.root_id != relu.id
        or relu.input_ids != (add.id,)
        or matmul.id not in add.input_ids
    ):
        raise ValueError("candidate nodes do not form MatMul -> Add -> ReLU")

    reasons: list[str] = []
    other_input_ids = tuple(
        input_id for input_id in add.input_ids if input_id != matmul.id
    )
    if len(add.input_ids) != 2 or len(other_input_ids) != 1:
        reasons.append("not_bias_add")
    else:
        bias = graph.node(other_input_ids[0])
        if bias.operator_id != "constant":
            reasons.append("not_bias_add")
        elif (
            len(matmul.shape) != 2
            or len(bias.shape) != 1
            or matmul.shape[1] != bias.shape[0]
        ):
            reasons.append("unsupported_bias_shape")

    if _consumer_ids(graph, matmul.id) != {add.id}:
        reasons.append("matmul_has_extra_consumer")
    if _consumer_ids(graph, add.id) != {relu.id}:
        reasons.append("add_has_extra_consumer")
    return LegalityResult(
        candidate=candidate,
        legal=not reasons,
        reasons=tuple(reasons),
    )


def _available_node_id(graph: ExecutionGraph, prefix: str) -> str:
    existing = {node.id for node in graph.nodes}
    index = 0
    while f"{prefix}.{index}" in existing:
        index += 1
    return f"{prefix}.{index}"


def rewrite_linear_relu(
    graph: ExecutionGraph,
    legality: LegalityResult,
) -> ExecutionGraph:
    """Replace one legal MatMul/Add/ReLU chain with a logical fused node."""

    checked = check_linear_relu_legality(graph, legality.candidate)
    if checked != legality:
        raise ValueError("legality result does not match the supplied graph")
    if not legality.legal:
        reasons = ", ".join(legality.reasons)
        raise ValueError(f"cannot rewrite an illegal candidate: {reasons}")

    matmul = graph.node(legality.candidate.node_ids[0])
    add = graph.node(legality.candidate.node_ids[1])
    relu = graph.node(legality.candidate.node_ids[2])
    if len(matmul.input_ids) != 2:
        raise ValueError("linearRelu rewrite requires a binary MatMul")
    bias_ids = tuple(input_id for input_id in add.input_ids if input_id != matmul.id)
    if len(bias_ids) != 1:
        raise ValueError("linearRelu rewrite requires one bias input")

    fused_operator = LinearReluOperator()
    fused_id = _available_node_id(graph, fused_operator.name)
    fused = GraphNode(
        id=fused_id,
        operator_id=fused_operator.name,
        input_ids=(*matmul.input_ids, bias_ids[0]),
        mask=fused_operator.mask,
        shape=relu.shape,
        dtype=relu.dtype,
    )
    removed_ids = {matmul.id, add.id, relu.id}
    rewritten_nodes: list[GraphNode] = []
    for node in graph.nodes:
        if node.id == relu.id:
            rewritten_nodes.append(fused)
        elif node.id not in removed_ids:
            rewritten_nodes.append(
                replace(
                    node,
                    input_ids=tuple(
                        fused.id if input_id == relu.id else input_id
                        for input_id in node.input_ids
                    ),
                )
            )
    return ExecutionGraph(
        id=graph.id,
        name=graph.name,
        nodes=tuple(rewritten_nodes),
        outputs=tuple(
            fused.id if output_id == relu.id else output_id
            for output_id in graph.outputs
        ),
    )


def _identity_matches(
    graph: ExecutionGraph,
    candidates: tuple[GraphNode, ...],
    operator_id: str,
    identity: int | float,
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for node in candidates:
        if node.operator_id != operator_id or len(node.input_ids) != 2:
            continue
        left_id, right_id = node.input_ids
        left = graph.node(left_id)
        right = graph.node(right_id)
        if left.operator_id == "constant" and left.constant_value == identity:
            replacements[node.id] = right.id
        elif right.operator_id == "constant" and right.constant_value == identity:
            replacements[node.id] = left.id
    return replacements


def _bypass_nodes(
    graph: ExecutionGraph,
    replacements: dict[str, str],
) -> ExecutionGraph:
    def resolve(node_id: str) -> str:
        seen: set[str] = set()
        while node_id in replacements:
            if node_id in seen:
                raise ValueError("rewrite replacements contain a cycle")
            seen.add(node_id)
            node_id = replacements[node_id]
        return node_id

    return ExecutionGraph(
        id=graph.id,
        name=graph.name,
        nodes=tuple(
            replace(node, input_ids=tuple(resolve(value) for value in node.input_ids))
            for node in graph.nodes
            if node.id not in replacements
        ),
        outputs=tuple(resolve(output) for output in graph.outputs),
    )


def simplify_graph(graph: ExecutionGraph) -> GraphSimplification:
    """Apply only the identity rewrites required by the frontend vertical slice."""

    required = OperatorMask.ELEMENTWISE | OperatorMask.PURE
    rules = (
        ("add-zero", "add", 0),
        ("mul-one", "mul", 1),
    )
    current = graph
    metrics: list[FrontendRewriteMetrics] = []
    removed_nodes: list[str] = []
    for rule_id, operator_id, identity in rules:
        nodes_scanned = len(current.nodes)
        candidates = tuple(node for node in current.nodes if node.mask.matches(required))
        replacements = _identity_matches(current, candidates, operator_id, identity)
        if replacements:
            current = _bypass_nodes(current, replacements)
            removed_nodes.extend(replacements)
        metrics.append(
            FrontendRewriteMetrics(
                rule_id=rule_id,
                nodes_scanned=nodes_scanned,
                mask_accepted=len(candidates),
                condition_accepted=len(replacements),
                rewrites_applied=len(replacements),
            )
        )
    return GraphSimplification(
        graph=current,
        metrics=tuple(metrics),
        node_count_before=len(graph.nodes),
        node_count_after=len(current.nodes),
        removed_nodes=tuple(removed_nodes),
        rewrite_count=len(removed_nodes),
    )


__all__ = (
    "ExecutionGraph",
    "FrontendRewriteMetrics",
    "GraphNode",
    "GraphSimplification",
    "LegalityResult",
    "PatternCandidate",
    "PatternSearchResult",
    "ReferenceValue",
    "execute_graph",
    "check_linear_relu_legality",
    "find_linear_relu_candidates",
    "invoke_operator",
    "reference_values_close",
    "rewrite_linear_relu",
    "simplify_graph",
    "trace_model",
)
