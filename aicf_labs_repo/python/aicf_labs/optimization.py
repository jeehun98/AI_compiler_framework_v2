"""Small bridges from verified frontend rewrites to optimization records."""

from .frontend import (
    ExecutionGraph,
    LegalityResult,
    check_linear_relu_legality,
    rewrite_linear_relu,
)
from .trace_registry import TraceRecord
from .tracing import (
    Attribute,
    DecisionKind,
    ExecutionPlan,
    OptimizationDecision,
    PlannedExecutionUnit,
    ValueSpec,
)
from .verification import GraphExecutionComparison


def _value_spec(graph: ExecutionGraph, node_id: str) -> ValueSpec:
    node = graph.node(node_id)
    return ValueSpec(id=node.id, shape=node.shape, dtype=node.dtype)


def record_linear_relu_semantic_fusion(
    original: ExecutionGraph,
    rewritten: ExecutionGraph,
    legality: LegalityResult,
    verification: GraphExecutionComparison,
) -> TraceRecord:
    """Record one verified logical rewrite without backend or runtime claims."""

    checked_legality = check_linear_relu_legality(original, legality.candidate)
    if checked_legality != legality or not legality.legal:
        raise ValueError("semantic fusion recording requires a legal candidate")
    expected_rewrite = rewrite_linear_relu(original, legality)
    if rewritten != expected_rewrite:
        raise ValueError("rewritten graph does not match the legal candidate")
    if not verification.equivalent:
        raise ValueError("semantic fusion recording requires verified equivalence")

    fused_nodes = tuple(
        node
        for node in rewritten.nodes
        if node.operator_id == "linearRelu"
        and node.id not in {original_node.id for original_node in original.nodes}
    )
    if len(fused_nodes) != 1:
        raise ValueError("rewritten graph must contain one new linearRelu node")
    fused = fused_nodes[0]
    decision_id = f"decision.{fused.id}"
    plan_id = f"plan.{fused.id}"
    unit_id = f"{plan_id}.unit0"

    decision = OptimizationDecision(
        id=decision_id,
        kind=DecisionKind.SEMANTIC_FUSION,
        inputs=legality.candidate.node_ids,
        outputs=(unit_id,),
        preconditions=(
            "Add uses a supported rank-1 constant bias.",
            "MatMul output has only the Add consumer.",
            "Add output has only the ReLU consumer.",
        ),
        semantic_preservation=(
            "Reference execution confirmed equivalent graph outputs under the "
            "recorded tolerances.",
        ),
        expected_effects=(
            "Replace MatMul, Add, and ReLU with one logical linearRelu node.",
        ),
        references=(
            Attribute("pattern_id", legality.candidate.pattern_id),
            Attribute("rewritten_node_id", fused.id),
            Attribute("verification", "reference_execution"),
            Attribute("rel_tol", str(verification.rel_tol)),
            Attribute("abs_tol", str(verification.abs_tol)),
        ),
    )
    unit = PlannedExecutionUnit(
        id=unit_id,
        logical_operator_ids=legality.candidate.node_ids,
        inputs=tuple(_value_spec(rewritten, node_id) for node_id in fused.input_ids),
        outputs=(_value_spec(rewritten, fused.id),),
        decision_ids=(decision.id,),
        expected_kernel_launches=None,
        implementation_binding_id=None,
    )
    plan = ExecutionPlan(id=plan_id, units=(unit,))
    return TraceRecord(
        logical_operator_ids=legality.candidate.node_ids,
        decisions=(decision,),
        plans=(plan,),
    )


__all__ = ("record_linear_relu_semantic_fusion",)
