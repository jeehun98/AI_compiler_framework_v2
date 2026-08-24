"""Minimal model metadata, traced reference frontend, and execution evidence API."""

from .implementation import Implementation, Observation, SassEvidence
from .frontend import (
    ExecutionGraph,
    FrontendRewriteMetrics,
    GraphNode,
    GraphSimplification,
    LegalityResult,
    PatternCandidate,
    PatternSearchResult,
    check_linear_relu_legality,
    execute_graph,
    find_linear_relu_candidates,
    rewrite_linear_relu,
    simplify_graph,
    trace_model,
)
from .layer import Layer
from .masks import MaskSummary, OperatorMask, summarize_masks
from .model import Model
from .operator import Operator
from .optimization import record_linear_relu_semantic_fusion
from .sequential import Sequential
from .trace_registry import TraceRecord, TraceValidationError
from .tracing import (
    ArtifactReference,
    Attribute,
    BindingStatus,
    DecisionKind,
    EvidenceSource,
    ExecutionEvidence,
    ExecutionPlan,
    ImplementationBinding,
    LatencyUnit,
    OptimizationDecision,
    PlannedExecutionUnit,
    ValidationResult,
    ValidationStatus,
    ValueSpec,
)
from .verification import (
    GraphExecutionComparison,
    PlanEvidenceComparison,
    VerificationCheck,
    VerificationStatus,
    compare_graph_executions,
    compare_plan_to_evidence,
)

__all__ = (
    "Implementation",
    "ImplementationBinding",
    "Layer",
    "LatencyUnit",
    "Model",
    "MaskSummary",
    "Observation",
    "Operator",
    "OperatorMask",
    "OptimizationDecision",
    "PlanEvidenceComparison",
    "PlannedExecutionUnit",
    "SassEvidence",
    "Sequential",
    "summarize_masks",
    "ArtifactReference",
    "Attribute",
    "BindingStatus",
    "DecisionKind",
    "EvidenceSource",
    "ExecutionEvidence",
    "ExecutionGraph",
    "ExecutionPlan",
    "FrontendRewriteMetrics",
    "GraphExecutionComparison",
    "GraphNode",
    "GraphSimplification",
    "LegalityResult",
    "PatternCandidate",
    "PatternSearchResult",
    "TraceRecord",
    "TraceValidationError",
    "ValidationResult",
    "ValidationStatus",
    "ValueSpec",
    "VerificationCheck",
    "VerificationStatus",
    "check_linear_relu_legality",
    "compare_graph_executions",
    "compare_plan_to_evidence",
    "execute_graph",
    "find_linear_relu_candidates",
    "rewrite_linear_relu",
    "record_linear_relu_semantic_fusion",
    "simplify_graph",
    "trace_model",
)
