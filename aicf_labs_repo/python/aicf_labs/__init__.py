"""Minimal declarative model, layer, operator, and implementation API."""

from .implementation import Implementation, SassEvidence
from .layer import Layer
from .masks import HardwareMask, Monotonicity, Observation, OperatorMask, State
from .model import Model
from .operator import Operator
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
    PlanEvidenceComparison,
    VerificationCheck,
    VerificationStatus,
    compare_plan_to_evidence,
)

__all__ = (
    "HardwareMask",
    "Implementation",
    "ImplementationBinding",
    "Layer",
    "LatencyUnit",
    "Model",
    "Monotonicity",
    "Observation",
    "Operator",
    "OperatorMask",
    "OptimizationDecision",
    "PlanEvidenceComparison",
    "PlannedExecutionUnit",
    "SassEvidence",
    "Sequential",
    "State",
    "ArtifactReference",
    "Attribute",
    "BindingStatus",
    "DecisionKind",
    "EvidenceSource",
    "ExecutionEvidence",
    "ExecutionPlan",
    "TraceRecord",
    "TraceValidationError",
    "ValidationResult",
    "ValidationStatus",
    "ValueSpec",
    "VerificationCheck",
    "VerificationStatus",
    "compare_plan_to_evidence",
)
