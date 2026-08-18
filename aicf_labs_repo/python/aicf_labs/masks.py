"""Typed states used by operator and implementation properties."""

from dataclasses import dataclass
from enum import Enum


class State(Enum):
    """Knowledge state for a semantic or implementation property."""

    YES = "yes"
    NO = "no"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class Observation(Enum):
    """Inspection status for a concrete piece of implementation evidence."""

    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    NOT_INSPECTED = "not_inspected"
    NOT_APPLICABLE = "not_applicable"


class Monotonicity(Enum):
    """Direction-aware monotonicity that must not be reduced to a Boolean."""

    STRICTLY_INCREASING = "strictly_increasing"
    NONDECREASING = "nondecreasing"
    STRICTLY_DECREASING = "strictly_decreasing"
    NONINCREASING = "nonincreasing"
    NOT_MONOTONIC = "not_monotonic"
    CONDITIONAL = "conditional"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperatorMask:
    """Typed operator properties used by later matching and optimization work."""

    elementwise: State = State.UNKNOWN
    reduction: State = State.UNKNOWN
    shape_preserving: State = State.UNKNOWN
    rank_preserving: State = State.UNKNOWN
    element_independent: State = State.UNKNOWN

    linear: State = State.UNKNOWN
    idempotent: State = State.UNKNOWN
    zero_preserving: State = State.UNKNOWN
    invertible: State = State.UNKNOWN
    monotonicity: Monotonicity = Monotonicity.UNKNOWN

    producer_fusible: State = State.UNKNOWN
    consumer_fusible: State = State.UNKNOWN
    epilogue_fusible: State = State.UNKNOWN

    requires_materialization: State = State.UNKNOWN
    requires_global_sync: State = State.UNKNOWN


@dataclass(frozen=True)
class HardwareMask:
    """Hardware properties of one implementation, never of an operator itself."""

    uses_cuda_core: State = State.UNKNOWN
    uses_tensor_core: State = State.UNKNOWN
    uses_sfu: State = State.UNKNOWN
    uses_shared_memory: State = State.UNKNOWN
    uses_barrier: State = State.UNKNOWN
    uses_atomic: State = State.UNKNOWN
