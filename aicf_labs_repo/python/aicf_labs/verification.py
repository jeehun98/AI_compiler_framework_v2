"""Small, explicit comparisons between planned and observed execution facts."""

from dataclasses import dataclass
from enum import Enum

from .tracing import ExecutionEvidence, ExecutionPlan, PlannedExecutionUnit


class VerificationStatus(Enum):
    """Outcome of comparing one expected fact with available observation."""

    CONFIRMED = "confirmed"
    MISMATCH = "mismatch"
    UNOBSERVED = "unobserved"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class VerificationCheck:
    """Expected and observed values for one verification dimension."""

    status: VerificationStatus
    expected: int | None
    observed: int | None
    details: str


@dataclass(frozen=True)
class PlanEvidenceComparison:
    """Extensible comparison result; currently checks kernel launch count only."""

    plan_id: str
    evidence_id: str | None
    kernel_launches: VerificationCheck


def _expected_kernel_launches(
    plan: ExecutionPlan | PlannedExecutionUnit,
) -> int | None:
    if isinstance(plan, PlannedExecutionUnit):
        return plan.expected_kernel_launches
    if isinstance(plan, ExecutionPlan):
        expected_values = [unit.expected_kernel_launches for unit in plan.units]
        if any(value is None for value in expected_values):
            return None
        return sum(value for value in expected_values if value is not None)
    raise TypeError(
        "plan must be an ExecutionPlan or PlannedExecutionUnit"
    )


def compare_plan_to_evidence(
    plan: ExecutionPlan | PlannedExecutionUnit,
    evidence: ExecutionEvidence | None,
) -> PlanEvidenceComparison:
    """Compare expected and observed launches without assuming missing data failed."""

    if not isinstance(plan, (ExecutionPlan, PlannedExecutionUnit)):
        raise TypeError(
            "plan must be an ExecutionPlan or PlannedExecutionUnit"
        )
    if evidence is not None and not isinstance(evidence, ExecutionEvidence):
        raise TypeError("evidence must be ExecutionEvidence or None")
    if evidence is not None:
        if isinstance(plan, PlannedExecutionUnit):
            valid_subject_ids = {plan.id}
            if plan.implementation_binding_id is not None:
                valid_subject_ids.add(plan.implementation_binding_id)
        else:
            valid_subject_ids = {plan.id}
            if len(plan.units) == 1:
                valid_subject_ids.add(plan.units[0].id)
                if plan.units[0].implementation_binding_id is not None:
                    valid_subject_ids.add(
                        plan.units[0].implementation_binding_id
                    )
        if evidence.subject_id not in valid_subject_ids:
            raise ValueError(
                f"evidence subject '{evidence.subject_id}' is unrelated to "
                f"plan '{plan.id}'"
            )

    expected = _expected_kernel_launches(plan)
    observed = (
        evidence.observed_kernel_launches if evidence is not None else None
    )
    if expected is None:
        status = VerificationStatus.NOT_APPLICABLE
        details = "The plan does not define an expected kernel launch count."
    elif observed is None:
        status = VerificationStatus.UNOBSERVED
        details = "No kernel launch count has been observed."
    elif expected == observed:
        status = VerificationStatus.CONFIRMED
        details = "Expected and observed kernel launch counts match."
    else:
        status = VerificationStatus.MISMATCH
        details = "Expected and observed kernel launch counts differ."

    plan_id = plan.id
    return PlanEvidenceComparison(
        plan_id=plan_id,
        evidence_id=evidence.id if evidence is not None else None,
        kernel_launches=VerificationCheck(
            status=status,
            expected=expected,
            observed=observed,
            details=details,
        ),
    )


__all__ = (
    "PlanEvidenceComparison",
    "VerificationCheck",
    "VerificationStatus",
    "compare_plan_to_evidence",
)
