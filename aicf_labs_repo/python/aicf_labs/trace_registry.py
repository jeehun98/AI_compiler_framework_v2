"""In-memory reference validation for execution trace records."""

from dataclasses import dataclass

from .tracing import (
    ExecutionEvidence,
    ExecutionPlan,
    ImplementationBinding,
    OptimizationDecision,
    PlannedExecutionUnit,
    _validate_stable_id,
)


class TraceValidationError(ValueError):
    """Raised when trace IDs or cross-record references are inconsistent."""


@dataclass(frozen=True)
class TraceRecord:
    """A small aggregate that validates logical, plan, lowering, and evidence refs."""

    logical_operator_ids: tuple[str, ...]
    decisions: tuple[OptimizationDecision, ...] = ()
    plans: tuple[ExecutionPlan, ...] = ()
    bindings: tuple[ImplementationBinding, ...] = ()
    evidence: tuple[ExecutionEvidence, ...] = ()

    def __post_init__(self) -> None:
        self._validate_tuple(
            self.logical_operator_ids,
            str,
            "TraceRecord.logical_operator_ids",
        )
        for logical_operator_id in self.logical_operator_ids:
            try:
                _validate_stable_id(
                    logical_operator_id,
                    "TraceRecord.logical_operator_ids",
                )
            except (TypeError, ValueError) as error:
                raise TraceValidationError(str(error)) from error
        self._validate_tuple(
            self.decisions,
            OptimizationDecision,
            "TraceRecord.decisions",
        )
        self._validate_tuple(self.plans, ExecutionPlan, "TraceRecord.plans")
        self._validate_tuple(
            self.bindings,
            ImplementationBinding,
            "TraceRecord.bindings",
        )
        self._validate_tuple(
            self.evidence,
            ExecutionEvidence,
            "TraceRecord.evidence",
        )
        self.validate()

    @staticmethod
    def _validate_tuple(
        values: tuple[object, ...], expected_type: type, field_name: str
    ) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{field_name} must be a tuple")
        if not all(isinstance(value, expected_type) for value in values):
            raise TypeError(
                f"{field_name} must contain only {expected_type.__name__} values"
            )

    def validate(self) -> None:
        """Validate duplicate IDs and all internal trace references."""

        units = tuple(unit for plan in self.plans for unit in plan.units)
        categorized_ids = (
            ("logical operator", self.logical_operator_ids),
            ("decision", tuple(decision.id for decision in self.decisions)),
            ("plan", tuple(plan.id for plan in self.plans)),
            ("unit", tuple(unit.id for unit in units)),
            ("binding", tuple(binding.id for binding in self.bindings)),
            ("evidence", tuple(item.id for item in self.evidence)),
        )
        seen: dict[str, str] = {}
        for category, identifiers in categorized_ids:
            for identifier in identifiers:
                if identifier in seen:
                    raise TraceValidationError(
                        f"duplicate ID '{identifier}' appears as {seen[identifier]} "
                        f"and {category}"
                    )
                seen[identifier] = category

        logical_ids = set(self.logical_operator_ids)
        plan_ids = {plan.id for plan in self.plans}
        unit_ids = {unit.id for unit in units}
        decision_ids = {decision.id for decision in self.decisions}
        binding_by_id = {binding.id: binding for binding in self.bindings}

        valid_decision_inputs = logical_ids | unit_ids
        for decision in self.decisions:
            for input_id in decision.inputs:
                if input_id not in valid_decision_inputs:
                    raise TraceValidationError(
                        f"decision '{decision.id}' references missing input "
                        f"'{input_id}'"
                    )
            for output_id in decision.outputs:
                if output_id not in unit_ids:
                    raise TraceValidationError(
                        f"decision '{decision.id}' references missing output unit "
                        f"'{output_id}'"
                    )

        for unit in units:
            for logical_id in unit.logical_operator_ids:
                if logical_id not in logical_ids:
                    raise TraceValidationError(
                        f"unit '{unit.id}' references missing logical operator "
                        f"'{logical_id}'"
                    )
            for decision_id in unit.decision_ids:
                if decision_id not in decision_ids:
                    raise TraceValidationError(
                        f"unit '{unit.id}' references missing decision "
                        f"'{decision_id}'"
                    )
            if unit.implementation_binding_id is not None:
                binding = binding_by_id.get(unit.implementation_binding_id)
                if binding is None:
                    raise TraceValidationError(
                        f"unit '{unit.id}' references missing binding "
                        f"'{unit.implementation_binding_id}'"
                    )
                if binding.unit_id != unit.id:
                    raise TraceValidationError(
                        f"binding '{binding.id}' targets '{binding.unit_id}', not "
                        f"referencing unit '{unit.id}'"
                    )

        for binding in self.bindings:
            if binding.unit_id not in unit_ids:
                raise TraceValidationError(
                    f"binding '{binding.id}' references missing unit "
                    f"'{binding.unit_id}'"
                )

        valid_evidence_subjects = plan_ids | unit_ids | set(binding_by_id)
        for item in self.evidence:
            if item.subject_id not in valid_evidence_subjects:
                raise TraceValidationError(
                    f"evidence '{item.id}' references missing subject "
                    f"'{item.subject_id}'"
                )


__all__ = ("TraceRecord", "TraceValidationError")
