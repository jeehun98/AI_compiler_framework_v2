"""Explicit implementation selection for immutable execution traces."""

from dataclasses import replace

from .implementation import Implementation
from .operator import Operator
from .trace_registry import TraceRecord
from .tracing import Attribute, BindingStatus, PlannedExecutionUnit


def _registered_implementation(
    operator: Operator,
    implementation: Implementation,
) -> None:
    same_name = tuple(
        candidate
        for candidate in operator.implementations
        if candidate.name == implementation.name
    )
    if implementation not in same_name:
        raise ValueError(
            f"implementation '{implementation.name}' is not registered on "
            f"operator '{operator.name}'"
        )
    if len(same_name) != 1:
        raise ValueError(
            f"implementation name '{implementation.name}' is ambiguous on "
            f"operator '{operator.name}'"
        )


def _validate_known_dtypes(
    unit: PlannedExecutionUnit,
    implementation: Implementation,
) -> None:
    incompatible_inputs = tuple(
        value.dtype
        for value in unit.inputs
        if value.dtype is not None and value.dtype != implementation.input_dtype
    )
    if incompatible_inputs:
        raise ValueError(
            "planned input dtype is incompatible with implementation input dtype "
            f"'{implementation.input_dtype}'"
        )
    incompatible_outputs = tuple(
        value.dtype
        for value in unit.outputs
        if value.dtype is not None and value.dtype != implementation.output_dtype
    )
    if incompatible_outputs:
        raise ValueError(
            "planned output dtype is incompatible with implementation output dtype "
            f"'{implementation.output_dtype}'"
        )


def select_implementation(
    trace: TraceRecord,
    unit: PlannedExecutionUnit,
    operator: Operator,
    implementation: Implementation,
    *,
    backend: str,
    target: str | None = None,
) -> TraceRecord:
    """Explicitly select one registered implementation without executing it."""

    if not isinstance(trace, TraceRecord):
        raise TypeError("trace must be a TraceRecord")
    if not isinstance(unit, PlannedExecutionUnit):
        raise TypeError("unit must be a PlannedExecutionUnit")
    if not isinstance(operator, Operator):
        raise TypeError("operator must be an Operator")
    if not isinstance(implementation, Implementation):
        raise TypeError("implementation must be an Implementation")

    trace_units = tuple(
        candidate for plan in trace.plans for candidate in plan.units
    )
    matching_units = tuple(
        candidate for candidate in trace_units if candidate.id == unit.id
    )
    if len(matching_units) != 1 or matching_units[0] != unit:
        raise ValueError(f"unit '{unit.id}' does not match a unit in the trace")
    if unit.implementation_binding_id is None:
        raise ValueError("unit must reference an UNBOUND implementation binding")

    current = next(
        (
            binding
            for binding in trace.bindings
            if binding.id == unit.implementation_binding_id
        ),
        None,
    )
    if current is None or current.unit_id != unit.id:
        raise ValueError("unit does not reference a matching implementation binding")
    if current.status is not BindingStatus.UNBOUND:
        raise ValueError("explicit selection requires an UNBOUND binding")

    _registered_implementation(operator, implementation)
    if backend != "cuda":
        raise ValueError(
            "the current Implementation model describes only the 'cuda' backend"
        )
    if current.backend is not None and current.backend != backend:
        raise ValueError("selected backend conflicts with the existing binding")
    if current.target is not None and target is not None and current.target != target:
        raise ValueError("selected target conflicts with the existing binding")
    if len(unit.inputs) != operator.arity:
        raise ValueError(
            f"unit input count {len(unit.inputs)} does not match operator arity "
            f"{operator.arity}"
        )
    _validate_known_dtypes(unit, implementation)

    if any(item.key == "selection_mode" for item in current.configuration):
        raise ValueError("selection_mode is reserved for implementation selection")
    implementation_ref = (
        f"operator:{operator.name}:implementation:{implementation.name}"
    )
    selected = replace(
        current,
        backend=backend,
        target=current.target if target is None else target,
        status=BindingStatus.SELECTED,
        implementation_ref=implementation_ref,
        selection_reason=(
            f"Explicitly selected '{implementation.name}' from operator "
            f"'{operator.name}'."
        ),
        configuration=current.configuration
        + (Attribute("selection_mode", "explicit"),),
    )
    bindings = tuple(
        selected if binding.id == current.id else binding
        for binding in trace.bindings
    )
    return replace(trace, bindings=bindings)


__all__ = ("select_implementation",)
