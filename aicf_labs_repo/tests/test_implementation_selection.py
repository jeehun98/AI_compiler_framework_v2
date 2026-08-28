"""Production explicit implementation selection stays separate from execution."""

from dataclasses import replace
import unittest

from aicf_labs import (
    BindingStatus,
    Implementation,
    ImplementationBinding,
    ExecutionPlan,
    PlannedExecutionUnit,
    TraceRecord,
    ValueSpec,
    VerificationStatus,
    compare_plan_to_evidence,
    select_implementation,
)
from aicf_labs.operators import ReluOperator


def _unbound_relu_trace(
    *,
    input_dtype: str = "fp32",
) -> tuple[TraceRecord, PlannedExecutionUnit]:
    unit = PlannedExecutionUnit(
        id="plan.relu.fp32.unit0",
        logical_operator_ids=("relu.0",),
        inputs=(ValueSpec("value.relu.input", shape=(32,), dtype=input_dtype),),
        outputs=(ValueSpec("value.relu.output", shape=(32,), dtype="fp32"),),
        expected_kernel_launches=None,
        implementation_binding_id="binding.relu.0",
    )
    binding = ImplementationBinding(
        id="binding.relu.0",
        unit_id=unit.id,
        backend=None,
        status=BindingStatus.UNBOUND,
    )
    trace = TraceRecord(
        logical_operator_ids=("relu.0",),
        plans=(ExecutionPlan("plan.relu.fp32", (unit,)),),
        bindings=(binding,),
    )
    return trace, unit


class ExplicitImplementationSelectionTests(unittest.TestCase):
    def test_explicitly_selects_registered_relu_implementation(self) -> None:
        trace, unit = _unbound_relu_trace()
        operator = ReluOperator()
        implementation = operator.implementations[0]

        selected_trace = select_implementation(
            trace,
            unit,
            operator,
            implementation,
            backend="cuda",
            target="sm_86",
        )

        self.assertIs(trace.bindings[0].status, BindingStatus.UNBOUND)
        self.assertEqual(selected_trace.plans, trace.plans)
        self.assertEqual(
            selected_trace.plans[0].units[0].implementation_binding_id,
            "binding.relu.0",
        )
        selected = selected_trace.bindings[0]
        self.assertIs(selected.status, BindingStatus.SELECTED)
        self.assertEqual(selected.backend, "cuda")
        self.assertEqual(selected.target, "sm_86")
        self.assertEqual(
            selected.implementation_ref,
            "operator:relu:implementation:fp32_scalar",
        )
        self.assertEqual(
            {item.key: item.value for item in selected.configuration},
            {"selection_mode": "explicit"},
        )
        self.assertIsNone(unit.expected_kernel_launches)
        self.assertEqual(selected_trace.evidence, ())
        self.assertIs(
            compare_plan_to_evidence(unit, None).kernel_launches.status,
            VerificationStatus.NOT_APPLICABLE,
        )

    def test_rejects_an_unregistered_implementation(self) -> None:
        trace, unit = _unbound_relu_trace()
        operator = ReluOperator()
        missing = Implementation(
            name="missing",
            source_file="operators/relu/missing.cu",
            kernel_name="missing_relu",
            input_dtype="fp32",
            output_dtype="fp32",
        )

        with self.assertRaisesRegex(ValueError, "is not registered"):
            select_implementation(
                trace,
                unit,
                operator,
                missing,
                backend="cuda",
            )

    def test_rejects_backend_and_known_dtype_mismatches(self) -> None:
        trace, unit = _unbound_relu_trace()
        operator = ReluOperator()
        implementation = operator.implementations[0]

        with self.assertRaisesRegex(ValueError, "only the 'cuda' backend"):
            select_implementation(
                trace,
                unit,
                operator,
                implementation,
                backend="cpu",
            )

        fp16_trace, fp16_unit = _unbound_relu_trace(input_dtype="fp16")
        with self.assertRaisesRegex(ValueError, "input dtype is incompatible"):
            select_implementation(
                fp16_trace,
                fp16_unit,
                operator,
                implementation,
                backend="cuda",
            )

    def test_rejects_a_unit_or_binding_state_mismatch(self) -> None:
        trace, unit = _unbound_relu_trace()
        operator = ReluOperator()
        implementation = operator.implementations[0]
        unrelated = replace(unit, id="plan.relu.other.unit0")

        with self.assertRaisesRegex(ValueError, "does not match a unit"):
            select_implementation(
                trace,
                unrelated,
                operator,
                implementation,
                backend="cuda",
            )

        selected = select_implementation(
            trace,
            unit,
            operator,
            implementation,
            backend="cuda",
        )
        with self.assertRaisesRegex(ValueError, "requires an UNBOUND binding"):
            select_implementation(
                selected,
                unit,
                operator,
                implementation,
                backend="cuda",
            )


if __name__ == "__main__":
    unittest.main()
