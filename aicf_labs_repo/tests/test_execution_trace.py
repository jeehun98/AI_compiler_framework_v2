"""Repository-independent tests for plan, lowering, and evidence records."""

import unittest

from aicf_labs import (
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
    TraceRecord,
    TraceValidationError,
    ValidationResult,
    ValidationStatus,
    ValueSpec,
    VerificationStatus,
    compare_plan_to_evidence,
)


def make_add_unit(
    *,
    binding_id: str | None = "lowering.add.cuda.sm86",
    expected_kernel_launches: int | None = 1,
) -> PlannedExecutionUnit:
    return PlannedExecutionUnit(
        id="plan.add.fp32.unit0",
        logical_operator_ids=("add.0",),
        inputs=(
            ValueSpec(id="value.a", shape=("N",), dtype="fp32", layout="dense"),
            ValueSpec(id="value.b", shape=("N",), dtype="fp32", layout="dense"),
        ),
        outputs=(
            ValueSpec(id="value.y", shape=("N",), dtype="fp32", layout="dense"),
        ),
        expected_kernel_launches=expected_kernel_launches,
        implementation_binding_id=binding_id,
    )


def make_add_binding(
    *,
    binding_id: str = "lowering.add.cuda.sm86",
    unit_id: str = "plan.add.fp32.unit0",
) -> ImplementationBinding:
    return ImplementationBinding(
        id=binding_id,
        unit_id=unit_id,
        backend="cuda",
        target="sm_86",
        implementation_ref="operator:add",
        selection_reason="Use the existing scalar FP32 add experiment.",
        status=BindingStatus.SELECTED,
        configuration=(Attribute("dtype", "fp32"),),
    )


def make_launch_evidence(
    observed_kernel_launches: int | None,
    *,
    evidence_id: str = "evidence.add.sm86.synthetic",
) -> ExecutionEvidence:
    return ExecutionEvidence(
        id=evidence_id,
        subject_id="lowering.add.cuda.sm86",
        sources=(EvidenceSource.RUNTIME_TRACE,),
        observed_kernel_launches=observed_kernel_launches,
        observed_kernel_names=("add_fp32",),
    )


class ExecutionTraceTests(unittest.TestCase):
    def test_single_operator_execution_plan(self) -> None:
        unit = make_add_unit()
        plan = ExecutionPlan(id="plan.add.fp32", units=(unit,))

        self.assertEqual(plan.units, (unit,))
        self.assertEqual(unit.logical_operator_ids, ("add.0",))
        self.assertEqual(unit.expected_kernel_launches, 1)
        self.assertEqual(unit.outputs[0].dtype, "fp32")

    def test_semantic_fusion_decision_and_planned_unit(self) -> None:
        decision = OptimizationDecision(
            id="decision.linear_bias_relu",
            kind=DecisionKind.SEMANTIC_FUSION,
            inputs=("matmul.0", "add.0", "relu.0"),
            outputs=("plan.linear_bias_relu.fp32.unit0",),
            preconditions=(
                "add.0 is the sole consumer of matmul.0",
                "relu.0 is the sole consumer of add.0",
                "bias broadcast is supported",
            ),
            semantic_preservation=(
                "operator order and FP32 special-value policy are preserved",
            ),
            expected_effects=(
                "reduce kernel launches",
                "remove intermediate global-memory traffic",
            ),
        )
        unit = PlannedExecutionUnit(
            id="plan.linear_bias_relu.fp32.unit0",
            logical_operator_ids=("matmul.0", "add.0", "relu.0"),
            inputs=(
                ValueSpec(id="value.x", shape=("B", 128), dtype="fp32"),
                ValueSpec(id="value.w", shape=(128, 64), dtype="fp32"),
                ValueSpec(id="value.bias", shape=(64,), dtype="fp32"),
            ),
            outputs=(
                ValueSpec(id="value.output", shape=("B", 64), dtype="fp32"),
            ),
            decision_ids=(decision.id,),
            expected_kernel_launches=1,
            implementation_binding_id="lowering.linear_bias_relu.cuda.sm86",
        )
        binding = ImplementationBinding(
            id="lowering.linear_bias_relu.cuda.sm86",
            unit_id=unit.id,
            backend="cuda",
            target="sm_86",
            status=BindingStatus.UNBOUND,
            selection_reason=(
                "No verified fused MatMul+bias+ReLU implementation is available."
            ),
        )
        plan = ExecutionPlan(id="plan.linear_bias_relu.fp32", units=(unit,))
        trace = TraceRecord(
            logical_operator_ids=("matmul.0", "add.0", "relu.0"),
            decisions=(decision,),
            plans=(plan,),
            bindings=(binding,),
        )

        self.assertEqual(len(trace.plans[0].units[0].logical_operator_ids), 3)
        self.assertIs(binding.status, BindingStatus.UNBOUND)
        self.assertIsNone(binding.implementation_ref)
        comparison = compare_plan_to_evidence(unit, None)
        self.assertIs(
            comparison.kernel_launches.status,
            VerificationStatus.UNOBSERVED,
        )

    def test_selected_implementation_binding(self) -> None:
        binding = make_add_binding()
        self.assertIs(binding.status, BindingStatus.SELECTED)
        self.assertEqual(binding.implementation_ref, "operator:add")

    def test_execution_evidence_can_keep_unobserved_fields_none(self) -> None:
        evidence = ExecutionEvidence(
            id="evidence.add.sm86.partial",
            subject_id="lowering.add.cuda.sm86",
            sources=(EvidenceSource.BINARY_SASS,),
            observed_instruction_features=("FADD",),
        )
        self.assertIsNone(evidence.observed_kernel_launches)
        self.assertIsNone(evidence.observed_kernel_names)
        self.assertIsNone(evidence.latency)
        self.assertEqual(evidence.observed_instruction_features, ("FADD",))

    def test_execution_evidence_records_latency_and_validation(self) -> None:
        evidence = ExecutionEvidence(
            id="evidence.add.sm86.benchmark",
            subject_id="lowering.add.cuda.sm86",
            sources=(EvidenceSource.BENCHMARK, EvidenceSource.VALIDATION),
            latency=26.0,
            latency_unit=LatencyUnit.MICROSECONDS,
            validation=ValidationResult(
                ValidationStatus.PASSED,
                "FP32 reference comparison passed.",
            ),
        )
        self.assertEqual(evidence.latency, 26.0)
        self.assertIs(evidence.latency_unit, LatencyUnit.MICROSECONDS)
        self.assertIs(evidence.validation.status, ValidationStatus.PASSED)

    def test_kernel_launch_count_match(self) -> None:
        comparison = compare_plan_to_evidence(
            make_add_unit(),
            make_launch_evidence(1),
        )
        self.assertIs(
            comparison.kernel_launches.status,
            VerificationStatus.CONFIRMED,
        )
        self.assertEqual(comparison.kernel_launches.expected, 1)
        self.assertEqual(comparison.kernel_launches.observed, 1)

    def test_kernel_launch_count_mismatch(self) -> None:
        comparison = compare_plan_to_evidence(
            make_add_unit(),
            make_launch_evidence(2),
        )
        self.assertIs(
            comparison.kernel_launches.status,
            VerificationStatus.MISMATCH,
        )
        self.assertEqual(comparison.kernel_launches.observed, 2)

    def test_missing_evidence_is_unobserved(self) -> None:
        comparison = compare_plan_to_evidence(make_add_unit(), None)
        self.assertIs(
            comparison.kernel_launches.status,
            VerificationStatus.UNOBSERVED,
        )

    def test_comparison_rejects_unrelated_evidence(self) -> None:
        unrelated = ExecutionEvidence(
            id="evidence.relu.synthetic",
            subject_id="lowering.relu.cuda.sm86",
            sources=(EvidenceSource.RUNTIME_TRACE,),
            observed_kernel_launches=1,
        )
        with self.assertRaisesRegex(ValueError, "unrelated"):
            compare_plan_to_evidence(make_add_unit(), unrelated)

    def test_undefined_expectation_is_not_applicable(self) -> None:
        comparison = compare_plan_to_evidence(
            make_add_unit(expected_kernel_launches=None),
            make_launch_evidence(1),
        )
        self.assertIs(
            comparison.kernel_launches.status,
            VerificationStatus.NOT_APPLICABLE,
        )

    def test_trace_record_accepts_plan_lowering_and_evidence(self) -> None:
        unit = make_add_unit()
        plan = ExecutionPlan(id="plan.add.fp32", units=(unit,))
        binding = make_add_binding()
        evidence = make_launch_evidence(1)

        trace = TraceRecord(
            logical_operator_ids=("add.0",),
            plans=(plan,),
            bindings=(binding,),
            evidence=(evidence,),
        )
        self.assertEqual(trace.evidence[0].subject_id, binding.id)

    def test_duplicate_ids_are_rejected(self) -> None:
        unit = make_add_unit(binding_id=None)
        plan = ExecutionPlan(id="plan.add.fp32", units=(unit,))
        first = make_add_binding(binding_id="lowering.duplicate")
        second = make_add_binding(binding_id="lowering.duplicate")

        with self.assertRaisesRegex(TraceValidationError, "duplicate ID"):
            TraceRecord(
                logical_operator_ids=("add.0",),
                plans=(plan,),
                bindings=(first, second),
            )

    def test_logical_operator_ids_must_be_stable_ids(self) -> None:
        with self.assertRaisesRegex(TraceValidationError, "stable ID"):
            TraceRecord(logical_operator_ids=("operators/add/add.cu",))

    def test_broken_references_are_rejected(self) -> None:
        unit = make_add_unit(binding_id=None)
        plan = ExecutionPlan(id="plan.add.fp32", units=(unit,))
        broken_binding = make_add_binding(unit_id="plan.missing.unit")

        with self.assertRaisesRegex(TraceValidationError, "missing unit"):
            TraceRecord(
                logical_operator_ids=("add.0",),
                plans=(plan,),
                bindings=(broken_binding,),
            )

        broken_evidence = ExecutionEvidence(
            id="evidence.broken",
            subject_id="lowering.missing",
            sources=(EvidenceSource.RUNTIME_TRACE,),
        )
        with self.assertRaisesRegex(TraceValidationError, "missing subject"):
            TraceRecord(
                logical_operator_ids=("add.0",),
                plans=(plan,),
                evidence=(broken_evidence,),
            )

    def test_artifact_paths_must_be_safe_and_repository_relative(self) -> None:
        invalid_paths = (
            "../outside.sass",
            "/absolute/add.sass",
            "C:/absolute/add.sass",
            "operators\\add\\artifacts\\add.sass",
            "./operators/add/artifacts/add.sass",
        )
        for path in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    ArtifactReference(path, EvidenceSource.BINARY_SASS)


if __name__ == "__main__":
    unittest.main()
