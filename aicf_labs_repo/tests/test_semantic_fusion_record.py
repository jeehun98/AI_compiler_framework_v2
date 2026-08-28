"""Connect a verified logical rewrite to existing optimization trace records."""

from dataclasses import replace
from pathlib import Path
import unittest

from aicf_labs import (
    BindingStatus,
    DecisionKind,
    Sequential,
    VerificationStatus,
    check_linear_relu_legality,
    compare_graph_executions,
    compare_plan_to_evidence,
    find_linear_relu_candidates,
    record_linear_relu_semantic_fusion,
    rewrite_linear_relu,
    select_implementation,
    trace_model,
)
from aicf_labs.layers import Linear, ReLU
from aicf_labs.operators import LinearReluOperator, ReluOperator


X = [[1, 2], [3, 4]]
W1 = [[1, -1, 2], [0, 3, -1]]
B1 = [1, 0, -2]
W2 = [[2], [-1], [3]]
B2 = [4]


def _verified_rewrite():
    original = trace_model(
        Sequential(
            Linear(2, 3, weight=W1, bias_value=B1),
            ReLU(),
            Linear(3, 1, weight=W2, bias_value=B2),
        ),
        X,
    )
    candidate = find_linear_relu_candidates(original).candidates[0]
    legality = check_linear_relu_legality(original, candidate)
    rewritten = rewrite_linear_relu(original, legality)
    verification = compare_graph_executions(
        original,
        rewritten,
        (X,),
        rel_tol=0.0,
        abs_tol=0.0,
    )
    return original, rewritten, legality, verification


class SemanticFusionRecordTests(unittest.TestCase):
    def test_records_the_verified_rewrite_as_semantic_fusion(self) -> None:
        original, rewritten, legality, verification = _verified_rewrite()

        trace = record_linear_relu_semantic_fusion(
            original,
            rewritten,
            legality,
            verification,
        )

        self.assertEqual(trace.logical_operator_ids, legality.candidate.node_ids)
        self.assertEqual(len(trace.decisions), 1)
        decision = trace.decisions[0]
        self.assertIs(decision.kind, DecisionKind.SEMANTIC_FUSION)
        self.assertEqual(decision.inputs, ("matmul.0", "add.0", "relu.0"))
        self.assertEqual(decision.outputs, ("plan.linearRelu.0.unit0",))
        self.assertEqual(
            dict((item.key, item.value) for item in decision.references),
            {
                "pattern_id": "linear-relu-fusion",
                "rewritten_node_id": "linearRelu.0",
                "verification": "reference_execution",
                "rel_tol": "0.0",
                "abs_tol": "0.0",
            },
        )

    def test_plan_unit_has_explicit_unbound_backend_state_without_runtime_claims(
        self,
    ) -> None:
        original, rewritten, legality, verification = _verified_rewrite()
        trace = record_linear_relu_semantic_fusion(
            original,
            rewritten,
            legality,
            verification,
        )
        unit = trace.plans[0].units[0]

        self.assertEqual(
            tuple(value.id for value in unit.inputs),
            ("input.0", "constant.0", "constant.1"),
        )
        self.assertEqual(tuple(value.shape for value in unit.inputs), ((2, 2), (2, 3), (3,)))
        self.assertEqual(unit.outputs[0].id, "linearRelu.0")
        self.assertEqual(unit.outputs[0].shape, (2, 3))
        self.assertEqual(unit.decision_ids, (trace.decisions[0].id,))
        self.assertIsNone(unit.expected_kernel_launches)
        self.assertEqual(
            unit.implementation_binding_id,
            "binding.linearRelu.0.unbound",
        )
        self.assertEqual(len(trace.bindings), 1)
        binding = trace.bindings[0]
        self.assertEqual(binding.id, unit.implementation_binding_id)
        self.assertEqual(binding.unit_id, unit.id)
        self.assertIs(binding.status, BindingStatus.UNBOUND)
        self.assertIsNone(binding.backend)
        self.assertIsNone(binding.target)
        self.assertIsNone(binding.implementation_ref)
        self.assertEqual(
            binding.selection_reason,
            "Backend and implementation resolution has not been performed.",
        )
        self.assertEqual(trace.evidence, ())
        self.assertIs(
            compare_plan_to_evidence(unit, None).kernel_launches.status,
            VerificationStatus.NOT_APPLICABLE,
        )

        relu_implementation = ReluOperator().implementations[0]
        with self.assertRaisesRegex(ValueError, "is not registered"):
            select_implementation(
                trace,
                unit,
                LinearReluOperator(),
                relu_implementation,
                backend="cuda",
            )
        self.assertIs(binding.status, BindingStatus.UNBOUND)

    def test_unverified_equivalence_cannot_be_recorded(self) -> None:
        original, rewritten, legality, verification = _verified_rewrite()
        failed = replace(verification, equivalent=False)

        with self.assertRaisesRegex(ValueError, "verified equivalence"):
            record_linear_relu_semantic_fusion(
                original,
                rewritten,
                legality,
                failed,
            )

    def test_primitive_cuda_assets_do_not_select_a_fused_implementation(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        for operator_name in ("gemm", "add", "relu"):
            source_path = (
                repository_root
                / "operators"
                / operator_name
                / f"{operator_name}.cu"
            )
            self.assertTrue(source_path.is_file())
        self.assertFalse((repository_root / "operators" / "linear_relu").exists())
        self.assertEqual(LinearReluOperator().implementations, ())

        original, rewritten, legality, verification = _verified_rewrite()
        trace = record_linear_relu_semantic_fusion(
            original,
            rewritten,
            legality,
            verification,
        )
        unit = trace.plans[0].units[0]
        binding = trace.bindings[0]

        self.assertIs(binding.status, BindingStatus.UNBOUND)
        self.assertIsNone(binding.backend)
        self.assertIsNone(binding.implementation_ref)
        self.assertIsNone(unit.expected_kernel_launches)
        self.assertEqual(trace.evidence, ())
        self.assertIs(
            compare_plan_to_evidence(unit, None).kernel_launches.status,
            VerificationStatus.NOT_APPLICABLE,
        )

    def test_a_different_rewritten_graph_cannot_be_recorded(self) -> None:
        original, rewritten, legality, verification = _verified_rewrite()

        with self.assertRaisesRegex(ValueError, "does not match"):
            record_linear_relu_semantic_fusion(
                original,
                original,
                legality,
                verification,
            )


if __name__ == "__main__":
    unittest.main()
