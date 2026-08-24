"""End-to-end model execution, tracing, rewriting, and verification."""

import json
from pathlib import Path
import unittest

from aicf_labs import (
    Model,
    OperatorMask,
    compare_graph_executions,
    execute_graph,
    simplify_graph,
    trace_model,
)
from aicf_labs.ops import add, matmul, mul, relu


X = [[1.0, -2.0], [3.0, 4.0]]
W = [[2.0, 1.0], [-1.0, 3.0]]


class ToyModel(Model):
    def forward(self, x: object, w: object) -> object:
        value = matmul(x, w)
        value = add(value, 0)
        value = relu(value)
        value = mul(value, 1)
        return value


class FrontendEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ToyModel()
        self.graph = trace_model(self.model, X, W, input_names=("x", "w"))

    def test_model_executes_through_public_operators(self) -> None:
        self.assertEqual(self.model(X, W), [[4.0, 0], [2.0, 15.0]])

    def test_model_execution_produces_expected_primitive_graph(self) -> None:
        self.assertEqual(
            self.graph.operator_sequence,
            ("matmul", "add", "relu", "mul"),
        )
        self.assertEqual(self.graph.inputs, ("input.0", "input.1"))
        self.assertEqual(self.graph.outputs, ("mul.0",))
        self.assertEqual(self.graph.node("matmul.0").input_ids, ("input.0", "input.1"))
        self.assertEqual(self.graph.node("add.0").input_ids, ("matmul.0", "constant.0"))
        self.assertEqual(self.graph.node("relu.0").input_ids, ("add.0",))
        self.assertEqual(self.graph.node("mul.0").input_ids, ("relu.0", "constant.1"))

    def test_constants_and_canonical_operator_masks_are_preserved(self) -> None:
        self.assertEqual(self.graph.node("constant.0").constant_value, 0)
        self.assertEqual(self.graph.node("constant.1").constant_value, 1)
        self.assertEqual(self.graph.node("matmul.0").mask, matmul.mask)
        self.assertEqual(self.graph.node("add.0").mask, add.mask)
        self.assertEqual(self.graph.node("relu.0").mask, relu.mask)
        self.assertEqual(self.graph.node("mul.0").mask, mul.mask)
        required = OperatorMask.ELEMENTWISE | OperatorMask.PURE
        self.assertTrue(self.graph.node("add.0").mask.matches(required))
        self.assertTrue(self.graph.node("relu.0").mask.matches(required))
        self.assertTrue(self.graph.node("mul.0").mask.matches(required))

    def test_serialized_graph_is_the_shared_web_fixture(self) -> None:
        fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "toy_model_graph.json"
        expected = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(self.graph.to_web_graph(), expected)

    def test_rewrite_simplifies_actual_trace_and_preserves_meaning(self) -> None:
        original_output = execute_graph(self.graph, (X, W))
        simplification = simplify_graph(self.graph)

        self.assertEqual(simplification.graph.operator_sequence, ("matmul", "relu"))
        self.assertEqual(simplification.removed_nodes, ("add.0", "mul.0"))
        self.assertEqual(simplification.node_count_before, 8)
        self.assertEqual(simplification.node_count_after, 6)
        self.assertEqual(simplification.rewrite_count, 2)

        comparison = compare_graph_executions(
            self.graph,
            simplification.graph,
            (X, W),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        self.assertTrue(comparison.equivalent)
        self.assertEqual(comparison.original_output, original_output)
        self.assertEqual(comparison.rewritten_output, original_output)

    def test_mask_screening_metrics_are_separate_from_rule_conditions(self) -> None:
        simplification = simplify_graph(self.graph)
        add_zero, mul_one = simplification.metrics

        self.assertEqual(
            (add_zero.nodes_scanned, add_zero.mask_accepted, add_zero.condition_accepted),
            (8, 3, 1),
        )
        self.assertEqual(add_zero.rewrites_applied, 1)
        self.assertAlmostEqual(add_zero.candidate_reduction, 5 / 8)
        self.assertEqual(
            (mul_one.nodes_scanned, mul_one.mask_accepted, mul_one.condition_accepted),
            (7, 2, 1),
        )
        self.assertEqual(mul_one.rewrites_applied, 1)
        self.assertAlmostEqual(mul_one.candidate_reduction, 5 / 7)

        # The same mask accepts Mul(x, 2), but the rule-specific identity
        # condition rejects it and no rewrite occurs.
        class NonIdentityModel(Model):
            def forward(self, x: object) -> object:
                return mul(x, 2)

        rejected = simplify_graph(trace_model(NonIdentityModel(), [1.0, 2.0]))
        rejected_mul_one = rejected.metrics[1]
        self.assertEqual(rejected_mul_one.mask_accepted, 1)
        self.assertEqual(rejected_mul_one.condition_accepted, 0)
        self.assertEqual(rejected_mul_one.rewrites_applied, 0)
        self.assertEqual(rejected.rewrite_count, 0)


if __name__ == "__main__":
    unittest.main()
