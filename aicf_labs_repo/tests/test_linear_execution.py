"""Regression tests for Linear execution through primitive operators."""

import unittest

from aicf_labs import Model, execute_graph, trace_model
from aicf_labs.layers import Linear


X = [[1, 2], [3, 4]]
WEIGHT = [[2, 1], [-1, 3]]
BIAS = [1, -2]
EXPECTED = [[1, 5], [3, 13]]


class LinearModel(Model):
    def __init__(self) -> None:
        self.linear = Linear(
            2,
            2,
            weight=WEIGHT,
            bias_value=BIAS,
        )

    def forward(self, value: object) -> object:
        return self.linear(value)


class LinearExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LinearModel()
        self.graph = trace_model(self.model, X, input_names=("x",))

    def test_linear_eager_execution_is_exact(self) -> None:
        self.assertEqual(self.model(X), EXPECTED)

    def test_trace_contains_only_primitive_computation_nodes(self) -> None:
        self.assertEqual(self.graph.operator_sequence, ("matmul", "add"))
        self.assertNotIn("linear", self.graph.operator_sequence)
        self.assertEqual(self.graph.inputs, ("input.0",))
        self.assertEqual(self.graph.outputs, ("add.0",))

        weight, matmul, bias, add = self.graph.nodes[1:]
        self.assertEqual(weight.operator_id, "constant")
        self.assertEqual(weight.constant_value, ((2, 1), (-1, 3)))
        self.assertEqual(weight.shape, (2, 2))
        self.assertEqual(matmul.input_ids, ("input.0", weight.id))
        self.assertEqual(bias.operator_id, "constant")
        self.assertEqual(bias.constant_value, (1, -2))
        self.assertEqual(bias.shape, (2,))
        self.assertEqual(add.input_ids, (matmul.id, bias.id))

    def test_trace_preserves_the_layer_operators_canonical_masks(self) -> None:
        matmul_operator, add_operator = self.model.linear.operators
        self.assertEqual(self.graph.node("matmul.0").mask, matmul_operator.mask)
        self.assertEqual(self.graph.node("add.0").mask, add_operator.mask)

    def test_reference_graph_execution_matches_eager_execution(self) -> None:
        eager = self.model(X)
        traced = execute_graph(self.graph, (X,))
        self.assertEqual(traced, eager)
        self.assertEqual(traced, EXPECTED)

    def test_tensor_parameters_do_not_silently_violate_the_web_scalar_schema(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "scalar constants only"):
            self.graph.to_web_graph()

    def test_unbound_declaration_remains_available_but_is_not_executable(self) -> None:
        declaration = Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, "explicit weight"):
            declaration(X)


if __name__ == "__main__":
    unittest.main()
