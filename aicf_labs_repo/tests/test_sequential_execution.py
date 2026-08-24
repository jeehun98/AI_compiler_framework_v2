"""Regression tests for Sequential execution through callable layers."""

import unittest

from aicf_labs import Sequential, execute_graph, trace_model
from aicf_labs.layers import Flatten, Linear, ReLU


X = [[1, 2], [3, 4]]
WEIGHT = [[2, 1], [-1, 3]]
BIAS = [-1, -10]
EXPECTED = [[0, 0], [1, 5]]


class SequentialExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Sequential(
            Linear(2, 2, weight=WEIGHT, bias_value=BIAS),
            ReLU(),
        )
        self.graph = trace_model(self.model, X, input_names=("x",))

    def test_eager_execution_calls_layers_in_order(self) -> None:
        self.assertEqual(self.model(X), EXPECTED)

    def test_trace_contains_only_primitive_operations(self) -> None:
        self.assertEqual(self.graph.operator_sequence, ("matmul", "add", "relu"))
        self.assertNotIn("linear", self.graph.operator_sequence)
        self.assertNotIn("sequential", self.graph.operator_sequence)
        self.assertEqual(self.graph.inputs, ("input.0",))
        self.assertEqual(self.graph.outputs, ("relu.0",))
        self.assertEqual(
            self.graph.node("matmul.0").input_ids,
            ("input.0", "constant.0"),
        )
        self.assertEqual(
            self.graph.node("add.0").input_ids,
            ("matmul.0", "constant.1"),
        )
        self.assertEqual(self.graph.node("relu.0").input_ids, ("add.0",))

    def test_constants_and_canonical_masks_are_preserved(self) -> None:
        linear, relu = self.model
        matmul_operator, add_operator = linear.operators
        relu_operator = relu.operators[0]

        self.assertEqual(
            self.graph.node("constant.0").constant_value,
            ((2, 1), (-1, 3)),
        )
        self.assertEqual(self.graph.node("constant.1").constant_value, (-1, -10))
        self.assertEqual(self.graph.node("matmul.0").mask, matmul_operator.mask)
        self.assertEqual(self.graph.node("add.0").mask, add_operator.mask)
        self.assertEqual(self.graph.node("relu.0").mask, relu_operator.mask)

    def test_reference_graph_execution_matches_eager_execution(self) -> None:
        self.assertEqual(execute_graph(self.graph, (X,)), self.model(X))
        self.assertEqual(execute_graph(self.graph, (X,)), EXPECTED)

    def test_unsupported_layer_fails_without_adding_flatten_execution(self) -> None:
        model = Sequential(
            Linear(2, 2, weight=WEIGHT, bias_value=BIAS),
            Flatten(),
        )
        with self.assertRaisesRegex(NotImplementedError, "Flatten execution"):
            model(X)


if __name__ == "__main__":
    unittest.main()
