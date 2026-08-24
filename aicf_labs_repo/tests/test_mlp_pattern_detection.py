"""MLP composition and non-mutating Linear-ReLU pattern detection."""

import unittest

from aicf_labs import (
    Model,
    Sequential,
    execute_graph,
    find_linear_relu_candidates,
    trace_model,
)
from aicf_labs.layers import Linear, ReLU
from aicf_labs.ops import add, matmul, relu


X = [[1, 2], [3, 4]]
W1 = [[1, -1, 2], [0, 3, -1]]
B1 = [1, 0, -2]
W2 = [[2], [-1], [3]]
B2 = [4]
EXPECTED = [[3], [3]]


class MatMulReluModel(Model):
    def forward(self, value: object, weight: object) -> object:
        return relu(matmul(value, weight))


class AddReluModel(Model):
    def forward(self, value: object) -> object:
        return relu(add(value, 0))


class MlpPatternDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Sequential(
            Linear(2, 3, weight=W1, bias_value=B1),
            ReLU(),
            Linear(3, 1, weight=W2, bias_value=B2),
        )
        self.graph = trace_model(self.model, X, input_names=("x",))

    def test_mlp_eager_and_graph_execution_are_exact(self) -> None:
        self.assertEqual(self.model(X), EXPECTED)
        self.assertEqual(execute_graph(self.graph, (X,)), EXPECTED)

    def test_mlp_trace_connects_five_primitive_operations(self) -> None:
        self.assertEqual(
            self.graph.operator_sequence,
            ("matmul", "add", "relu", "matmul", "add"),
        )
        self.assertEqual(self.graph.inputs, ("input.0",))
        self.assertEqual(self.graph.outputs, ("add.1",))
        self.assertEqual(
            self.graph.node("matmul.0").input_ids,
            ("input.0", "constant.0"),
        )
        self.assertEqual(
            self.graph.node("add.0").input_ids,
            ("matmul.0", "constant.1"),
        )
        self.assertEqual(self.graph.node("relu.0").input_ids, ("add.0",))
        self.assertEqual(
            self.graph.node("matmul.1").input_ids,
            ("relu.0", "constant.2"),
        )
        self.assertEqual(
            self.graph.node("add.1").input_ids,
            ("matmul.1", "constant.3"),
        )

    def test_two_linear_layers_keep_distinct_parameter_identities_and_masks(self) -> None:
        self.assertEqual(
            tuple(node.id for node in self.graph.nodes if node.operator_id == "constant"),
            ("constant.0", "constant.1", "constant.2", "constant.3"),
        )
        self.assertEqual(
            self.graph.node("constant.0").constant_value,
            ((1, -1, 2), (0, 3, -1)),
        )
        self.assertEqual(self.graph.node("constant.1").constant_value, (1, 0, -2))
        self.assertEqual(
            self.graph.node("constant.2").constant_value,
            ((2,), (-1,), (3,)),
        )
        self.assertEqual(self.graph.node("constant.3").constant_value, (4,))
        self.assertNotEqual("constant.0", "constant.2")
        self.assertNotEqual("constant.1", "constant.3")
        first_linear, relu_layer, second_linear = self.model
        self.assertEqual(self.graph.node("matmul.0").mask, first_linear.operators[0].mask)
        self.assertEqual(self.graph.node("add.0").mask, first_linear.operators[1].mask)
        self.assertEqual(self.graph.node("relu.0").mask, relu_layer.operators[0].mask)
        self.assertEqual(self.graph.node("matmul.1").mask, second_linear.operators[0].mask)
        self.assertEqual(self.graph.node("add.1").mask, second_linear.operators[1].mask)

    def test_actual_mlp_trace_has_one_linear_relu_candidate_without_mutation(self) -> None:
        original = self.graph
        result = find_linear_relu_candidates(self.graph)

        self.assertEqual(result.nodes_scanned, 10)
        self.assertEqual(result.mask_accepted, 1)
        self.assertEqual(result.structural_matches, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].pattern_id, "linear-relu-fusion")
        self.assertEqual(
            result.candidates[0].node_ids,
            ("matmul.0", "add.0", "relu.0"),
        )
        self.assertEqual(result.candidates[0].root_id, "relu.0")
        self.assertIs(self.graph, original)

    def test_linear_without_relu_is_not_a_candidate(self) -> None:
        graph = trace_model(
            Sequential(Linear(2, 3, weight=W1, bias_value=B1)),
            X,
        )
        self.assertEqual(find_linear_relu_candidates(graph).candidates, ())

    def test_matmul_directly_followed_by_relu_is_not_a_candidate(self) -> None:
        graph = trace_model(MatMulReluModel(), X, W1)
        self.assertEqual(find_linear_relu_candidates(graph).candidates, ())

    def test_add_not_produced_by_matmul_is_not_a_candidate(self) -> None:
        graph = trace_model(AddReluModel(), X)
        self.assertEqual(find_linear_relu_candidates(graph).candidates, ())


if __name__ == "__main__":
    unittest.main()
