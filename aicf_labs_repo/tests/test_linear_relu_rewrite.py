"""Minimal logical rewrite for one legal Linear-ReLU candidate."""

import unittest

from aicf_labs import (
    ExecutionGraph,
    Model,
    Sequential,
    check_linear_relu_legality,
    compare_graph_executions,
    find_linear_relu_candidates,
    rewrite_linear_relu,
    trace_model,
)
from aicf_labs.layers import Linear, ReLU
from aicf_labs.operators import LinearReluOperator
from aicf_labs.ops import add, matmul, relu


X = [[1, 2], [3, 4]]
W1 = [[1, -1, 2], [0, 3, -1]]
B1 = [1, 0, -2]
W2 = [[2], [-1], [3]]
B2 = [4]


class DynamicBiasModel(Model):
    def forward(self, value: object, weight: object, bias: object) -> object:
        return relu(add(matmul(value, weight), bias))


def _mlp_graph() -> ExecutionGraph:
    return trace_model(
        Sequential(
            Linear(2, 3, weight=W1, bias_value=B1),
            ReLU(),
            Linear(3, 1, weight=W2, bias_value=B2),
        ),
        X,
    )


class LinearReluRewriteTests(unittest.TestCase):
    def test_rewrites_only_the_legal_candidate_to_one_logical_node(self) -> None:
        graph = _mlp_graph()
        search = find_linear_relu_candidates(graph)
        legality = check_linear_relu_legality(graph, search.candidates[0])

        rewritten = rewrite_linear_relu(graph, legality)

        self.assertEqual(
            rewritten.operator_sequence,
            ("linearRelu", "matmul", "add"),
        )
        self.assertEqual(len(graph.nodes), 10)
        self.assertEqual(len(rewritten.nodes), 8)
        self.assertNotIn("matmul.0", tuple(node.id for node in rewritten.nodes))
        self.assertNotIn("add.0", tuple(node.id for node in rewritten.nodes))
        self.assertNotIn("relu.0", tuple(node.id for node in rewritten.nodes))

        fused = rewritten.node("linearRelu.0")
        self.assertEqual(
            fused.input_ids,
            ("input.0", "constant.0", "constant.1"),
        )
        self.assertEqual(fused.mask, LinearReluOperator().mask)
        self.assertEqual(
            rewritten.node("matmul.1").input_ids,
            (fused.id, "constant.2"),
        )
        self.assertEqual(rewritten.outputs, ("add.1",))
        self.assertEqual(LinearReluOperator().implementations, ())

    def test_logical_rewrite_preserves_reference_execution_exactly(self) -> None:
        graph = _mlp_graph()
        candidate = find_linear_relu_candidates(graph).candidates[0]
        legality = check_linear_relu_legality(graph, candidate)
        rewritten = rewrite_linear_relu(graph, legality)

        comparison = compare_graph_executions(
            graph,
            rewritten,
            (X,),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        self.assertTrue(comparison.equivalent)
        self.assertEqual(comparison.original_output, [[3], [3]])
        self.assertEqual(comparison.rewritten_output, [[3], [3]])

    def test_rewrite_updates_graph_output_when_relu_is_the_output(self) -> None:
        graph = trace_model(
            Sequential(Linear(2, 3, weight=W1, bias_value=B1), ReLU()),
            X,
        )
        candidate = find_linear_relu_candidates(graph).candidates[0]
        legality = check_linear_relu_legality(graph, candidate)

        rewritten = rewrite_linear_relu(graph, legality)

        self.assertEqual(rewritten.outputs, ("linearRelu.0",))
        self.assertEqual(rewritten.operator_sequence, ("linearRelu",))
        self.assertTrue(
            compare_graph_executions(
                graph,
                rewritten,
                (X,),
                rel_tol=0.0,
                abs_tol=0.0,
            ).equivalent
        )

    def test_rewrite_does_not_mutate_the_original_graph(self) -> None:
        graph = _mlp_graph()
        nodes_before = graph.nodes
        outputs_before = graph.outputs
        candidate = find_linear_relu_candidates(graph).candidates[0]
        legality = check_linear_relu_legality(graph, candidate)

        rewritten = rewrite_linear_relu(graph, legality)

        self.assertIsNot(rewritten, graph)
        self.assertIs(graph.nodes, nodes_before)
        self.assertIs(graph.outputs, outputs_before)
        self.assertEqual(
            graph.operator_sequence,
            ("matmul", "add", "relu", "matmul", "add"),
        )

    def test_illegal_candidate_cannot_be_rewritten(self) -> None:
        graph = trace_model(DynamicBiasModel(), X, W1, B1)
        candidate = find_linear_relu_candidates(graph).candidates[0]
        legality = check_linear_relu_legality(graph, candidate)

        self.assertFalse(legality.legal)
        with self.assertRaisesRegex(ValueError, "cannot rewrite an illegal candidate"):
            rewrite_linear_relu(graph, legality)


if __name__ == "__main__":
    unittest.main()
