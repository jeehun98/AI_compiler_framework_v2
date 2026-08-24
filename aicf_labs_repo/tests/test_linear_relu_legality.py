"""Legality checks remain separate from Linear-ReLU pattern detection."""

import unittest

from aicf_labs import (
    ExecutionGraph,
    Model,
    Sequential,
    check_linear_relu_legality,
    find_linear_relu_candidates,
    trace_model,
)
from aicf_labs.layers import Linear, ReLU
from aicf_labs.ops import add, matmul, mul, relu


X = [[1, 2], [3, 4]]
W1 = [[1, -1, 2], [0, 3, -1]]
B1 = [1, 0, -2]
W2 = [[2], [-1], [3]]
B2 = [4]


def _candidate(graph: ExecutionGraph):
    search = find_linear_relu_candidates(graph)
    if len(search.candidates) != 1:
        raise AssertionError(f"expected one structural candidate, got {len(search.candidates)}")
    return search, search.candidates[0]


class DynamicBiasModel(Model):
    def forward(self, value: object, weight: object, bias: object) -> object:
        return relu(add(matmul(value, weight), bias))


class MatMulExtraConsumerModel(Model):
    def forward(self, value: object) -> tuple[object, object]:
        product = matmul(value, W1)
        biased = add(product, B1)
        return relu(biased), mul(product, 1)


class AddExtraConsumerModel(Model):
    def forward(self, value: object) -> tuple[object, object]:
        product = matmul(value, W1)
        biased = add(product, B1)
        return relu(biased), mul(biased, 1)


class LinearReluLegalityTests(unittest.TestCase):
    def test_existing_mlp_candidate_is_legal_and_metrics_remain_separate(self) -> None:
        graph = trace_model(
            Sequential(
                Linear(2, 3, weight=W1, bias_value=B1),
                ReLU(),
                Linear(3, 1, weight=W2, bias_value=B2),
            ),
            X,
        )
        search, candidate = _candidate(graph)
        result = check_linear_relu_legality(graph, candidate)
        results = (result,)

        self.assertEqual(search.nodes_scanned, 10)
        self.assertEqual(search.mask_accepted, 1)
        self.assertEqual(search.structural_matches, 1)
        self.assertEqual(sum(item.legal for item in results), 1)
        self.assertEqual(sum(not item.legal for item in results), 0)
        self.assertTrue(result.legal)
        self.assertEqual(result.reasons, ())

    def test_dynamic_add_input_is_structural_but_not_a_bias_add(self) -> None:
        graph = trace_model(DynamicBiasModel(), X, W1, B1)
        search, candidate = _candidate(graph)
        result = check_linear_relu_legality(graph, candidate)

        self.assertEqual(search.structural_matches, 1)
        self.assertFalse(result.legal)
        self.assertEqual(result.reasons, ("not_bias_add",))

    def test_matmul_extra_consumer_is_rejected(self) -> None:
        graph = trace_model(MatMulExtraConsumerModel(), X)
        search, candidate = _candidate(graph)
        result = check_linear_relu_legality(graph, candidate)

        self.assertEqual(search.structural_matches, 1)
        self.assertFalse(result.legal)
        self.assertEqual(result.reasons, ("matmul_has_extra_consumer",))

    def test_add_extra_consumer_is_rejected(self) -> None:
        graph = trace_model(AddExtraConsumerModel(), X)
        search, candidate = _candidate(graph)
        result = check_linear_relu_legality(graph, candidate)

        self.assertEqual(search.structural_matches, 1)
        self.assertFalse(result.legal)
        self.assertEqual(result.reasons, ("add_has_extra_consumer",))

    def test_legality_check_does_not_mutate_the_graph(self) -> None:
        graph = trace_model(
            Sequential(Linear(2, 3, weight=W1, bias_value=B1), ReLU()),
            X,
        )
        _, candidate = _candidate(graph)
        nodes_before = graph.nodes
        outputs_before = graph.outputs

        check_linear_relu_legality(graph, candidate)

        self.assertIs(graph.nodes, nodes_before)
        self.assertIs(graph.outputs, outputs_before)


if __name__ == "__main__":
    unittest.main()
