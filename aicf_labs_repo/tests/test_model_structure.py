"""Tests for the declarative model/layer/operator representation."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from aicf_labs import (
    Monotonicity,
    Observation,
    Sequential,
    State,
)
from aicf_labs.layers import Flatten, Linear, ReLU
from aicf_labs.operators import ReluOperator


class ModelStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Sequential(
            Linear(128, 64, bias=True),
            ReLU(),
            Flatten(),
        )

    def test_sequential_stores_layers_in_order(self) -> None:
        self.assertEqual(len(self.model), 3)
        self.assertIsInstance(self.model[0], Linear)
        self.assertIsInstance(self.model[1], ReLU)
        self.assertIsInstance(self.model[2], Flatten)
        self.assertEqual(tuple(self.model), self.model[:])

    def test_layers_expand_to_expected_operators(self) -> None:
        self.assertEqual(
            [operator.name for operator in self.model[0].operators],
            ["matmul", "add"],
        )
        self.assertEqual(
            [operator.name for operator in self.model[1].operators],
            ["relu"],
        )
        self.assertEqual(
            [operator.name for operator in self.model[2].operators],
            ["reshape"],
        )
        self.assertEqual(
            [operator.name for operator in self.model.operators()],
            ["matmul", "add", "relu", "reshape"],
        )

    def test_linear_without_bias_has_only_matmul(self) -> None:
        layer = Linear(128, 64, bias=False)
        self.assertEqual([operator.name for operator in layer.operators], ["matmul"])

    def test_relu_mask(self) -> None:
        relu = self.model[1].operators[0]
        self.assertIs(relu.mask.elementwise, State.YES)
        self.assertIs(relu.mask.shape_preserving, State.YES)
        self.assertIs(relu.mask.idempotent, State.YES)
        self.assertIs(relu.mask.zero_preserving, State.YES)
        self.assertIs(relu.mask.invertible, State.NO)
        self.assertIs(
            relu.mask.monotonicity,
            Monotonicity.NONDECREASING,
        )
        self.assertIs(relu.mask.epilogue_fusible, State.YES)

    def test_only_relu_has_a_connected_implementation(self) -> None:
        matmul, add, relu, reshape = self.model.operators()
        self.assertEqual(matmul.implementations, ())
        self.assertEqual(add.implementations, ())
        self.assertEqual(reshape.implementations, ())
        self.assertEqual(
            [implementation.name for implementation in relu.implementations],
            ["fp32_scalar"],
        )

    def test_relu_implementation_matches_repository_evidence(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        implementation = self.model[1].operators[0].implementations[0]
        source_path = repository_root / implementation.source_file
        self.assertTrue(source_path.is_file())
        self.assertIn(
            "__global__ void relu_fp32",
            source_path.read_text(encoding="utf-8"),
        )

        observed = next(
            evidence
            for evidence in implementation.sass_evidence
            if evidence.status is Observation.OBSERVED
        )
        sass_text = (repository_root / observed.file).read_text(encoding="utf-8-sig")
        self.assertIn(observed.instruction, sass_text)

        hmma = next(
            evidence
            for evidence in implementation.sass_evidence
            if evidence.instruction == "HMMA"
        )
        self.assertIs(hmma.status, Observation.NOT_OBSERVED)
        self.assertNotIn("HMMA", sass_text)

    def test_summary_shows_layers_operators_and_implementation(self) -> None:
        summary = self.model.summary()
        self.assertIn("[0] Linear(in_features=128, out_features=64, bias=True)", summary)
        self.assertIn("matmul", summary)
        self.assertIn("add", summary)
        self.assertIn("relu", summary)
        self.assertIn("[fp32_scalar]", summary)
        self.assertIn("Flatten(start_dim=1)", summary)
        self.assertEqual(str(self.model), summary)

    def test_sequential_rejects_operators(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "Sequential accepts only Layer instances",
        ):
            Sequential(ReluOperator())  # type: ignore[arg-type]

    def test_layer_parameters_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_features must be positive"):
            Linear(0, 64)
        with self.assertRaisesRegex(TypeError, "out_features must be an integer"):
            Linear(128, True)
        with self.assertRaisesRegex(TypeError, "bias must be a bool"):
            Linear(128, 64, bias=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "start_dim must be an integer"):
            Flatten(start_dim=True)

    def test_semantic_objects_are_immutable(self) -> None:
        relu = self.model[1].operators[0]
        with self.assertRaises(FrozenInstanceError):
            self.model._layers = ()  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            relu.name = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            relu.mask.elementwise = State.NO  # type: ignore[misc]

    def test_representation_has_no_execution_api(self) -> None:
        self.assertFalse(hasattr(self.model, "forward"))
        self.assertFalse(hasattr(self.model, "compile"))
        self.assertFalse(hasattr(self.model, "optimize"))


if __name__ == "__main__":
    unittest.main()
