"""Regression tests for masks as cheap positive feature indexes."""

import unittest

from aicf_labs import Operator, OperatorMask, Sequential, summarize_masks
from aicf_labs.layers import Linear, ReLU


class MaskMatchingTests(unittest.TestCase):
    def test_required_features_filter_model_operators(self) -> None:
        model = Sequential(Linear(8, 4, bias=True), ReLU())
        required = (
            OperatorMask.ELEMENTWISE
            | OperatorMask.PURE
            | OperatorMask.SHAPE_PRESERVING
        )

        candidates = [
            operator.name
            for operator in model.operators()
            if operator.matches(required)
        ]

        self.assertEqual(candidates, ["relu"])

    def test_different_operators_share_common_features(self) -> None:
        model = Sequential(Linear(8, 4, bias=True), ReLU())
        add, relu = model.operators()[1:]
        common = OperatorMask.ELEMENTWISE | OperatorMask.PURE

        self.assertTrue(add.matches(common))
        self.assertTrue(relu.matches(common))
        self.assertNotEqual(add.name, relu.name)

    def test_region_summary_is_only_and_or_aggregation(self) -> None:
        model = Sequential(Linear(8, 4, bias=True), ReLU())
        add, relu = model.operators()[1:]

        summary = summarize_masks((add.mask, relu.mask))

        self.assertEqual(
            summary.common,
            OperatorMask.ELEMENTWISE | OperatorMask.PURE,
        )
        self.assertTrue(summary.present & OperatorMask.BROADCAST)
        self.assertTrue(
            summary.matches(OperatorMask.ELEMENTWISE | OperatorMask.PURE)
        )
        self.assertFalse(summary.matches(OperatorMask.PURE, OperatorMask.BROADCAST))

    def test_vocabulary_contains_only_current_index_features(self) -> None:
        self.assertEqual(
            set(OperatorMask.__members__),
            {
                "NONE",
                "ELEMENTWISE",
                "REDUCTION",
                "COMMUTATIVE",
                "PURE",
                "SHAPE_PRESERVING",
                "PERMUTATION",
                "BROADCAST",
            },
        )

    def test_new_operator_reuses_matching_with_only_feature_metadata(self) -> None:
        sigmoid = Operator(
            name="sigmoid",
            expression="y = 1 / (1 + exp(-x))",
            category="elementwise",
            arity=1,
            mask=(
                OperatorMask.ELEMENTWISE
                | OperatorMask.PURE
                | OperatorMask.SHAPE_PRESERVING
            ),
        )
        required = OperatorMask.ELEMENTWISE | OperatorMask.PURE

        self.assertTrue(sigmoid.matches(required))


if __name__ == "__main__":
    unittest.main()
