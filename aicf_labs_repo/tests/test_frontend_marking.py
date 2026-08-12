from aicf_labs.frontend_lab import GENERIC_ELEMENTWISE_SCREEN, default_operator_registry, propagate_common_mask
from aicf_labs.frontend_lab.marking import OpMask


def test_mask_definition_and_operator_marking_are_separate():
    registry = default_operator_registry()
    add = registry.get("add")
    relu = registry.get("relu")

    assert add.has(OpMask.COMMUTATIVE)
    assert relu.has(OpMask.ELEMENT_LOCAL)
    assert not relu.has(OpMask.COMMUTATIVE)


def test_common_mask_propagation_preserves_shared_fusion_screen_bits():
    registry = default_operator_registry()
    result = propagate_common_mask(("bias_add", "relu"), registry)

    assert result.preserves(OpMask.SIDE_EFFECT_FREE | OpMask.ELEMENT_LOCAL)
    assert GENERIC_ELEMENTWISE_SCREEN.candidate_survives(result)


def test_property_disappears_when_chain_enters_reduction():
    registry = default_operator_registry()
    result = propagate_common_mask(("bias_add", "relu", "reduce_sum"), registry)

    assert not result.preserves(OpMask.ELEMENT_LOCAL)
    assert not GENERIC_ELEMENTWISE_SCREEN.candidate_survives(result)
    assert "ELEMENT_LOCAL" in result.steps[-1].removed_properties
