from __future__ import annotations

from .mask import OpMask
from .operator_mark import OperatorMark
from .registry import OperatorMarkRegistry


BUILTIN_OPERATOR_MARKS: tuple[OperatorMark, ...] = (
    OperatorMark(
        "add",
        OpMask.COMMUTATIVE
        | OpMask.ASSOCIATIVE
        | OpMask.ELEMENTWISE
        | OpMask.SHAPE_PRESERVING
        | OpMask.SIDE_EFFECT_FREE
        | OpMask.ELEMENT_LOCAL
        | OpMask.BROADCASTABLE
        | OpMask.MATERIALIZATION_OPTIONAL
        | OpMask.FUSION_FRIENDLY,
        "Mathematical add; floating-point reassociation still requires a numerical contract.",
    ),
    OperatorMark(
        "mul",
        OpMask.COMMUTATIVE
        | OpMask.ASSOCIATIVE
        | OpMask.ELEMENTWISE
        | OpMask.SHAPE_PRESERVING
        | OpMask.SIDE_EFFECT_FREE
        | OpMask.ELEMENT_LOCAL
        | OpMask.BROADCASTABLE
        | OpMask.MATERIALIZATION_OPTIONAL
        | OpMask.FUSION_FRIENDLY,
    ),
    OperatorMark(
        "bias_add",
        OpMask.ELEMENTWISE
        | OpMask.SHAPE_PRESERVING
        | OpMask.SIDE_EFFECT_FREE
        | OpMask.ELEMENT_LOCAL
        | OpMask.BROADCASTABLE
        | OpMask.MATERIALIZATION_OPTIONAL
        | OpMask.FUSION_FRIENDLY,
    ),
    OperatorMark(
        "relu",
        OpMask.ELEMENTWISE
        | OpMask.SHAPE_PRESERVING
        | OpMask.SIDE_EFFECT_FREE
        | OpMask.ELEMENT_LOCAL
        | OpMask.MATERIALIZATION_OPTIONAL
        | OpMask.FUSION_FRIENDLY,
    ),
    OperatorMark(
        "gemm",
        OpMask.SIDE_EFFECT_FREE,
        "GEMM is not marked element-local because each output element reduces across K.",
    ),
    OperatorMark(
        "reduce_sum",
        OpMask.SIDE_EFFECT_FREE | OpMask.REDUCTION,
        "Reduction-specific fusion should be handled by a dedicated rule, not generic element-local screening.",
    ),
)


def default_operator_registry() -> OperatorMarkRegistry:
    registry = OperatorMarkRegistry()
    for mark in BUILTIN_OPERATOR_MARKS:
        registry.register(mark)
    return registry
