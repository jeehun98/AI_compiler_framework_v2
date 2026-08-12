from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag


class MaskDomain(str, Enum):
    ALGEBRA = "algebra"
    TENSOR = "tensor"
    EFFECT = "effect"
    TRANSFORM = "transform"


class OpMask(IntFlag):
    """Binary vocabulary used for cheap operator-property screening.

    A set bit means that the operator explicitly advertises the property.
    A cleared bit means "not advertised / not satisfied" at this screening
    layer. Detailed legality analysis remains a separate step.
    """

    NONE = 0

    COMMUTATIVE = 1 << 0
    ASSOCIATIVE = 1 << 1
    ELEMENTWISE = 1 << 2
    SHAPE_PRESERVING = 1 << 3
    SIDE_EFFECT_FREE = 1 << 4
    ELEMENT_LOCAL = 1 << 5
    BROADCASTABLE = 1 << 6
    MATERIALIZATION_OPTIONAL = 1 << 7
    FUSION_FRIENDLY = 1 << 8
    REDUCTION = 1 << 9


@dataclass(frozen=True)
class MaskDefinition:
    bit: OpMask
    domain: MaskDomain
    description: str


MASK_DEFINITIONS: tuple[MaskDefinition, ...] = (
    MaskDefinition(OpMask.COMMUTATIVE, MaskDomain.ALGEBRA, "operand order may be exchanged in the mathematical operator"),
    MaskDefinition(OpMask.ASSOCIATIVE, MaskDomain.ALGEBRA, "grouping may be changed in the mathematical operator"),
    MaskDefinition(OpMask.ELEMENTWISE, MaskDomain.TENSOR, "output elements are computed element-by-element"),
    MaskDefinition(OpMask.SHAPE_PRESERVING, MaskDomain.TENSOR, "output shape is preserved from the primary tensor input"),
    MaskDefinition(OpMask.SIDE_EFFECT_FREE, MaskDomain.EFFECT, "operator has no externally visible side effect"),
    MaskDefinition(OpMask.ELEMENT_LOCAL, MaskDomain.TENSOR, "one output element does not require neighboring tensor elements"),
    MaskDefinition(OpMask.BROADCASTABLE, MaskDomain.TENSOR, "operator admits broadcast-style tensor operands"),
    MaskDefinition(OpMask.MATERIALIZATION_OPTIONAL, MaskDomain.TRANSFORM, "intermediate result may be consumed without mandatory tensor materialization"),
    MaskDefinition(OpMask.FUSION_FRIENDLY, MaskDomain.TRANSFORM, "operator advertises suitability for generic fusion screening"),
    MaskDefinition(OpMask.REDUCTION, MaskDomain.TENSOR, "operator combines values across a reduction domain"),
)


def definition_for(bit: OpMask) -> MaskDefinition:
    for definition in MASK_DEFINITIONS:
        if definition.bit == bit:
            return definition
    raise KeyError(f"unknown mask bit: {bit!r}")


def mask_names(mask: OpMask) -> tuple[str, ...]:
    return tuple(definition.bit.name for definition in MASK_DEFINITIONS if mask & definition.bit)
