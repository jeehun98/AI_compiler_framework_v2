"""ReLU operator semantics and inspected repository implementation."""

from dataclasses import dataclass

from ..implementation import Implementation, Observation, SassEvidence
from ..masks import OperatorMask
from ..operator import Operator


_RELU_FP32_SCALAR = Implementation(
    name="fp32_scalar",
    source_file="operators/relu/relu.cu",
    kernel_name="relu_fp32",
    input_dtype="fp32",
    output_dtype="fp32",
    sass_evidence=(
        SassEvidence(
            file="operators/relu/artifacts/relu.sass",
            observation=(
                "The inspected relu_fp32 stream lowers max(x, 0) to FP32 "
                "min/max against RZ at offset 0x00b0."
            ),
            instruction="FMNMX R7, RZ, R2, !PT",
            status=Observation.OBSERVED,
        ),
        SassEvidence(
            file="operators/relu/artifacts/relu.sass",
            observation="No Tensor Core instruction occurs in the inspected stream.",
            instruction="HMMA",
            status=Observation.NOT_OBSERVED,
        ),
        SassEvidence(
            file="operators/relu/artifacts/relu.sass",
            observation="No SFU instruction occurs in the inspected stream.",
            instruction="MUFU",
            status=Observation.NOT_OBSERVED,
        ),
        SassEvidence(
            file="operators/relu/artifacts/relu.sass",
            observation="No shared-memory load or store occurs in the inspected stream.",
            instruction="LDS/STS",
            status=Observation.NOT_OBSERVED,
        ),
        SassEvidence(
            file="operators/relu/artifacts/relu.sass",
            observation="No block barrier occurs in the inspected stream.",
            instruction="BAR",
            status=Observation.NOT_OBSERVED,
        ),
        SassEvidence(
            file="operators/relu/artifacts/relu.sass",
            observation="No atomic instruction occurs in the inspected stream.",
            instruction="ATOM",
            status=Observation.NOT_OBSERVED,
        ),
    ),
)


@dataclass(frozen=True, init=False)
class ReluOperator(Operator):
    """Declarative ReLU with the verified FP32 scalar implementation attached."""

    def __init__(self) -> None:
        super().__init__(
            name="relu",
            expression="y = max(x, 0)",
            category="elementwise",
            arity=1,
            mask=(
                OperatorMask.ELEMENTWISE
                | OperatorMask.PURE
                | OperatorMask.SHAPE_PRESERVING
            ),
            implementations=(_RELU_FP32_SCALAR,),
        )
