"""ReLU operator semantics and inspected repository implementation."""

from dataclasses import dataclass

from ..implementation import Implementation, SassEvidence
from ..masks import (
    HardwareMask,
    Monotonicity,
    Observation,
    OperatorMask,
    State,
)
from ..operator import Operator


_RELU_FP32_SCALAR = Implementation(
    name="fp32_scalar",
    source_file="operators/relu/relu.cu",
    kernel_name="relu_fp32",
    input_dtype="fp32",
    output_dtype="fp32",
    hardware=HardwareMask(
        uses_cuda_core=State.YES,
        uses_tensor_core=State.NO,
        uses_sfu=State.NO,
        uses_shared_memory=State.NO,
        uses_barrier=State.NO,
        uses_atomic=State.NO,
    ),
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
            mask=OperatorMask(
                elementwise=State.YES,
                reduction=State.NO,
                shape_preserving=State.YES,
                rank_preserving=State.YES,
                element_independent=State.YES,
                linear=State.NO,
                idempotent=State.YES,
                zero_preserving=State.YES,
                invertible=State.NO,
                monotonicity=Monotonicity.NONDECREASING,
                producer_fusible=State.YES,
                consumer_fusible=State.YES,
                epilogue_fusible=State.YES,
                requires_materialization=State.NO,
                requires_global_sync=State.NO,
            ),
            implementations=(_RELU_FP32_SCALAR,),
        )
