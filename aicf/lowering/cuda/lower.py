from __future__ import annotations

from ...ir.module import IRModule
from .ir import CUDALoweredModule
from .rules import CUDA_LOWERINGS


def lower_to_cuda(module: IRModule, context) -> CUDALoweredModule:
    """Lower optimized IR operations using registered CUDA lowering rules.

    Optimization decisions are assumed to have already happened. This layer
    only translates each selected IR operation into target-side kernel plans.
    """

    kernels = []

    for index, op in enumerate(module.ops):
        rule = CUDA_LOWERINGS.require(op.name)
        produced = rule(
            op,
            index=index,
            context=context,
        )
        kernels.extend(produced)

    return CUDALoweredModule(
        source_ir=module,
        kernels=kernels,
    )