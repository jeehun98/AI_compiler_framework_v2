from __future__ import annotations
from dataclasses import dataclass
from ...ir.module import IRModule


@dataclass
class CUDALoweredModule:
    """Placeholder target representation.

    TODO: loop/tile/thread mapping/shared-memory/intrinsic level IR.
    """
    source_ir: IRModule
    kernels: list[str]


def lower_to_cuda(module: IRModule, context) -> CUDALoweredModule:
    # One symbolic kernel per operation for now.
    kernels = [f"kernel_{i}_{op.name}" for i, op in enumerate(module.ops)]
    return CUDALoweredModule(module, kernels)
