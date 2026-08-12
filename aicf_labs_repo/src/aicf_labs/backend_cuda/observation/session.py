from __future__ import annotations

from dataclasses import dataclass

from ..kernels import KernelSpec
from .ast import observe_ast_dump
from .model import KernelObservation, LayerObservation
from .ptx import observe_ptx
from .sass import observe_sass
from .source import observe_cuda_source


@dataclass(frozen=True)
class ObservationInputs:
    ast_dump: str | None = None
    ptx: str | None = None
    sass: str | None = None
    runtime: LayerObservation | None = None


def observe_kernel(
    kernel: KernelSpec,
    inputs: ObservationInputs = ObservationInputs(),
) -> KernelObservation:
    return KernelObservation(
        kernel=kernel,
        source=observe_cuda_source(kernel.source()),
        ast=observe_ast_dump(inputs.ast_dump) if inputs.ast_dump else None,
        ptx=observe_ptx(inputs.ptx) if inputs.ptx else None,
        sass=observe_sass(inputs.sass) if inputs.sass else None,
        runtime=inputs.runtime,
    )
