from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CUDAExecutableImage:
    kernels: list[str]
    code: str


def codegen(lowered) -> CUDAExecutableImage:
    # TODO: emit CUDA C++/PTX or invoke NVRTC/NVCC.
    body = "\n".join(f"// TODO codegen for {name}" for name in lowered.kernels)
    return CUDAExecutableImage(lowered.kernels, body)
