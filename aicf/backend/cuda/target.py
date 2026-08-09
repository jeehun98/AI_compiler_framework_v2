from dataclasses import dataclass


@dataclass(frozen=True)
class CUDATarget:
    arch: str = "sm_86"
