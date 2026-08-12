from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentSpec:
    gpu_name: str | None = None
    gpu_arch: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    compiler: str | None = None
    compiler_version: str | None = None
