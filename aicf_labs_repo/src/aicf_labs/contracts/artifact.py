from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSet:
    cuda_source: str | None = None
    ptx: str | None = None
    cubin_path: str | None = None
    sass: str | None = None
