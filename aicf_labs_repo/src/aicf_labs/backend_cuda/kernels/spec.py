from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LaunchSpec:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


@dataclass(frozen=True)
class KernelSpec:
    """A fixed CUDA implementation chosen for observation."""

    name: str
    workload_kind: str
    entry: str
    source_path: Path
    launch: LaunchSpec
    architecture: str = "sm_86"
    compile_options: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    def source(self) -> str:
        return self.source_path.read_text(encoding="utf-8")
