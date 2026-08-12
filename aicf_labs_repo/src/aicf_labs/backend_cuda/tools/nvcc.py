from __future__ import annotations

from pathlib import Path

from .command import ExternalTool


class NvccTool(ExternalTool):
    executable = "nvcc"

    def ptx_command(self, source: Path, output: Path, arch: str) -> tuple[str, ...]:
        return (self.executable, "-ptx", f"-arch={arch}", str(source), "-o", str(output))

    def cubin_command(self, source: Path, output: Path, arch: str) -> tuple[str, ...]:
        return (self.executable, "-cubin", f"-arch={arch}", str(source), "-o", str(output))
