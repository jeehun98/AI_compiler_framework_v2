from __future__ import annotations

from pathlib import Path

from .command import ExternalTool


class ClangAstTool(ExternalTool):
    executable = "clang++"

    def ast_command(self, source: Path, cuda_arch: str) -> tuple[str, ...]:
        return (
            self.executable,
            "-x",
            "cuda",
            f"--cuda-gpu-arch={cuda_arch}",
            "-Xclang",
            "-ast-dump",
            "-fsyntax-only",
            str(source),
        )
