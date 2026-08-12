from __future__ import annotations

from pathlib import Path

from .command import ExternalTool


class CuobjdumpTool(ExternalTool):
    executable = "cuobjdump"

    def sass_command(self, binary: Path) -> tuple[str, ...]:
        return (self.executable, "--dump-sass", str(binary))

    def resource_command(self, binary: Path) -> tuple[str, ...]:
        return (self.executable, "--dump-resource-usage", str(binary))
