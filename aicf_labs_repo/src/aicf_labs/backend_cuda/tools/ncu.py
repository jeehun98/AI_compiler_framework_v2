from __future__ import annotations

from .command import ExternalTool


class NsightComputeTool(ExternalTool):
    executable = "ncu"

    def profile_command(self, target: tuple[str, ...]) -> tuple[str, ...]:
        return (self.executable, "--set", "full", *target)
