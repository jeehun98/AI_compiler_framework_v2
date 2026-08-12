from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess


@dataclass(frozen=True)
class ToolResult:
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


class ExternalTool:
    executable: str

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def run(self, command: tuple[str, ...]) -> ToolResult:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
        return ToolResult(command, completed.stdout, completed.stderr, completed.returncode)
