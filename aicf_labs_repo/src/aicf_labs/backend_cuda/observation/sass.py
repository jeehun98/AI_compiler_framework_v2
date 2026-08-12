from __future__ import annotations

from collections import Counter
import re

from .model import LayerObservation


_SASS_OPCODE = re.compile(r"(?:/\*[0-9A-Fa-f]+\*/\s*)?([A-Z][A-Z0-9_.]+)\b")


def observe_sass(sass: str) -> LayerObservation:
    counts: Counter[str] = Counter()
    for raw in sass.splitlines():
        line = raw.strip()
        if not line or line.startswith(("code for", "Function", ".headerflags")):
            continue
        match = _SASS_OPCODE.search(line)
        if match:
            counts[match.group(1)] += 1

    return LayerObservation(
        layer="sass",
        metrics={
            "instruction_count": sum(counts.values()),
            "opcode_counts": dict(sorted(counts.items())),
            "global_loads": sum(v for k, v in counts.items() if k.startswith(("LDG", "LDGSTS"))),
            "global_stores": sum(v for k, v in counts.items() if k.startswith("STG")),
            "ffma": sum(v for k, v in counts.items() if k.startswith("FFMA")),
        },
    )
