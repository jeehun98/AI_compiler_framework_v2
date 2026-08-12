from __future__ import annotations

from collections import Counter
import re

from .model import LayerObservation


def _ptx_opcode(line: str) -> str | None:
    text = line.split("//", 1)[0].strip()
    if not text or text.startswith((".", "{", "}")) or text.endswith(":"):
        return None
    if text.startswith("@"):
        parts = text.split(None, 1)
        if len(parts) < 2:
            return None
        text = parts[1]
    match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)\b", text)
    return match.group(1) if match else None


def observe_ptx(ptx: str) -> LayerObservation:
    counts: Counter[str] = Counter()
    for line in ptx.splitlines():
        opcode = _ptx_opcode(line)
        if opcode:
            counts[opcode] += 1
    return LayerObservation(
        layer="ptx",
        metrics={
            "instruction_count": sum(counts.values()),
            "opcode_counts": dict(sorted(counts.items())),
            "ld_global": sum(v for k, v in counts.items() if k.startswith("ld.global")),
            "st_global": sum(v for k, v in counts.items() if k.startswith("st.global")),
            "fma": sum(v for k, v in counts.items() if "fma" in k),
            "predicated_instructions": sum(1 for line in ptx.splitlines() if line.lstrip().startswith("@")),
        },
    )
