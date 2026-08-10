from __future__ import annotations

from ..compiler.analysis.use_def import UseDefAnalysis
from ..ir.module import IRModule
from ..ir.operation import Operation


def _op_name(analysis: UseDefAnalysis, op: Operation) -> str:
    return f"op{analysis.op_index(op)}:{op.name}"


def format_ir_use_def(module: IRModule, analysis: UseDefAnalysis) -> str:
    """Render IR producer/use information for inspection."""

    values = [*module.inputs, *module.parameters]
    for op in module.ops:
        values.extend(op.results)

    lines: list[str] = []

    for value in values:
        producer = analysis.producer(value)
        producer_text = (
            "<root>" if producer is None else _op_name(analysis, producer)
        )

        uses = analysis.uses(value)
        uses_text = (
            ", ".join(
                f"{_op_name(analysis, use.user)}[operand={use.operand_index}]"
                for use in uses
            )
            or "<none>"
        )

        lines.append(
            f"{value.name}: producer={producer_text}, uses=[{uses_text}]"
        )

    return "\n".join(lines)