from __future__ import annotations

from .module import IRModule


def verify_ir(module: IRModule) -> None:
    """Verify basic SSA-like identity and topological invariants.

    The current IR is intentionally small, so verification focuses on the
    properties needed by existing analyses and rewrites:
    - roots are unique and have unique names
    - operands must already be defined
    - each result is defined exactly once
    - value names remain unique
    - outputs refer to defined values
    """

    roots = [*module.inputs, *module.parameters]

    defined = set()
    names: set[str] = set()

    for value in roots:
        if value in defined:
            raise ValueError(
                f"IR root value {value.name} is registered more than once"
            )
        if value.name in names:
            raise ValueError(f"duplicate IR value name: {value.name}")
        defined.add(value)
        names.add(value.name)

    seen_ops = set()

    for op_index, op in enumerate(module.ops):
        if op in seen_ops:
            raise ValueError(
                f"operation object appears more than once at index {op_index}: {op.name}"
            )
        seen_ops.add(op)

        for operand_index, operand in enumerate(op.operands):
            if operand not in defined:
                raise ValueError(
                    f"operation {op.name} operand {operand_index} uses "
                    f"undefined IR value {operand.name}"
                )

        for result in op.results:
            if result in defined:
                raise ValueError(f"IR value {result.name} is defined more than once")
            if result.name in names:
                raise ValueError(f"duplicate IR value name: {result.name}")
            defined.add(result)
            names.add(result.name)

    for output in module.outputs:
        if output not in defined:
            raise ValueError(f"IR output {output.name} is undefined")