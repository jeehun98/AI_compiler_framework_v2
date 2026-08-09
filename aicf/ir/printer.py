from .module import IRModule


def format_ir(module: IRModule) -> str:
    lines = []
    for op in module.ops:
        lhs = ", ".join(v.name for v in op.results)
        rhs = ", ".join(v.name for v in op.operands)
        attrs = f" {op.attrs}" if op.attrs else ""
        lines.append(f"{lhs} = {op.name}({rhs}){attrs}")
    if module.outputs:
        lines.append("return " + ", ".join(v.name for v in module.outputs))
    return "\n".join(lines)
