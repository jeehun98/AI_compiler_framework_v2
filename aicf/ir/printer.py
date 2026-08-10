from .module import IRModule


def _format_type(value) -> str:
    dims = "x".join(str(dim) for dim in value.type.shape)
    return f"tensor<{dims}x{value.type.dtype}>"


def format_ir(module: IRModule) -> str:
    lines = []

    for value in module.inputs:
        lines.append(f"input {value.name} : {_format_type(value)}")
    for value in module.parameters:
        lines.append(f"param {value.name} : {_format_type(value)}")

    if lines and module.ops:
        lines.append("")

    for op in module.ops:
        lhs = ", ".join(v.name for v in op.results)
        rhs = ", ".join(v.name for v in op.operands)
        attrs = f" {op.attrs}" if op.attrs else ""
        lines.append(f"{lhs} = {op.name}({rhs}){attrs}")

    if module.outputs:
        lines.append("return " + ", ".join(v.name for v in module.outputs))
    return "\n".join(lines)
