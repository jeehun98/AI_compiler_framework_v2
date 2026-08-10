def _format_type(value) -> str:
    dims = "x".join(str(dim) for dim in value.spec.shape)
    return f"tensor<{dims}x{value.spec.dtype}>"


def format_graph(graph) -> str:
    lines = []

    for value in graph.inputs:
        label = f" @{value.name}" if value.name else ""
        lines.append(f"input %{value.id}{label} : {_format_type(value)}")

    for value in graph.parameters:
        label = f" @{value.name}" if value.name else ""
        lines.append(f"param %{value.id}{label} : {_format_type(value)}")

    if lines and graph.nodes:
        lines.append("")

    for node in graph.nodes:
        ins = ", ".join(f"%{v.id}" for v in node.inputs)
        outs = ", ".join(f"%{v.id}" for v in node.outputs)
        lines.append(f"{outs} = {node.op}({ins})")

    return "\n".join(lines)
