def format_graph(graph) -> str:
    lines = []
    for node in graph.nodes:
        ins = ", ".join(f"%{v.id}" for v in node.inputs)
        outs = ", ".join(f"%{v.id}" for v in node.outputs)
        lines.append(f"{outs} = {node.op}({ins})")
    return "\n".join(lines)
