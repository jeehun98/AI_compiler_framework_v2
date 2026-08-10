def _node_name(node) -> str:
    return f"n{node.id}:{node.op}"


def format_use_def(graph) -> str:
    """Render the graph's producer/use relationships for inspection."""

    values = [
        *graph.inputs,
        *graph.parameters,
    ]

    for node in graph.nodes:
        values.extend(node.outputs)

    lines = []

    for value in values:
        producer = (
            "<root>"
            if value.producer is None
            else _node_name(value.producer)
        )

        users = (
            ", ".join(
                _node_name(node)
                for node in value.users
            )
            or "<none>"
        )

        label = (
            f" @{value.name}"
            if value.name
            else ""
        )

        lines.append(
            f"%{value.id}{label}: "
            f"producer={producer}, "
            f"users=[{users}]"
        )

    return "\n".join(lines)