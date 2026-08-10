from .graph import Graph


def verify_graph(graph: Graph) -> None:
    """Verify basic graph identity, topology and use-def integrity."""

    roots = [
        *graph.inputs,
        *graph.parameters,
    ]

    ids = {
        value.id
        for value in roots
    }

    if len(ids) != len(roots):
        raise ValueError(
            "duplicate graph input/parameter value id"
        )

    for value in roots:
        if value.producer is not None:
            raise ValueError(
                f"graph input/parameter %{value.id} "
                "unexpectedly has a producer"
            )

    # Nodes are expected to be stored
    # in topological construction order.
    for node in graph.nodes:
        for value in node.inputs:
            if value.id not in ids:
                raise ValueError(
                    f"node {node.id} uses undefined "
                    f"value %{value.id}"
                )

            expected_uses = sum(
                candidate is value
                for candidate in node.inputs
            )

            recorded_uses = sum(
                user is node
                for user in value.users
            )

            if recorded_uses != expected_uses:
                raise ValueError(
                    f"use-list mismatch for %{value.id} "
                    f"at node {node.id}: "
                    f"expected {expected_uses}, "
                    f"recorded {recorded_uses}"
                )

        for value in node.outputs:
            if value.id in ids:
                raise ValueError(
                    f"duplicate value id: {value.id}"
                )

            if value.producer is not node:
                raise ValueError(
                    f"producer mismatch for %{value.id}: "
                    f"expected node {node.id}"
                )

            ids.add(value.id)

    for value in graph.outputs:
        if value.id not in ids:
            raise ValueError(
                f"graph output %{value.id} is undefined"
            )