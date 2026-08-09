from .graph import Graph


def verify_graph(graph: Graph) -> None:
    """Minimal structural verification hook.

    TODO: dominance, use-def integrity, alias/mutation constraints, shape checks.
    """
    ids = {v.id for v in graph.inputs}
    for node in graph.nodes:
        for value in node.outputs:
            if value.id in ids:
                raise ValueError(f"duplicate value id: {value.id}")
            ids.add(value.id)
