from __future__ import annotations

from ...graph import Graph
from ..marking import OperatorMarkRegistry, default_operator_registry
from .propagation import MaskPropagationResult, propagate_common_mask


def analyze_graph_marks(
    graph: Graph,
    registry: OperatorMarkRegistry | None = None,
) -> MaskPropagationResult:
    """Run the existing cheap mask propagation over a native canonical graph."""
    registry = registry or default_operator_registry()
    return propagate_common_mask(graph.operator_names(), registry)
