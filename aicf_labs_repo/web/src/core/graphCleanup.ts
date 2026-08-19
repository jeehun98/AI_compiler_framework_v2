import type { Graph } from '../domain/graph';

export function cleanupUnusedNodes(graph: Graph): Graph {
  const retained = new Set<string>();
  const queue = [...graph.outputs];

  while (queue.length > 0) {
    const nodeId = queue.shift();
    if (!nodeId || retained.has(nodeId)) continue;
    retained.add(nodeId);
    for (const edge of graph.edges) {
      if (edge.targetNodeId === nodeId) queue.push(edge.sourceNodeId);
    }
  }

  return {
    ...graph,
    nodes: graph.nodes.filter(({ id }) => retained.has(id)),
    edges: graph.edges.filter(({ sourceNodeId, targetNodeId }) => retained.has(sourceNodeId) && retained.has(targetNodeId)),
  };
}
