import type { Graph } from '../domain/graph';

export interface TopologyResult {
  order: string[];
  cycleNodeIds: string[];
}

export function findCycleNodeIds(graph: Graph): string[] {
  const nodeIds = new Set(graph.nodes.map(({ id }) => id));
  const outgoing = new Map([...nodeIds].map((id) => [id, [] as string[]]));
  for (const edge of graph.edges) {
    if (nodeIds.has(edge.sourceNodeId) && nodeIds.has(edge.targetNodeId)) {
      outgoing.get(edge.sourceNodeId)?.push(edge.targetNodeId);
    }
  }

  let nextIndex = 0;
  const indexes = new Map<string, number>();
  const lowLinks = new Map<string, number>();
  const stack: string[] = [];
  const onStack = new Set<string>();
  const cycleIds = new Set<string>();

  const visit = (nodeId: string) => {
    indexes.set(nodeId, nextIndex);
    lowLinks.set(nodeId, nextIndex);
    nextIndex += 1;
    stack.push(nodeId);
    onStack.add(nodeId);

    for (const targetId of outgoing.get(nodeId) ?? []) {
      if (!indexes.has(targetId)) {
        visit(targetId);
        lowLinks.set(nodeId, Math.min(lowLinks.get(nodeId) ?? 0, lowLinks.get(targetId) ?? 0));
      } else if (onStack.has(targetId)) {
        lowLinks.set(nodeId, Math.min(lowLinks.get(nodeId) ?? 0, indexes.get(targetId) ?? 0));
      }
    }

    if (lowLinks.get(nodeId) !== indexes.get(nodeId)) return;
    const component: string[] = [];
    let current: string | undefined;
    do {
      current = stack.pop();
      if (!current) break;
      onStack.delete(current);
      component.push(current);
    } while (current !== nodeId);

    const selfLoop = component.length === 1
      && (outgoing.get(component[0]) ?? []).includes(component[0]);
    if (component.length > 1 || selfLoop) component.forEach((id) => cycleIds.add(id));
  };

  [...nodeIds].sort().forEach((nodeId) => {
    if (!indexes.has(nodeId)) visit(nodeId);
  });
  return [...cycleIds].sort();
}

export function topologicalSort(graph: Graph): TopologyResult {
  const nodeIds = new Set(graph.nodes.map(({ id }) => id));
  const indegree = new Map([...nodeIds].map((id) => [id, 0]));
  const outgoing = new Map([...nodeIds].map((id) => [id, [] as string[]]));

  for (const edge of graph.edges) {
    if (!nodeIds.has(edge.sourceNodeId) || !nodeIds.has(edge.targetNodeId)) continue;
    outgoing.get(edge.sourceNodeId)?.push(edge.targetNodeId);
    indegree.set(edge.targetNodeId, (indegree.get(edge.targetNodeId) ?? 0) + 1);
  }

  const queue = [...nodeIds].filter((id) => indegree.get(id) === 0).sort();
  const order: string[] = [];

  while (queue.length > 0) {
    const id = queue.shift();
    if (!id) break;
    order.push(id);
    for (const target of outgoing.get(id) ?? []) {
      const nextDegree = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, nextDegree);
      if (nextDegree === 0) {
        queue.push(target);
        queue.sort();
      }
    }
  }

  return {
    order,
    cycleNodeIds: findCycleNodeIds(graph),
  };
}

export function sinkNodeIds(graph: Graph): string[] {
  const sources = new Set(graph.edges.map(({ sourceNodeId }) => sourceNodeId));
  return graph.nodes.map(({ id }) => id).filter((id) => !sources.has(id)).sort();
}
