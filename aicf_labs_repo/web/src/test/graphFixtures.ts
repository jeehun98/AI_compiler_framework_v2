import type { Graph, GraphDocument, GraphEdge, GraphNode, InputPortId, NodeParameters } from '../domain/graph';
import type { OperatorId } from '../domain/operator';

export function node(id: string, operatorId: OperatorId, parameters: NodeParameters = {}): GraphNode {
  return { id, operatorId, parameters };
}

export function edge(sourceNodeId: string, targetNodeId: string, targetPort: InputPortId, id = `${sourceNodeId}-${targetNodeId}-${targetPort}`): GraphEdge {
  return { id, sourceNodeId, sourcePort: 'out', targetNodeId, targetPort };
}

export function graph(nodes: GraphNode[], edges: GraphEdge[], name = 'Test graph', outputs?: string[]): Graph {
  const sourceNodeIds = new Set(edges.map(({ sourceNodeId }) => sourceNodeId));
  const inferredOutputs = nodes.map(({ id }) => id).filter((id) => !sourceNodeIds.has(id));
  return { id: 'test-graph', name, nodes, edges, outputs: outputs ?? inferredOutputs };
}

export function document(graphValue: Graph): GraphDocument {
  return {
    schemaVersion: 1,
    graph: graphValue,
    layout: {
      positions: Object.fromEntries(graphValue.nodes.map((graphNode, index) => [graphNode.id, { x: index * 180, y: 80 }])),
      viewport: { x: 0, y: 0, zoom: 1 },
    },
  };
}
