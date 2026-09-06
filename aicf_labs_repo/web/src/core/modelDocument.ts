import type { Graph, GraphDocument, GraphNode } from '../domain/graph';
import type { ModelDocument, ModelLayer, ModelLayerType } from '../domain/model';

function safeId(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-|-$/g, '').toLowerCase() || 'model';
}

function layerType(node: GraphNode): ModelLayerType {
  switch (node.operatorId) {
    case 'matmul': return 'Linear';
    case 'relu': return 'ReLU';
    case 'mul': return 'Scale';
    case 'reduceSum': return 'Reduction';
    default: return 'Custom';
  }
}

function derivedLayerName(node: GraphNode, graph: Graph, counters: Map<string, number>): string {
  const incomingNodeIds = graph.edges.filter(({ targetNodeId }) => targetNodeId === node.id).map(({ sourceNodeId }) => sourceNodeId);
  const hasConstantInput = incomingNodeIds.some((id) => graph.nodes.find((candidate) => candidate.id === id)?.operatorId === 'constant');
  let base: string;
  switch (node.operatorId) {
    case 'matmul': base = 'Linear'; break;
    case 'relu': base = 'ReLU'; break;
    case 'reduceSum': base = 'Reduction'; break;
    case 'transpose': base = 'Transpose'; break;
    case 'mul': base = hasConstantInput ? 'ScaleLayer' : 'Multiply'; break;
    case 'add': base = hasConstantInput ? 'BiasAdd' : 'Add'; break;
    default: base = node.operatorId;
  }
  const index = counters.get(base) ?? 0;
  counters.set(base, index + 1);
  return `${base}_${index}`;
}

export function deriveModelLayersFromGraph(graph: Graph): ModelLayer[] {
  const counters = new Map<string, number>();
  return graph.nodes
    .filter(({ operatorId }) => operatorId !== 'input' && operatorId !== 'constant')
    .map((node) => ({
      id: node.id,
      name: derivedLayerName(node, graph, counters),
      type: layerType(node),
      graphNodeIds: [node.id],
      graphId: graph.id,
      source: 'derived',
    }));
}

export function createDerivedModelDocument(document: GraphDocument, name = document.graph.name, id = document.graph.id): ModelDocument {
  return {
    id: `model:${safeId(id)}`,
    name,
    layers: deriveModelLayersFromGraph(document.graph),
    graphIds: [document.graph.id],
  };
}

export function createUserDefinedLayer(model: ModelDocument, type: ModelLayerType): ModelLayer {
  const base = type === 'Custom' ? 'CustomLayer' : type;
  const occupiedNames = new Set(model.layers.map(({ name }) => name));
  let index = 0;
  while (occupiedNames.has(`${base}_${index}`)) index += 1;
  const name = `${base}_${index}`;
  const occupiedIds = new Set(model.layers.map(({ id }) => id));
  let id = safeId(name);
  let suffix = 2;
  while (occupiedIds.has(id)) id = `${safeId(name)}-${suffix++}`;
  return { id, name, type, graphNodeIds: [], source: 'user-defined' };
}

export function createEmptyGraphDocument(model: ModelDocument, name?: string): GraphDocument {
  const occupiedIds = new Set(model.graphIds);
  let index = 1;
  while (occupiedIds.has(`graph-${index}`)) index += 1;
  const graphName = name?.trim() || `ScratchGraph_${index}`;
  return {
    schemaVersion: 1,
    graph: { id: `graph-${index}`, name: graphName, nodes: [], edges: [], outputs: [] },
    layout: { positions: {}, viewport: { x: 0, y: 0, zoom: 1 } },
  };
}
