import type { GraphDocument, GraphNode, NodeParameters } from '../domain/graph';
import type { OperatorId } from '../domain/operator';

function node(id: string, operatorId: OperatorId, parameters: NodeParameters = {}): GraphNode {
  return { id, operatorId, parameters };
}

export interface GraphExample {
  id: 'x-times-one' | 'constant-expression' | 'double-transpose';
  label: string;
  document: GraphDocument;
}

export function createEmptyDocument(): GraphDocument {
  return {
    schemaVersion: 1,
    graph: { id: 'untitled', name: '새 계산 그래프', nodes: [], edges: [], outputs: [] },
    layout: { positions: {}, viewport: { x: 0, y: 0, zoom: 1 } },
  };
}

export const EXAMPLE_DOCUMENTS: readonly GraphExample[] = [
  {
    id: 'x-times-one',
    label: 'x × 1',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-x-times-one', name: 'x × 1',
        nodes: [node('x', 'input', { symbol: 'x', shape: [] }), node('one', 'constant', { value: 1 }), node('mul', 'mul')],
        edges: [
          { id: 'x-to-mul', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'mul', targetPort: 'in-0' },
          { id: 'one-to-mul', sourceNodeId: 'one', sourcePort: 'out', targetNodeId: 'mul', targetPort: 'in-1' },
        ],
        outputs: ['mul'],
      },
      layout: { positions: { x: { x: 40, y: 65 }, one: { x: 40, y: 225 }, mul: { x: 330, y: 145 } }, viewport: { x: 0, y: 0, zoom: 1 } },
    },
  },
  {
    id: 'constant-expression',
    label: '(2 × 3) + 0',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-constant-expression', name: '(2 × 3) + 0',
        nodes: [node('two', 'constant', { value: 2 }), node('three', 'constant', { value: 3 }), node('mul', 'mul'), node('zero', 'constant', { value: 0 }), node('add', 'add')],
        edges: [
          { id: 'two-to-mul', sourceNodeId: 'two', sourcePort: 'out', targetNodeId: 'mul', targetPort: 'in-0' },
          { id: 'three-to-mul', sourceNodeId: 'three', sourcePort: 'out', targetNodeId: 'mul', targetPort: 'in-1' },
          { id: 'mul-to-add', sourceNodeId: 'mul', sourcePort: 'out', targetNodeId: 'add', targetPort: 'in-0' },
          { id: 'zero-to-add', sourceNodeId: 'zero', sourcePort: 'out', targetNodeId: 'add', targetPort: 'in-1' },
        ],
        outputs: ['add'],
      },
      layout: { positions: { two: { x: 25, y: 30 }, three: { x: 25, y: 160 }, mul: { x: 275, y: 90 }, zero: { x: 275, y: 245 }, add: { x: 535, y: 145 } }, viewport: { x: 0, y: 0, zoom: 1 } },
    },
  },
  {
    id: 'double-transpose',
    label: 'Transpose(Transpose(x))',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-double-transpose', name: 'Transpose(Transpose(x))',
        nodes: [node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }), node('inner', 'transpose'), node('outer', 'transpose')],
        edges: [
          { id: 'x-to-inner', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'inner', targetPort: 'in-0' },
          { id: 'inner-to-outer', sourceNodeId: 'inner', sourcePort: 'out', targetNodeId: 'outer', targetPort: 'in-0' },
        ],
        outputs: ['outer'],
      },
      layout: { positions: { x: { x: 40, y: 145 }, inner: { x: 300, y: 145 }, outer: { x: 560, y: 145 } }, viewport: { x: 0, y: 0, zoom: 1 } },
    },
  },
] as const;

export function getExample(exampleId: string): GraphExample | undefined {
  return EXAMPLE_DOCUMENTS.find(({ id }) => id === exampleId);
}
