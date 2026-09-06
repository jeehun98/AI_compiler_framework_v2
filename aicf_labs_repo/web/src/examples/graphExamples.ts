import type { GraphDocument, GraphNode, NodeParameters } from '../domain/graph';
import type { OperatorId } from '../domain/operator';

function node(id: string, operatorId: OperatorId, parameters: NodeParameters = {}): GraphNode {
  return { id, operatorId, parameters };
}

export interface GraphExample {
  id:
    | 'x-times-one'
    | 'constant-expression'
    | 'double-transpose'
    | 'scale-matmul-left'
    | 'scale-relu-positive'
    | 'scale-relu-negative'
    | 'reassociation-add'
    | 'reassociation-shared-inner'
    | 'fusion-relu-reduce-sum'
    | 'fusion-add-scalar-reduce-sum'
    | 'fusion-shared-producer'
    | 'fusion-unknown-transpose'
    | 'fusion-unknown-matmul';
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
  {
    id: 'scale-matmul-left',
    label: 'Scale · MatMul Left',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-scale-matmul-left', name: 'Scale · MatMul Left',
        nodes: [
          node('a', 'input', { symbol: 'A', shape: ['m', 'k'] }),
          node('alpha', 'constant', { value: 2 }),
          node('scale', 'mul'),
          node('b', 'input', { symbol: 'B', shape: ['k', 'n'] }),
          node('matmul', 'matmul'),
        ],
        edges: [
          { id: 'a-to-scale', sourceNodeId: 'a', sourcePort: 'out', targetNodeId: 'scale', targetPort: 'in-0' },
          { id: 'alpha-to-scale', sourceNodeId: 'alpha', sourcePort: 'out', targetNodeId: 'scale', targetPort: 'in-1' },
          { id: 'scale-to-matmul', sourceNodeId: 'scale', sourcePort: 'out', targetNodeId: 'matmul', targetPort: 'in-0' },
          { id: 'b-to-matmul', sourceNodeId: 'b', sourcePort: 'out', targetNodeId: 'matmul', targetPort: 'in-1' },
        ],
        outputs: ['matmul'],
      },
      layout: {
        positions: {
          a: { x: 30, y: 30 }, alpha: { x: 30, y: 180 }, scale: { x: 280, y: 95 },
          b: { x: 280, y: 285 }, matmul: { x: 560, y: 165 },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'scale-relu-positive',
    label: 'Scale · ReLU Positive',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-scale-relu-positive', name: 'Scale · ReLU Positive',
        nodes: [
          node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
          node('alpha', 'constant', { value: 2 }),
          node('scale', 'mul'),
          node('relu', 'relu'),
        ],
        edges: [
          { id: 'x-to-scale', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'scale', targetPort: 'in-0' },
          { id: 'alpha-to-scale', sourceNodeId: 'alpha', sourcePort: 'out', targetNodeId: 'scale', targetPort: 'in-1' },
          { id: 'scale-to-relu', sourceNodeId: 'scale', sourcePort: 'out', targetNodeId: 'relu', targetPort: 'in-0' },
        ],
        outputs: ['relu'],
      },
      layout: {
        positions: { x: { x: 35, y: 70 }, alpha: { x: 35, y: 235 }, scale: { x: 300, y: 145 }, relu: { x: 575, y: 145 } },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'scale-relu-negative',
    label: 'Scale · ReLU Negative',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-scale-relu-negative', name: 'Scale · ReLU Negative',
        nodes: [
          node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
          node('alpha', 'constant', { value: -1 }),
          node('scale', 'mul'),
          node('relu', 'relu'),
        ],
        edges: [
          { id: 'x-to-scale', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'scale', targetPort: 'in-0' },
          { id: 'alpha-to-scale', sourceNodeId: 'alpha', sourcePort: 'out', targetNodeId: 'scale', targetPort: 'in-1' },
          { id: 'scale-to-relu', sourceNodeId: 'scale', sourcePort: 'out', targetNodeId: 'relu', targetPort: 'in-0' },
        ],
        outputs: ['relu'],
      },
      layout: {
        positions: { x: { x: 35, y: 70 }, alpha: { x: 35, y: 235 }, scale: { x: 300, y: 145 }, relu: { x: 575, y: 145 } },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'reassociation-add',
    label: 'Reassociation · Add',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-reassociation-add', name: 'Reassociation · Add',
        nodes: [
          node('a', 'input', { symbol: 'a', shape: ['n'] }),
          node('b', 'input', { symbol: 'b', shape: ['n'] }),
          node('inner', 'add'),
          node('c', 'input', { symbol: 'c', shape: ['n'] }),
          node('outer', 'add'),
        ],
        edges: [
          { id: 'a-to-inner', sourceNodeId: 'a', sourcePort: 'out', targetNodeId: 'inner', targetPort: 'in-0' },
          { id: 'b-to-inner', sourceNodeId: 'b', sourcePort: 'out', targetNodeId: 'inner', targetPort: 'in-1' },
          { id: 'inner-to-outer', sourceNodeId: 'inner', sourcePort: 'out', targetNodeId: 'outer', targetPort: 'in-0' },
          { id: 'c-to-outer', sourceNodeId: 'c', sourcePort: 'out', targetNodeId: 'outer', targetPort: 'in-1' },
        ],
        outputs: ['outer'],
      },
      layout: {
        positions: {
          a: { x: 30, y: 30 }, b: { x: 30, y: 190 }, inner: { x: 285, y: 105 },
          c: { x: 285, y: 300 }, outer: { x: 570, y: 175 },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'reassociation-shared-inner',
    label: 'Reassociation · Shared Inner',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-reassociation-shared-inner', name: 'Reassociation · Shared Inner',
        nodes: [
          node('a', 'input', { symbol: 'a', shape: ['n'] }),
          node('b', 'input', { symbol: 'b', shape: ['n'] }),
          node('inner', 'add'),
          node('c', 'input', { symbol: 'c', shape: ['n'] }),
          node('outer', 'add'),
          node('observer', 'relu'),
        ],
        edges: [
          { id: 'a-to-inner', sourceNodeId: 'a', sourcePort: 'out', targetNodeId: 'inner', targetPort: 'in-0' },
          { id: 'b-to-inner', sourceNodeId: 'b', sourcePort: 'out', targetNodeId: 'inner', targetPort: 'in-1' },
          { id: 'inner-to-outer', sourceNodeId: 'inner', sourcePort: 'out', targetNodeId: 'outer', targetPort: 'in-0' },
          { id: 'c-to-outer', sourceNodeId: 'c', sourcePort: 'out', targetNodeId: 'outer', targetPort: 'in-1' },
          { id: 'inner-to-observer', sourceNodeId: 'inner', sourcePort: 'out', targetNodeId: 'observer', targetPort: 'in-0' },
        ],
        outputs: ['outer', 'observer'],
      },
      layout: {
        positions: {
          a: { x: 30, y: 40 }, b: { x: 30, y: 210 }, inner: { x: 285, y: 125 },
          c: { x: 285, y: 315 }, outer: { x: 575, y: 75 }, observer: { x: 575, y: 260 },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'fusion-relu-reduce-sum',
    label: 'Fusion · ReLU → ReduceSum',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-fusion-relu-reduce-sum', name: 'Fusion · ReLU → ReduceSum',
        nodes: [
          node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
          node('producer', 'relu'),
          node('reducer', 'reduceSum', { axis: 1, keepDims: false }),
        ],
        edges: [
          { id: 'x-to-producer', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-0' },
          { id: 'producer-to-reducer', sourceNodeId: 'producer', sourcePort: 'out', targetNodeId: 'reducer', targetPort: 'in-0' },
        ],
        outputs: ['reducer'],
      },
      layout: {
        positions: { x: { x: 35, y: 145 }, producer: { x: 310, y: 145 }, reducer: { x: 585, y: 145 } },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'fusion-add-scalar-reduce-sum',
    label: 'Fusion · Add Scalar → ReduceSum',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-fusion-add-scalar-reduce-sum', name: 'Fusion · Add Scalar → ReduceSum',
        nodes: [
          node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
          node('scalar', 'constant', { value: 2 }),
          node('producer', 'add'),
          node('reducer', 'reduceSum', { axis: 'all', keepDims: false }),
        ],
        edges: [
          { id: 'x-to-producer', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-0' },
          { id: 'scalar-to-producer', sourceNodeId: 'scalar', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-1' },
          { id: 'producer-to-reducer', sourceNodeId: 'producer', sourcePort: 'out', targetNodeId: 'reducer', targetPort: 'in-0' },
        ],
        outputs: ['reducer'],
      },
      layout: {
        positions: {
          x: { x: 35, y: 65 }, scalar: { x: 35, y: 235 }, producer: { x: 310, y: 145 }, reducer: { x: 585, y: 145 },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'fusion-shared-producer',
    label: 'Fusion · Shared Producer',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-fusion-shared-producer', name: 'Fusion · Shared Producer',
        nodes: [
          node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
          node('producer', 'relu'),
          node('reducer', 'reduceSum', { axis: 1, keepDims: false }),
          node('bias', 'constant', { value: 1 }),
          node('observer', 'add'),
        ],
        edges: [
          { id: 'x-to-producer', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-0' },
          { id: 'producer-to-reducer', sourceNodeId: 'producer', sourcePort: 'out', targetNodeId: 'reducer', targetPort: 'in-0' },
          { id: 'producer-to-observer', sourceNodeId: 'producer', sourcePort: 'out', targetNodeId: 'observer', targetPort: 'in-0' },
          { id: 'bias-to-observer', sourceNodeId: 'bias', sourcePort: 'out', targetNodeId: 'observer', targetPort: 'in-1' },
        ],
        outputs: ['reducer', 'observer'],
      },
      layout: {
        positions: {
          x: { x: 35, y: 150 }, producer: { x: 300, y: 150 }, reducer: { x: 585, y: 55 },
          bias: { x: 300, y: 330 }, observer: { x: 585, y: 260 },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'fusion-unknown-transpose',
    label: 'Fusion · Unknown Transpose',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-fusion-unknown-transpose', name: 'Fusion · Unknown Transpose',
        nodes: [
          node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
          node('producer', 'transpose'),
          node('reducer', 'reduceSum', { axis: 'all', keepDims: false }),
        ],
        edges: [
          { id: 'x-to-producer', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-0' },
          { id: 'producer-to-reducer', sourceNodeId: 'producer', sourcePort: 'out', targetNodeId: 'reducer', targetPort: 'in-0' },
        ],
        outputs: ['reducer'],
      },
      layout: {
        positions: { x: { x: 35, y: 145 }, producer: { x: 310, y: 145 }, reducer: { x: 585, y: 145 } },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
  {
    id: 'fusion-unknown-matmul',
    label: 'Fusion · Unknown MatMul',
    document: {
      schemaVersion: 1,
      graph: {
        id: 'example-fusion-unknown-matmul', name: 'Fusion · Unknown MatMul',
        nodes: [
          node('a', 'input', { symbol: 'A', shape: ['m', 'k'] }),
          node('b', 'input', { symbol: 'B', shape: ['k', 'n'] }),
          node('producer', 'matmul'),
          node('reducer', 'reduceSum', { axis: 'all', keepDims: false }),
        ],
        edges: [
          { id: 'a-to-producer', sourceNodeId: 'a', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-0' },
          { id: 'b-to-producer', sourceNodeId: 'b', sourcePort: 'out', targetNodeId: 'producer', targetPort: 'in-1' },
          { id: 'producer-to-reducer', sourceNodeId: 'producer', sourcePort: 'out', targetNodeId: 'reducer', targetPort: 'in-0' },
        ],
        outputs: ['reducer'],
      },
      layout: {
        positions: {
          a: { x: 35, y: 65 }, b: { x: 35, y: 235 }, producer: { x: 310, y: 145 }, reducer: { x: 585, y: 145 },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
      },
    },
  },
] as const;

export function getExample(exampleId: string): GraphExample | undefined {
  return EXAMPLE_DOCUMENTS.find(({ id }) => id === exampleId);
}
