import { createDerivedModelDocument } from '../core/modelDocument';
import type { GraphDocument } from '../domain/graph';
import type { ModelExample } from '../domain/model';
import { EXAMPLE_DOCUMENTS, type GraphExample } from './graphExamples';

const tinyMlpGraph: GraphDocument = {
  schemaVersion: 1,
  graph: {
    id: 'tiny-mlp-forward',
    name: 'ForwardGraph',
    nodes: [
      { id: 'input', operatorId: 'input', parameters: { symbol: 'x', shape: ['m', 'k'] } },
      { id: 'weight-0', operatorId: 'input', parameters: { symbol: 'W0', shape: ['k', 'h'] } },
      { id: 'matmul-0', operatorId: 'matmul', parameters: {} },
      { id: 'bias-0', operatorId: 'input', parameters: { symbol: 'b0', shape: ['h'] } },
      { id: 'add-0', operatorId: 'add', parameters: {} },
      { id: 'relu-0', operatorId: 'relu', parameters: {} },
      { id: 'weight-1', operatorId: 'input', parameters: { symbol: 'W1', shape: ['h', 'n'] } },
      { id: 'matmul-1', operatorId: 'matmul', parameters: {} },
      { id: 'bias-1', operatorId: 'input', parameters: { symbol: 'b1', shape: ['n'] } },
      { id: 'add-1', operatorId: 'add', parameters: {} },
    ],
    edges: [
      { id: 'input-matmul-0', sourceNodeId: 'input', sourcePort: 'out', targetNodeId: 'matmul-0', targetPort: 'in-0' },
      { id: 'weight-0-matmul-0', sourceNodeId: 'weight-0', sourcePort: 'out', targetNodeId: 'matmul-0', targetPort: 'in-1' },
      { id: 'matmul-0-add-0', sourceNodeId: 'matmul-0', sourcePort: 'out', targetNodeId: 'add-0', targetPort: 'in-0' },
      { id: 'bias-0-add-0', sourceNodeId: 'bias-0', sourcePort: 'out', targetNodeId: 'add-0', targetPort: 'in-1' },
      { id: 'add-0-relu-0', sourceNodeId: 'add-0', sourcePort: 'out', targetNodeId: 'relu-0', targetPort: 'in-0' },
      { id: 'relu-0-matmul-1', sourceNodeId: 'relu-0', sourcePort: 'out', targetNodeId: 'matmul-1', targetPort: 'in-0' },
      { id: 'weight-1-matmul-1', sourceNodeId: 'weight-1', sourcePort: 'out', targetNodeId: 'matmul-1', targetPort: 'in-1' },
      { id: 'matmul-1-add-1', sourceNodeId: 'matmul-1', sourcePort: 'out', targetNodeId: 'add-1', targetPort: 'in-0' },
      { id: 'bias-1-add-1', sourceNodeId: 'bias-1', sourcePort: 'out', targetNodeId: 'add-1', targetPort: 'in-1' },
    ],
    outputs: ['add-1'],
  },
  layout: {
    positions: {
      input: { x: 20, y: 55 },
      'weight-0': { x: 20, y: 205 },
      'matmul-0': { x: 270, y: 115 },
      'bias-0': { x: 270, y: 300 },
      'add-0': { x: 510, y: 115 },
      'relu-0': { x: 750, y: 115 },
      'weight-1': { x: 750, y: 270 },
      'matmul-1': { x: 990, y: 145 },
      'bias-1': { x: 990, y: 300 },
      'add-1': { x: 1230, y: 145 },
    },
    viewport: { x: 0, y: 0, zoom: 1 },
  },
};

const TINY_MLP: ModelExample = {
  id: 'tiny-mlp',
  label: 'Tiny MLP',
  model: {
    id: 'model:tiny-mlp', name: 'TinyMLP', graphIds: [tinyMlpGraph.graph.id],
    layers: [
      { id: 'matmul-0', name: 'Linear_0', type: 'Linear', source: 'derived', graphId: tinyMlpGraph.graph.id, graphNodeIds: ['matmul-0', 'add-0'] },
      { id: 'relu-0', name: 'ReLU_0', type: 'ReLU', source: 'derived', graphId: tinyMlpGraph.graph.id, graphNodeIds: ['relu-0'] },
      { id: 'matmul-1', name: 'Linear_1', type: 'Linear', source: 'derived', graphId: tinyMlpGraph.graph.id, graphNodeIds: ['matmul-1', 'add-1'] },
    ],
  },
  graphs: [tinyMlpGraph],
  selectedGraphId: tinyMlpGraph.graph.id,
};

function modelNameForExample(example: GraphExample): string {
  switch (example.id) {
    case 'scale-matmul-left': return 'TinyScaleLinear';
    case 'fusion-relu-reduce-sum': return 'TinyReduction';
    default: return example.label;
  }
}

function modelLabelForExample(example: GraphExample): string {
  switch (example.id) {
    case 'scale-matmul-left': return 'Tiny Scale + Linear';
    case 'fusion-relu-reduce-sum': return 'Tiny Reduction';
    default: return example.label;
  }
}

function wrapGraphExample(example: GraphExample): ModelExample {
  const modelName = modelNameForExample(example);
  return {
    id: example.id,
    label: modelLabelForExample(example),
    model: createDerivedModelDocument(example.document, modelName, example.id),
    graphs: [example.document],
    selectedGraphId: example.document.graph.id,
  };
}

export const MODEL_EXAMPLES: readonly ModelExample[] = [
  TINY_MLP,
  ...EXAMPLE_DOCUMENTS.map(wrapGraphExample),
];

export function getModelExample(exampleId: string): ModelExample | undefined {
  return MODEL_EXAMPLES.find(({ id }) => id === exampleId);
}
