import { describe, expect, it } from 'vitest';
import { searchRewriteCandidates } from '../core/rewriteEngine';
import { validateGraph } from '../core/validateGraph';
import { getModelExample, MODEL_EXAMPLES } from './modelExamples';

describe('model examples', () => {
  it('keeps model and graph references valid', () => {
    expect(new Set(MODEL_EXAMPLES.map(({ id }) => id)).size).toBe(MODEL_EXAMPLES.length);
    for (const example of MODEL_EXAMPLES) {
      const graphIds = new Set(example.graphs.map(({ graph }) => graph.id));
      expect(graphIds.has(example.selectedGraphId), example.id).toBe(true);
      expect(example.model.graphIds.every((id) => graphIds.has(id)), example.id).toBe(true);
      expect(example.graphs.every(({ graph }) => validateGraph(graph).valid), example.id).toBe(true);
    }
  });

  it('loads TinyMLP with three model layers mapped to graph node identities', () => {
    const example = getModelExample('tiny-mlp');
    expect(example?.model).toMatchObject({
      name: 'TinyMLP',
      layers: [
        { name: 'Linear_0', type: 'Linear', graphNodeIds: ['matmul-0', 'add-0'] },
        { name: 'ReLU_0', type: 'ReLU', graphNodeIds: ['relu-0'] },
        { name: 'Linear_1', type: 'Linear', graphNodeIds: ['matmul-1', 'add-1'] },
      ],
    });
  });

  it('reuses ScaleThroughLinear in TinyScaleLinear', () => {
    const example = getModelExample('scale-matmul-left');
    expect(example?.model).toMatchObject({
      name: 'TinyScaleLinear',
      layers: [
        { name: 'ScaleLayer_0', graphNodeIds: ['scale'] },
        { name: 'Linear_0', graphNodeIds: ['matmul'] },
      ],
    });
    expect(searchRewriteCandidates(example!.graphs[0]!.graph).candidates).toContainEqual(expect.objectContaining({ ruleName: 'ScaleThroughLinear' }));
  });
});
