import { describe, expect, it } from 'vitest';
import { getExample } from '../examples/graphExamples';
import { searchRewriteCandidates } from './rewriteEngine';
import { buildCrossLayerModel, observationForSelection } from './layerObservations';

describe('cross-layer observation model', () => {
  it('references real semantic results without changing TransformationAttempt', () => {
    const graph = getExample('scale-matmul-left')!.document.graph;
    const search = searchRewriteCandidates(graph);
    const candidate = search.candidates.find(({ ruleName }) => ruleName === 'ScaleThroughLinear')!;
    const model = buildCrossLayerModel(graph, search.candidates, search.attempts, candidate.id, candidate.attemptId);

    const semantic = model.observations.find(({ id }) => id === `semantic:candidate:${candidate.id}`);
    expect(semantic).toMatchObject({ source: 'real', provenance: 'rewrite-engine', status: 'valid' });
    expect(semantic?.relatedTransformationAttemptIds).toEqual([candidate.attemptId]);
    expect(search.attempts.find(({ id }) => id === candidate.attemptId)?.status).toBe('APPLICABLE');
  });

  it('keeps downstream projections explicitly typed as placeholders', () => {
    const graph = getExample('scale-matmul-left')!.document.graph;
    const search = searchRewriteCandidates(graph);
    const candidate = search.candidates.find(({ ruleName }) => ruleName === 'ScaleThroughLinear')!;
    const model = buildCrossLayerModel(graph, search.candidates, search.attempts, candidate.id, candidate.attemptId);

    expect(model.trace.map(({ workspace }) => workspace)).toEqual(['graph', 'kernel', 'runtime', 'hardware']);
    expect(model.trace.slice(1).every(({ source }) => source === 'placeholder')).toBe(true);
    expect(model.observations.find(({ id }) => id === 'trace:hardware')).toMatchObject({
      source: 'placeholder',
      status: 'improved',
    });
  });

  it('resolves node-scoped observations for the active layer', () => {
    const graph = getExample('scale-matmul-left')!.document.graph;
    const search = searchRewriteCandidates(graph);
    const model = buildCrossLayerModel(graph, search.candidates, search.attempts, null, null);

    expect(observationForSelection(model.observations, 'kernel', null, 'matmul')).toMatchObject({
      workspace: 'kernel',
      relatedNodeIds: ['matmul'],
      source: 'placeholder',
    });
  });
});
