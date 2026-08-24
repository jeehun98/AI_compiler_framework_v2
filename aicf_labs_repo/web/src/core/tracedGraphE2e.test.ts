import { describe, expect, it } from 'vitest';
import tracedGraphFixture from '../../../fixtures/toy_model_graph.json';
import type { Graph } from '../domain/graph';
import { findRewriteCandidates, searchRewriteCandidates } from './rewriteEngine';
import { validateGraph } from './validateGraph';

const tracedGraph = tracedGraphFixture as Graph;

function applyRule(graph: Graph, ruleId: string): Graph {
  const candidate = findRewriteCandidates(graph).find((item) => item.ruleId === ruleId);
  expect(candidate, `expected ${ruleId} on traced ToyModel graph`).toBeDefined();
  if (!candidate) throw new Error(`missing ${ruleId}`);
  return candidate.graph;
}

describe('Python-traced ToyModel graph', () => {
  it('uses the canonical Web graph schema and existing rewrite rules', () => {
    expect(validateGraph(tracedGraph).valid).toBe(true);

    const withoutAdd = applyRule(tracedGraph, 'add-zero');
    const simplified = applyRule(withoutAdd, 'mul-one');

    expect(simplified.nodes
      .filter(({ operatorId }) => !['input', 'constant'].includes(operatorId))
      .map(({ operatorId }) => operatorId)).toEqual(['matmul', 'relu']);
    expect(simplified.outputs).toEqual(['relu.0']);
    expect(validateGraph(simplified).valid).toBe(true);
  });

  it('reports mask filtering on a graph produced by model execution', () => {
    const result = searchRewriteCandidates(tracedGraph);
    const addZero = result.metrics.find(({ ruleId }) => ruleId === 'add-zero');
    const mulOne = result.metrics.find(({ ruleId }) => ruleId === 'mul-one');

    expect(addZero).toMatchObject({
      nodesScanned: 8,
      maskAccepted: 3,
      conditionAccepted: 1,
      rewritesApplied: 1,
      candidateReduction: 0.625,
    });
    expect(mulOne).toMatchObject({
      nodesScanned: 8,
      maskAccepted: 3,
      conditionAccepted: 1,
      rewritesApplied: 1,
      candidateReduction: 0.625,
    });
  });
});
