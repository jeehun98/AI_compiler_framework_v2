import { describe, expect, it } from 'vitest';
import type { Graph } from '../domain/graph';
import { edge, graph, node } from '../test/graphFixtures';
import { graphToLatex } from './graphToLatex';
import { cleanupUnusedNodes } from './graphCleanup';
import { findRewriteCandidates } from './rewriteEngine';
import { REWRITE_RULES } from './rewriteRules';
import { validateGraph } from './validateGraph';

function candidate(value: Graph, ruleId: string) {
  const found = findRewriteCandidates(value).find((item) => item.ruleId === ruleId);
  expect(found, `expected candidate for ${ruleId}`).toBeDefined();
  if (!found) throw new Error(`Missing ${ruleId} candidate`);
  return found;
}

describe('rewrite matching and application', () => {
  it('does not generate candidates for an invalid graph', () => {
    const invalid = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('zero', 'constant', { value: 0 }), node('add', 'add')],
      [edge('x', 'add', 'in-0', 'left'), edge('zero', 'add', 'in-1', 'right'), edge('add', 'add', 'in-1', 'self-loop')],
    );
    expect(findRewriteCandidates(invalid)).toEqual([]);
  });

  it('eliminates x + 0 without mutating the original graph', () => {
    const original = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('zero', 'constant', { value: 0 }), node('add', 'add')],
      [edge('x', 'add', 'in-0'), edge('zero', 'add', 'in-1')],
    );
    const snapshot = JSON.stringify(original);

    const rewritten = candidate(original, 'add-zero').graph;

    expect(JSON.stringify(original)).toBe(snapshot);
    expect(rewritten.nodes.map(({ id }) => id).sort()).toEqual(['x', 'zero']);
    expect(graphToLatex(rewritten).combinedLatex).toBe('y &= x');
  });

  it('eliminates x × 1 and preserves a shared identity constant', () => {
    const original = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('one', 'constant', { value: 1 }), node('mul', 'mul'), node('relu', 'relu')],
      [edge('x', 'mul', 'in-0'), edge('one', 'mul', 'in-1'), edge('one', 'relu', 'in-0')],
    );

    const rewritten = candidate(original, 'mul-one').graph;

    expect(rewritten.nodes.some(({ id }) => id === 'one')).toBe(true);
    expect(rewritten.nodes.some(({ id }) => id === 'mul')).toBe(false);
    expect(validateGraph(rewritten).valid).toBe(true);
  });

  it('replaces x × 0 with the zero node without implicitly deleting provenance', () => {
    const original = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('zero', 'constant', { value: 0 }), node('mul', 'mul'), node('relu', 'relu')],
      [edge('x', 'mul', 'in-0'), edge('zero', 'mul', 'in-1'), edge('mul', 'relu', 'in-0')],
    );

    const rewritten = candidate(original, 'mul-zero').graph;

    expect(rewritten.nodes.map(({ id }) => id).sort()).toEqual(['relu', 'x', 'zero']);
    expect(rewritten.edges).toContainEqual(expect.objectContaining({ sourceNodeId: 'zero', targetNodeId: 'relu' }));
    expect(validateGraph(rewritten).valid).toBe(true);

    const cleaned = cleanupUnusedNodes(rewritten);
    expect(cleaned.nodes.map(({ id }) => id).sort()).toEqual(['relu', 'zero']);
  });

  it('constant-folds scalar Add, Mul, and ReLU', () => {
    const cases = [
      {
        expected: 5,
        value: graph([node('a', 'constant', { value: 2 }), node('b', 'constant', { value: 3 }), node('root', 'add')], [edge('a', 'root', 'in-0'), edge('b', 'root', 'in-1')]),
      },
      {
        expected: 6,
        value: graph([node('a', 'constant', { value: 2 }), node('b', 'constant', { value: 3 }), node('root', 'mul')], [edge('a', 'root', 'in-0'), edge('b', 'root', 'in-1')]),
      },
      {
        expected: 0,
        value: graph([node('a', 'constant', { value: -4 }), node('root', 'relu')], [edge('a', 'root', 'in-0')]),
      },
    ];

    for (const testCase of cases) {
      const rewritten = candidate(testCase.value, 'constant-fold').graph;
      expect(rewritten.nodes).toContainEqual(expect.objectContaining({ id: 'root', operatorId: 'constant', parameters: { value: testCase.expected } }));
      expect(rewritten.nodes.map(({ id }) => id).sort()).toEqual(['a', ...(testCase.value.nodes.some(({ id }) => id === 'b') ? ['b'] : []), 'root'].sort());
      expect(rewritten.edges).toEqual([]);
    }
  });

  it('swaps Add and Mul input ports without changing node count', () => {
    for (const operatorId of ['add', 'mul'] as const) {
      const original = graph(
        [node('left', 'input', { symbol: 'a', shape: [] }), node('right', 'input', { symbol: 'b', shape: [] }), node('root', operatorId)],
        [edge('left', 'root', 'in-0', 'left-edge'), edge('right', 'root', 'in-1', 'right-edge')],
      );
      const rewritten = candidate(original, `${operatorId}-commute`).graph;
      expect(rewritten.nodes).toHaveLength(original.nodes.length);
      expect(rewritten.edges).toEqual(expect.arrayContaining([
        expect.objectContaining({ id: 'left-edge', targetPort: 'in-1' }),
        expect.objectContaining({ id: 'right-edge', targetPort: 'in-0' }),
      ]));
    }
  });

  it('removes two consecutive Transpose nodes exactly', () => {
    const original = graph(
      [node('x', 'input', { symbol: 'X', shape: ['m', 'n'] }), node('inner', 'transpose'), node('outer', 'transpose'), node('sum', 'reduceSum', { axis: 'all', keepDims: false })],
      [edge('x', 'inner', 'in-0'), edge('inner', 'outer', 'in-0'), edge('outer', 'sum', 'in-0')],
    );

    const rewritten = candidate(original, 'double-transpose').graph;

    expect(rewritten.nodes.map(({ id }) => id).sort()).toEqual(['sum', 'x']);
    expect(rewritten.edges).toContainEqual(expect.objectContaining({ sourceNodeId: 'x', targetNodeId: 'sum' }));
    expect(validateGraph(rewritten).valid).toBe(true);
  });

  it('keeps a shared inner Transpose that has another consumer', () => {
    const original = graph(
      [node('x', 'input', { symbol: 'X', shape: ['m', 'n'] }), node('inner', 'transpose'), node('outer', 'transpose'), node('relu', 'relu')],
      [edge('x', 'inner', 'in-0'), edge('inner', 'outer', 'in-0'), edge('inner', 'relu', 'in-0')],
      'Shared transpose',
      ['outer', 'relu'],
    );
    const rewritten = candidate(original, 'double-transpose').graph;
    expect(rewritten.nodes.some(({ id }) => id === 'inner')).toBe(true);
    expect(rewritten.edges).toContainEqual(expect.objectContaining({ sourceNodeId: 'inner', targetNodeId: 'relu' }));
    expect(rewritten.outputs).toEqual(['x', 'relu']);
    expect(validateGraph(rewritten).valid).toBe(true);
  });

  it('does not match identity or transpose rules without the required pattern', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('two', 'constant', { value: 2 }), node('mul', 'mul'), node('t', 'transpose')],
      [edge('x', 'mul', 'in-0'), edge('two', 'mul', 'in-1'), edge('mul', 't', 'in-0')],
    );
    const ids = findRewriteCandidates(value).map(({ ruleId }) => ruleId);
    expect(ids).not.toContain('mul-one');
    expect(ids).not.toContain('mul-zero');
    expect(ids).not.toContain('double-transpose');
  });

  it('rewires every consumer and preserves shared subgraphs', () => {
    const original = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('zero', 'constant', { value: 0 }), node('add', 'add'), node('left', 'relu'), node('right', 'transpose')],
      [edge('x', 'add', 'in-0'), edge('zero', 'add', 'in-1'), edge('add', 'left', 'in-0'), edge('add', 'right', 'in-0')],
    );
    const rewritten = candidate(original, 'add-zero').graph;
    expect(rewritten.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceNodeId: 'x', targetNodeId: 'left' }),
      expect.objectContaining({ sourceNodeId: 'x', targetNodeId: 'right' }),
    ]));
    expect(rewritten.nodes.some(({ id }) => id === 'zero')).toBe(true);
    expect(validateGraph(rewritten).valid).toBe(true);
  });

  it('returns only candidates that pass graph validation', () => {
    const value = graph(
      [node('two', 'constant', { value: 2 }), node('three', 'constant', { value: 3 }), node('mul', 'mul'), node('zero', 'constant', { value: 0 }), node('add', 'add')],
      [edge('two', 'mul', 'in-0'), edge('three', 'mul', 'in-1'), edge('mul', 'add', 'in-0'), edge('zero', 'add', 'in-1')],
    );
    const candidates = findRewriteCandidates(value);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(({ graph: candidateGraph }) => validateGraph(candidateGraph).valid)).toBe(true);
  });
});

describe('rewrite metadata', () => {
  it('defines the seven requested rules in deterministic order', () => {
    expect(REWRITE_RULES.map(({ id }) => id)).toEqual([
      'add-zero', 'mul-one', 'mul-zero', 'constant-fold', 'add-commute', 'mul-commute', 'double-transpose',
    ]);
  });

  it('classifies every rule and keeps four independent freedom axes', () => {
    for (const rule of REWRITE_RULES) {
      expect(['exact', 'conditionally-exact', 'approximate']).toContain(rule.exactness);
      expect(Object.keys(rule.freedom).sort()).toEqual(['algebraic', 'implementation', 'numerical', 'structural']);
    }
    expect(REWRITE_RULES.find(({ id }) => id === 'double-transpose')?.exactness).toBe('exact');
  });

  it('deduplicates candidates that produce the same graph fingerprint', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('zero', 'constant', { value: 0 }), node('add', 'add')],
      [edge('x', 'add', 'in-0'), edge('zero', 'add', 'in-1')],
    );
    const first = REWRITE_RULES[0];
    const duplicate = { ...first, id: 'duplicate-add-zero', name: 'Duplicate add zero' };
    expect(findRewriteCandidates(value, [first, duplicate])).toHaveLength(1);
  });
});
