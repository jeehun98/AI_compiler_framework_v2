import { describe, expect, it } from 'vitest';
import type { Graph } from '../domain/graph';
import { LegalityStatus, type TransformationAttempt } from '../domain/rewrite';
import { edge, graph, node } from '../test/graphFixtures';
import { searchRewriteCandidates } from './rewriteEngine';
import { SCALE_PROPAGATION_RULES } from './rewriteRules';
import { formatTransformationAttemptsText, serializeTransformationAttempts } from './transformationTrace';
import { validateGraph } from './validateGraph';

function search(value: Graph) {
  return searchRewriteCandidates(value, SCALE_PROPAGATION_RULES);
}

function findAttempt(value: Graph, ruleId: string, operatorId?: string): TransformationAttempt {
  const found = search(value).attempts.find((attempt) =>
    attempt.ruleId === ruleId && (!operatorId || attempt.bindings.operator === operatorId));
  expect(found, `expected ${ruleId} attempt`).toBeDefined();
  if (!found) throw new Error(`Missing ${ruleId} attempt`);
  return found;
}

function findCandidate(value: Graph, ruleId: string) {
  const found = search(value).candidates.find((candidate) => candidate.ruleId === ruleId);
  expect(found, `expected ${ruleId} candidate`).toBeDefined();
  if (!found) throw new Error(`Missing ${ruleId} candidate`);
  return found;
}

function unaryScaleGraph(
  alpha: number,
  operatorId: 'relu' | 'transpose' = 'relu',
  inputParameters: Parameters<typeof node>[2] = { symbol: 'x', shape: ['m', 'n'] },
): Graph {
  return graph(
    [
      node('x', 'input', inputParameters),
      node('alpha', 'constant', { value: alpha }),
      node('scale', 'mul'),
      node('target', operatorId),
    ],
    [
      edge('x', 'scale', 'in-0', 'value-to-scale'),
      edge('alpha', 'scale', 'in-1', 'alpha-to-scale'),
      edge('scale', 'target', 'in-0', 'scale-to-target'),
    ],
    `${operatorId} scale graph`,
    ['target'],
  );
}

function matmulScaleGraph(scaledPort: 'in-0' | 'in-1'): Graph {
  const scaledInput = scaledPort === 'in-0' ? 'a' : 'b';
  const fixedInput = scaledPort === 'in-0' ? 'b' : 'a';
  const fixedPort = scaledPort === 'in-0' ? 'in-1' : 'in-0';
  return graph(
    [
      node('a', 'input', { symbol: 'A', shape: ['m', 'k'] }),
      node('b', 'input', { symbol: 'B', shape: ['k', 'n'] }),
      node('alpha', 'constant', { value: -2 }),
      node('scale', 'mul'),
      node('matmul', 'matmul'),
    ],
    [
      edge(scaledInput, 'scale', 'in-0', 'value-to-scale'),
      edge('alpha', 'scale', 'in-1', 'alpha-to-scale'),
      edge('scale', 'matmul', scaledPort, 'scale-to-matmul'),
      edge(fixedInput, 'matmul', fixedPort, 'fixed-to-matmul'),
    ],
    `MatMul ${scaledPort} scale graph`,
    ['matmul'],
  );
}

function reduceSumScaleGraph(): Graph {
  return graph(
    [
      node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
      node('alpha', 'constant', { value: 3 }),
      node('scale', 'mul'),
      node('sum', 'reduceSum', { axis: 1, keepDims: true }),
    ],
    [
      edge('x', 'scale', 'in-0', 'value-to-scale'),
      edge('alpha', 'scale', 'in-1', 'alpha-to-scale'),
      edge('scale', 'sum', 'in-0', 'scale-to-sum'),
    ],
    'ReduceSum scale graph',
    ['sum'],
  );
}

describe('property-driven scale propagation mathematics', () => {
  it.each(['in-0', 'in-1'] as const)('accepts MatMul scale propagation at scoped %s', (port) => {
    const attempt = findAttempt(matmulScaleGraph(port), 'scale-through-linear');
    expect(attempt.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.requiredProperties).toContain(`LINEAR(input:${port.slice(3)})`);
    expect(attempt.mathematicalLegality.evidence?.join(' ')).toContain(`LINEAR(input:${port.slice(3)})`);
    expect(attempt.mathematicalLegality.checkedConditions).toContain(
      `${port === 'in-0' ? 'in-1' : 'in-0'} remains connected to ${port === 'in-0' ? 'b' : 'a'}`,
    );
  });

  it('accepts ReduceSum through its scoped LINEAR claim', () => {
    const attempt = findAttempt(reduceSumScaleGraph(), 'scale-through-linear');
    expect(attempt.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.requiredProperties).toContain('LINEAR(input:0)');
  });

  it.each([2, 0])('accepts ReLU positive homogeneity for alpha=%s', (alpha) => {
    const attempt = findAttempt(unaryScaleGraph(alpha), 'scale-through-positive-homogeneous');
    expect(attempt.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.mathematicalLegality.reason).toContain('Positive homogeneity');
    expect(attempt.mathematicalLegality.checkedConditions).toContain('alpha >= 0 is true');
  });

  it('rejects ReLU positive homogeneity for a negative scale', () => {
    const attempt = findAttempt(unaryScaleGraph(-1), 'scale-through-positive-homogeneous');
    expect(attempt.status).toBe(LegalityStatus.REJECTED);
    expect(attempt.mathematicalLegality).toMatchObject({
      status: LegalityStatus.REJECTED,
      reason: 'Positive homogeneity requires alpha >= 0.',
    });
    expect(attempt.targetGraphId).toBeUndefined();
  });

  it('records missing scale properties as a rejection for an unsupported operator', () => {
    const value = unaryScaleGraph(2, 'transpose');
    const linear = findAttempt(value, 'scale-through-linear');
    const positive = findAttempt(value, 'scale-through-positive-homogeneous');
    expect(linear.status).toBe(LegalityStatus.REJECTED);
    expect(linear.reason).toContain('Required LINEAR(input:0) property is absent');
    expect(positive.status).toBe(LegalityStatus.REJECTED);
    expect(positive.reason).toContain('Required POSITIVE_HOMOGENEOUS(input:0) property is absent');
  });
});

describe('scale propagation graph legality', () => {
  it('accepts a scale with one consumer that is not a graph output', () => {
    const attempt = findAttempt(unaryScaleGraph(2), 'scale-through-positive-homogeneous');
    expect(attempt.graphLegality.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.graphLegality.checkedConditions).toEqual(expect.arrayContaining([
      'scale consumer count = 1',
      'scale is not an explicit graph output',
    ]));
  });

  it('rejects a scale with multiple consumers', () => {
    const value = graph(
      [
        node('x', 'input', { symbol: 'x', shape: ['n'] }),
        node('alpha', 'constant', { value: 2 }),
        node('scale', 'mul'),
        node('left', 'relu'),
        node('right', 'relu'),
      ],
      [
        edge('x', 'scale', 'in-0'),
        edge('alpha', 'scale', 'in-1'),
        edge('scale', 'left', 'in-0'),
        edge('scale', 'right', 'in-0'),
      ],
      'Shared scale',
      ['left', 'right'],
    );
    const attempt = findAttempt(value, 'scale-through-positive-homogeneous', 'left');
    expect(attempt.status).toBe(LegalityStatus.REJECTED);
    expect(attempt.graphLegality.reason).toContain('exactly one consumer');
  });

  it('rejects a scale that is also an explicit graph output', () => {
    const value = unaryScaleGraph(2);
    value.outputs = ['scale', 'target'];
    const attempt = findAttempt(value, 'scale-through-positive-homogeneous');
    expect(attempt.status).toBe(LegalityStatus.REJECTED);
    expect(attempt.graphLegality.reason).toContain('explicit graph output');
  });

  it('keeps missing shape information as UNKNOWN', () => {
    const value = unaryScaleGraph(2, 'relu', {});
    expect(validateGraph(value).valid).toBe(true);
    const attempt = findAttempt(value, 'scale-through-positive-homogeneous');
    expect(attempt.mathematicalLegality.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.graphLegality.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.reason).toContain('Shape information is insufficient');
  });
});

describe('immutable scale rewrites', () => {
  it.each(['in-0', 'in-1'] as const)('moves a MatMul %s scale after MatMul', (port) => {
    const original = matmulScaleGraph(port);
    const snapshot = JSON.stringify(original);
    const rewritten = findCandidate(original, 'scale-through-linear').graph;
    const valueId = port === 'in-0' ? 'a' : 'b';

    expect(JSON.stringify(original)).toBe(snapshot);
    expect(rewritten.nodes).toEqual(original.nodes);
    expect(rewritten.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceNodeId: valueId, targetNodeId: 'matmul', targetPort: port }),
      expect.objectContaining({ sourceNodeId: 'matmul', targetNodeId: 'scale', targetPort: 'in-0' }),
      expect.objectContaining({ sourceNodeId: 'alpha', targetNodeId: 'scale', targetPort: 'in-1' }),
    ]));
    expect(rewritten.outputs).toEqual(['scale']);
    expect(validateGraph(rewritten).valid).toBe(true);
  });

  it('moves a scale after ReduceSum and preserves axis/keepDims', () => {
    const original = reduceSumScaleGraph();
    const rewritten = findCandidate(original, 'scale-through-linear').graph;
    expect(rewritten.nodes.find(({ id }) => id === 'sum')?.parameters).toEqual({ axis: 1, keepDims: true });
    expect(rewritten.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceNodeId: 'x', targetNodeId: 'sum', targetPort: 'in-0' }),
      expect.objectContaining({ sourceNodeId: 'sum', targetNodeId: 'scale', targetPort: 'in-0' }),
    ]));
    expect(rewritten.nodes.find(({ id }) => id === 'alpha')).toEqual(
      original.nodes.find(({ id }) => id === 'alpha'),
    );
    expect(rewritten.outputs).toEqual(['scale']);
    expect(validateGraph(rewritten).valid).toBe(true);
  });

  it('moves a non-negative scale after ReLU and preserves all node identities', () => {
    const original = unaryScaleGraph(2);
    const rewritten = findCandidate(original, 'scale-through-positive-homogeneous').graph;
    expect(rewritten.nodes.map(({ id }) => id)).toEqual(original.nodes.map(({ id }) => id));
    expect(rewritten.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceNodeId: 'x', targetNodeId: 'target', targetPort: 'in-0' }),
      expect.objectContaining({ sourceNodeId: 'target', targetNodeId: 'scale', targetPort: 'in-0' }),
      expect.objectContaining({ sourceNodeId: 'alpha', targetNodeId: 'scale', targetPort: 'in-1' }),
    ]));
    expect(rewritten.outputs).toEqual(['scale']);
    expect(validateGraph(rewritten).valid).toBe(true);
  });
});

describe('transformation attempt traces', () => {
  it('serializes applicable, rejected, and unknown attempts', () => {
    const applicable = findAttempt(matmulScaleGraph('in-0'), 'scale-through-linear');
    const rejected = findAttempt(unaryScaleGraph(-1), 'scale-through-positive-homogeneous');
    const unknown = findAttempt(unaryScaleGraph(2, 'relu', {}), 'scale-through-positive-homogeneous');
    const parsed = JSON.parse(serializeTransformationAttempts([applicable, rejected, unknown])) as TransformationAttempt[];

    expect(parsed.map(({ status }) => status)).toEqual([
      LegalityStatus.APPLICABLE,
      LegalityStatus.REJECTED,
      LegalityStatus.UNKNOWN,
    ]);
    expect(parsed[0]).toMatchObject({ semanticDomain: 'ABSTRACT_REAL', targetGraphId: applicable.targetGraphId });
    expect(parsed[1].targetGraphId).toBeUndefined();
    expect(parsed[2].graphLegality.status).toBe(LegalityStatus.UNKNOWN);
  });

  it('keeps attempt ordering and plain-text exploration deterministic', () => {
    const value = unaryScaleGraph(2);
    const first = search(value).attempts;
    const second = search(value).attempts;
    expect(serializeTransformationAttempts(first)).toBe(serializeTransformationAttempts(second));
    expect(first.map(({ ruleId }) => ruleId)).toEqual([
      'scale-through-linear',
      'scale-through-positive-homogeneous',
    ]);
    expect(formatTransformationAttemptsText(first)).toContain('ScaleThroughLinear');
    expect(formatTransformationAttemptsText(first)).toContain('ScaleThroughPositiveHomogeneous');
    expect(formatTransformationAttemptsText(first)).toContain('REJECTED');
  });
});
