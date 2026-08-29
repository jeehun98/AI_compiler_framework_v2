import { describe, expect, it } from 'vitest';
import { matchesMask, OperatorMask } from '../domain/operator';
import type { RewriteCandidate } from '../domain/rewrite';
import { edge, graph, node } from '../test/graphFixtures';
import { OPERATORS } from '../catalog/operators';
import {
  graphFingerprint,
  nodesMatchingMask,
  searchRewriteCandidates,
} from './rewriteEngine';
import { REWRITE_RULES } from './rewriteRules';

function representativeGraph() {
  return graph(
    [
      node('x', 'input', { symbol: 'x', shape: ['m', 'k'] }),
      node('y', 'input', { symbol: 'y', shape: ['k', 'n'] }),
      node('z', 'input', { symbol: 'z', shape: ['n'] }),
      node('zero', 'constant', { value: 0 }),
      node('one', 'constant', { value: 1 }),
      node('two', 'constant', { value: 2 }),
      node('three', 'constant', { value: 3 }),
      node('mm', 'matmul'),
      node('add-zero-root', 'add'),
      node('relu-root', 'relu'),
      node('mul-one-root', 'mul'),
      node('add-two-root', 'add'),
      node('mul-zero-root', 'mul'),
      node('transpose-inner', 'transpose'),
      node('transpose-outer', 'transpose'),
      node('fold-add-root', 'add'),
      node('sum-root', 'reduceSum', { axis: 'all', keepDims: false }),
      node('dead-mul', 'mul'),
      node('dead-relu', 'relu'),
      node('single-transpose', 'transpose'),
    ],
    [
      edge('x', 'mm', 'in-0'),
      edge('y', 'mm', 'in-1'),
      edge('mm', 'add-zero-root', 'in-0'),
      edge('zero', 'add-zero-root', 'in-1'),
      edge('add-zero-root', 'relu-root', 'in-0'),
      edge('relu-root', 'mul-one-root', 'in-0'),
      edge('one', 'mul-one-root', 'in-1'),
      edge('mul-one-root', 'add-two-root', 'in-0'),
      edge('two', 'add-two-root', 'in-1'),
      edge('add-two-root', 'mul-zero-root', 'in-0'),
      edge('zero', 'mul-zero-root', 'in-1'),
      edge('x', 'transpose-inner', 'in-0'),
      edge('transpose-inner', 'transpose-outer', 'in-0'),
      edge('two', 'fold-add-root', 'in-0'),
      edge('three', 'fold-add-root', 'in-1'),
      edge('z', 'sum-root', 'in-0'),
      edge('x', 'dead-mul', 'in-0'),
      edge('two', 'dead-mul', 'in-1'),
      edge('dead-mul', 'dead-relu', 'in-0'),
      edge('y', 'single-transpose', 'in-0'),
    ],
    'Representative mask experiment',
    ['mul-zero-root', 'transpose-outer', 'fold-add-root', 'sum-root'],
  );
}

function candidateSignatures(candidates: readonly RewriteCandidate[]) {
  return candidates
    .map(({ id, ruleId, graph: candidateGraph }) => ({
      id,
      ruleId,
      graph: graphFingerprint(candidateGraph),
    }))
    .sort((left, right) => `${left.ruleId}:${left.id}`.localeCompare(`${right.ruleId}:${right.id}`));
}

describe('rewrite mask instrumentation', () => {
  it('reports screening, condition, rewrite counts, and derived rates', () => {
    const result = searchRewriteCandidates(representativeGraph());
    const byRule = Object.fromEntries(result.metrics.map((metric) => [metric.ruleId, metric]));

    expect(result.metrics.map(({
      ruleId,
      nodesScanned,
      maskAccepted,
      conditionAccepted,
      rewritesApplied,
    }) => ({ ruleId, nodesScanned, maskAccepted, conditionAccepted, rewritesApplied }))).toEqual([
      { ruleId: 'add-zero', nodesScanned: 20, maskAccepted: 8, conditionAccepted: 1, rewritesApplied: 1 },
      { ruleId: 'mul-one', nodesScanned: 20, maskAccepted: 8, conditionAccepted: 1, rewritesApplied: 1 },
      { ruleId: 'mul-zero', nodesScanned: 20, maskAccepted: 8, conditionAccepted: 1, rewritesApplied: 1 },
      { ruleId: 'constant-fold', nodesScanned: 20, maskAccepted: 8, conditionAccepted: 1, rewritesApplied: 1 },
      { ruleId: 'add-commute', nodesScanned: 20, maskAccepted: 6, conditionAccepted: 3, rewritesApplied: 3 },
      { ruleId: 'mul-commute', nodesScanned: 20, maskAccepted: 6, conditionAccepted: 3, rewritesApplied: 3 },
      { ruleId: 'double-transpose', nodesScanned: 20, maskAccepted: 3, conditionAccepted: 1, rewritesApplied: 1 },
      { ruleId: 'scale-through-linear', nodesScanned: 20, maskAccepted: 13, conditionAccepted: 0, rewritesApplied: 0 },
      { ruleId: 'scale-through-positive-homogeneous', nodesScanned: 20, maskAccepted: 13, conditionAccepted: 1, rewritesApplied: 1 },
      { ruleId: 'reassociate-associative-right', nodesScanned: 20, maskAccepted: 13, conditionAccepted: 0, rewritesApplied: 0 },
    ]);
    expect(byRule['add-zero'].maskPassRate).toBeCloseTo(0.4);
    expect(byRule['add-zero'].conditionPassRate).toBeCloseTo(0.125);
    expect(byRule['add-zero'].rewriteRate).toBeCloseTo(0.05);
    expect(byRule['add-zero'].candidateReduction).toBeCloseTo(0.6);
    expect(byRule['double-transpose'].candidateReduction).toBeCloseTo(0.85);
  });

  it('produces identical candidates with mask screening enabled and disabled', () => {
    const value = representativeGraph();
    const maskOn = searchRewriteCandidates(value, REWRITE_RULES, { useMask: true });
    const maskOff = searchRewriteCandidates(value, REWRITE_RULES, { useMask: false });

    expect(candidateSignatures(maskOn.candidates)).toEqual(candidateSignatures(maskOff.candidates));
    expect(maskOn.metrics.every(({ maskAccepted, nodesScanned }) => maskAccepted <= nodesScanned)).toBe(true);
    expect(maskOff.metrics.every(({ maskAccepted, nodesScanned }) => maskAccepted === nodesScanned)).toBe(true);
    expect(maskOff.metrics.every(({ candidateReduction }) => candidateReduction === 0)).toBe(true);
  });

  it('keeps every unmasked legal match in the masked candidate superset', () => {
    const value = representativeGraph();

    for (const rule of REWRITE_RULES) {
      const legalWithoutMask = rule.matchStructure(value, value.nodes);
      const maskCandidates = nodesMatchingMask(value, rule.requiredMask);
      const acceptedRootIds = new Set(maskCandidates.map(({ id }) => id));
      expect(
        legalWithoutMask.every(({ rootNodeId }) => acceptedRootIds.has(rootNodeId)),
        `${rule.id} mask removed a legal root`,
      ).toBe(true);

      const maskOn = searchRewriteCandidates(value, [rule], { useMask: true });
      const maskOff = searchRewriteCandidates(value, [rule], { useMask: false });
      expect(candidateSignatures(maskOn.candidates)).toEqual(candidateSignatures(maskOff.candidates));
    }
  });

  it('treats mask false positives as normal matcher rejections', () => {
    const result = searchRewriteCandidates(representativeGraph());
    const mulOne = result.metrics.find(({ ruleId }) => ruleId === 'mul-one');

    expect(mulOne).toMatchObject({ maskAccepted: 8, conditionAccepted: 1 });
    expect(mulOne?.conditionAccepted).toBeLessThan(mulOne?.maskAccepted ?? 0);
  });
});

describe('current feature usage', () => {
  const features = [
    ['ELEMENTWISE', OperatorMask.ELEMENTWISE],
    ['REDUCTION', OperatorMask.REDUCTION],
    ['COMMUTATIVE', OperatorMask.COMMUTATIVE],
    ['PURE', OperatorMask.PURE],
    ['SHAPE_PRESERVING', OperatorMask.SHAPE_PRESERVING],
    ['PERMUTATION', OperatorMask.PERMUTATION],
    ['BROADCAST', OperatorMask.BROADCAST],
  ] as const;

  it('records which operators and rules consume each feature', () => {
    const usage = Object.fromEntries(features.map(([name, feature]) => [name, {
      operators: OPERATORS.filter(({ mask }) => matchesMask(mask, feature)).map(({ id }) => id),
      rules: REWRITE_RULES.filter(({ requiredMask }) => matchesMask(requiredMask, feature)).map(({ id }) => id),
    }]));

    expect(usage).toEqual({
      ELEMENTWISE: {
        operators: ['add', 'mul', 'relu'],
        rules: ['add-zero', 'mul-one', 'mul-zero', 'constant-fold'],
      },
      REDUCTION: { operators: ['reduceSum'], rules: [] },
      COMMUTATIVE: { operators: ['add', 'mul'], rules: ['add-commute', 'mul-commute'] },
      PURE: {
        operators: ['add', 'mul', 'matmul', 'relu', 'transpose', 'reduceSum'],
        rules: REWRITE_RULES.map(({ id }) => id),
      },
      SHAPE_PRESERVING: { operators: ['relu'], rules: [] },
      PERMUTATION: { operators: ['transpose'], rules: ['double-transpose'] },
      BROADCAST: { operators: ['add', 'mul'], rules: [] },
    });
  });

  it('keeps legacy PURE screening behavior and uses PURE as the scale-rule root screen', () => {
    const value = representativeGraph();

    for (const rule of REWRITE_RULES.filter(({ id }) =>
      !id.startsWith('scale-through-') && id !== 'reassociate-associative-right')) {
      const withPure = nodesMatchingMask(value, rule.requiredMask).length;
      const withoutPure = nodesMatchingMask(
        value,
        rule.requiredMask & ~OperatorMask.PURE,
      ).length;
      expect(withPure, rule.id).toBe(withoutPure);
    }
    for (const rule of REWRITE_RULES.filter(({ id }) =>
      id.startsWith('scale-through-') || id === 'reassociate-associative-right')) {
      expect(nodesMatchingMask(value, rule.requiredMask)).toHaveLength(13);
      expect(nodesMatchingMask(value, rule.requiredMask & ~OperatorMask.PURE)).toHaveLength(20);
    }
  });

  it('measures the active discriminator features independently', () => {
    const value = representativeGraph();
    const marginalReduction = (ruleId: string, feature: number) => {
      const rule = REWRITE_RULES.find(({ id }) => id === ruleId);
      if (!rule) throw new Error(`Missing rule ${ruleId}`);
      const accepted = nodesMatchingMask(value, rule.requiredMask).length;
      const acceptedWithoutFeature = nodesMatchingMask(
        value,
        rule.requiredMask & ~feature,
      ).length;
      return acceptedWithoutFeature - accepted;
    };

    expect(marginalReduction('add-zero', OperatorMask.ELEMENTWISE)).toBe(5);
    expect(marginalReduction('add-commute', OperatorMask.COMMUTATIVE)).toBe(7);
    expect(marginalReduction('double-transpose', OperatorMask.PERMUTATION)).toBe(10);
  });
});
