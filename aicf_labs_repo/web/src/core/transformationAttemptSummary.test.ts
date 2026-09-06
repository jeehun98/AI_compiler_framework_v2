import { describe, expect, it } from 'vitest';
import { getExample, type GraphExample } from '../examples/graphExamples';
import { LegalityStatus, type RewriteRule, type TransformationAttempt } from '../domain/rewrite';
import { searchRewriteCandidates } from './rewriteEngine';
import {
  REDUCTION_INPUT_FUSION_RULES,
  SCALE_PROPAGATION_RULES,
} from './rewriteRules';
import { selectTransformationAttemptReason } from './transformationAttemptSummary';

function attempt(
  exampleId: GraphExample['id'],
  rules: readonly RewriteRule[],
  ruleId: string,
  bindings: Readonly<Record<string, string>> = {},
): TransformationAttempt {
  const graph = getExample(exampleId)?.document.graph;
  if (!graph) throw new Error(`Missing example '${exampleId}'`);
  const found = searchRewriteCandidates(graph, rules).attempts.find((candidate) =>
    candidate.ruleId === ruleId
    && Object.entries(bindings).every(([binding, nodeId]) => candidate.bindings[binding] === nodeId));
  if (!found) throw new Error(`Missing attempt '${ruleId}' for '${exampleId}'`);
  return found;
}

describe('transformation attempt summary reason', () => {
  it('uses graph legality for an applicable attempt', () => {
    const value = attempt('scale-matmul-left', SCALE_PROPAGATION_RULES, 'scale-through-linear');
    expect(value.status).toBe(LegalityStatus.APPLICABLE);
    expect(selectTransformationAttemptReason(value)).toBe(value.graphLegality.reason);
  });

  it('uses the mathematical rejection reason', () => {
    const value = attempt(
      'scale-relu-negative', SCALE_PROPAGATION_RULES, 'scale-through-positive-homogeneous',
    );
    expect(selectTransformationAttemptReason(value)).toBe('Positive homogeneity requires alpha >= 0.');
  });

  it('uses the graph rejection reason when mathematics is applicable', () => {
    const value = attempt(
      'fusion-shared-producer', REDUCTION_INPUT_FUSION_RULES,
      'embed-elementwise-producer-into-reduction',
      { producer: 'producer', consumer: 'reducer' },
    );
    expect(selectTransformationAttemptReason(value)).toBe('The producer result has multiple consumers.');
  });

  it('uses detailed unsupported-footprint evidence for an unknown semantic result', () => {
    const value = attempt(
      'fusion-unknown-matmul', REDUCTION_INPUT_FUSION_RULES,
      'embed-elementwise-producer-into-reduction',
      { producer: 'producer', consumer: 'reducer' },
    );
    expect(value.status).toBe(LegalityStatus.UNKNOWN);
    expect(selectTransformationAttemptReason(value)).toContain('row/column dependency regions');
  });
});
