import { describe, expect, it } from 'vitest';
import { searchRewriteCandidates } from '../core/rewriteEngine';
import {
  REASSOCIATION_RULES,
  REDUCTION_INPUT_FUSION_RULES,
  SCALE_PROPAGATION_RULES,
} from '../core/rewriteRules';
import { validateGraph } from '../core/validateGraph';
import { LegalityStatus, type RewriteRule, type TransformationAttempt } from '../domain/rewrite';
import { MaterializationRequirement } from '../domain/semantic';
import { EXAMPLE_DOCUMENTS, getExample, type GraphExample } from './graphExamples';

const ORIGINAL_EXAMPLE_IDS = ['x-times-one', 'constant-expression', 'double-transpose'];

const TRANSFORMATION_STRUCTURES = {
  'scale-matmul-left': {
    operators: ['input', 'constant', 'mul', 'input', 'matmul'],
    edges: ['a:scale:in-0', 'alpha:scale:in-1', 'scale:matmul:in-0', 'b:matmul:in-1'],
    outputs: ['matmul'],
  },
  'scale-relu-positive': {
    operators: ['input', 'constant', 'mul', 'relu'],
    edges: ['x:scale:in-0', 'alpha:scale:in-1', 'scale:relu:in-0'],
    outputs: ['relu'],
  },
  'scale-relu-negative': {
    operators: ['input', 'constant', 'mul', 'relu'],
    edges: ['x:scale:in-0', 'alpha:scale:in-1', 'scale:relu:in-0'],
    outputs: ['relu'],
  },
  'reassociation-add': {
    operators: ['input', 'input', 'add', 'input', 'add'],
    edges: ['a:inner:in-0', 'b:inner:in-1', 'inner:outer:in-0', 'c:outer:in-1'],
    outputs: ['outer'],
  },
  'reassociation-shared-inner': {
    operators: ['input', 'input', 'add', 'input', 'add', 'relu'],
    edges: [
      'a:inner:in-0', 'b:inner:in-1', 'inner:outer:in-0', 'c:outer:in-1',
      'inner:observer:in-0',
    ],
    outputs: ['outer', 'observer'],
  },
  'fusion-relu-reduce-sum': {
    operators: ['input', 'relu', 'reduceSum'],
    edges: ['x:producer:in-0', 'producer:reducer:in-0'],
    outputs: ['reducer'],
  },
  'fusion-add-scalar-reduce-sum': {
    operators: ['input', 'constant', 'add', 'reduceSum'],
    edges: ['x:producer:in-0', 'scalar:producer:in-1', 'producer:reducer:in-0'],
    outputs: ['reducer'],
  },
  'fusion-shared-producer': {
    operators: ['input', 'relu', 'reduceSum', 'constant', 'add'],
    edges: [
      'x:producer:in-0', 'producer:reducer:in-0', 'producer:observer:in-0',
      'bias:observer:in-1',
    ],
    outputs: ['reducer', 'observer'],
  },
  'fusion-unknown-transpose': {
    operators: ['input', 'transpose', 'reduceSum'],
    edges: ['x:producer:in-0', 'producer:reducer:in-0'],
    outputs: ['reducer'],
  },
  'fusion-unknown-matmul': {
    operators: ['input', 'input', 'matmul', 'reduceSum'],
    edges: ['a:producer:in-0', 'b:producer:in-1', 'producer:reducer:in-0'],
    outputs: ['reducer'],
  },
} as const satisfies Partial<Record<GraphExample['id'], {
  operators: readonly string[];
  edges: readonly string[];
  outputs: readonly string[];
}>>;

function requireExample(id: GraphExample['id']): GraphExample {
  const found = getExample(id);
  expect(found, `missing example '${id}'`).toBeDefined();
  if (!found) throw new Error(`Missing example '${id}'`);
  return found;
}

function requireAttempt(
  exampleId: GraphExample['id'],
  rules: readonly RewriteRule[],
  ruleId: string,
  bindings: Readonly<Record<string, string>> = {},
): TransformationAttempt {
  const found = searchRewriteCandidates(requireExample(exampleId).document.graph, rules)
    .attempts.find((attempt) => attempt.ruleId === ruleId
      && Object.entries(bindings).every(([binding, nodeId]) => attempt.bindings[binding] === nodeId));
  expect(found, `missing ${ruleId} attempt for '${exampleId}'`).toBeDefined();
  if (!found) throw new Error(`Missing ${ruleId} attempt for '${exampleId}'`);
  return found;
}

describe('example gallery fixtures', () => {
  it('keeps every example id unique and preserves the original examples', () => {
    const ids = EXAMPLE_DOCUMENTS.map(({ id }) => id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual(expect.arrayContaining(ORIGINAL_EXAMPLE_IDS));
  });

  it('keeps every example graph valid and gives every node an initial position', () => {
    for (const example of EXAMPLE_DOCUMENTS) {
      expect(validateGraph(example.document.graph), example.id).toMatchObject({ valid: true });
      expect(Object.keys(example.document.layout.positions).sort(), example.id).toEqual(
        example.document.graph.nodes.map(({ id }) => id).sort(),
      );
    }
  });

  it.each(Object.entries(TRANSFORMATION_STRUCTURES))(
    'loads the intended nodes, edges, and outputs for %s',
    (id, expected) => {
      const graph = requireExample(id as GraphExample['id']).document.graph;
      expect(graph.nodes.map(({ operatorId }) => operatorId)).toEqual(expected.operators);
      expect(graph.edges.map(({ sourceNodeId, targetNodeId, targetPort }) =>
        `${sourceNodeId}:${targetNodeId}:${targetPort}`)).toEqual(expected.edges);
      expect(graph.outputs).toEqual(expected.outputs);
    },
  );
});

describe('example gallery reasoning probes', () => {
  it('derives linear scale propagation for the left MatMul input', () => {
    const attempt = requireAttempt('scale-matmul-left', SCALE_PROPAGATION_RULES, 'scale-through-linear');
    expect(attempt).toMatchObject({
      status: LegalityStatus.APPLICABLE,
      requiredCapabilities: ['SCALE_PROPAGATION'],
      requiredProperties: expect.arrayContaining(['LINEAR(input:0)']),
    });
  });

  it('accepts a non-negative ReLU scale and rejects a negative one mathematically', () => {
    const positive = requireAttempt(
      'scale-relu-positive', SCALE_PROPAGATION_RULES, 'scale-through-positive-homogeneous',
    );
    const negative = requireAttempt(
      'scale-relu-negative', SCALE_PROPAGATION_RULES, 'scale-through-positive-homogeneous',
    );
    expect(positive.status).toBe(LegalityStatus.APPLICABLE);
    expect(positive.mathematicalLegality.checkedConditions).toContain('alpha >= 0 is true');
    expect(negative.status).toBe(LegalityStatus.REJECTED);
    expect(negative.mathematicalLegality).toMatchObject({
      status: LegalityStatus.REJECTED,
      reason: 'Positive homogeneity requires alpha >= 0.',
    });
  });

  it('accepts Add reassociation and rejects a shared inner result only at graph legality', () => {
    const applicable = requireAttempt(
      'reassociation-add', REASSOCIATION_RULES, 'reassociate-associative-right',
    );
    const shared = requireAttempt(
      'reassociation-shared-inner', REASSOCIATION_RULES, 'reassociate-associative-right',
    );
    expect(applicable.status).toBe(LegalityStatus.APPLICABLE);
    expect(applicable.requiredCapabilities).toContain('REASSOCIATION');
    expect(shared).toMatchObject({
      status: LegalityStatus.REJECTED,
      mathematicalLegality: { status: LegalityStatus.APPLICABLE },
      graphLegality: {
        status: LegalityStatus.REJECTED,
        reason: 'The inner result has multiple consumers.',
      },
    });
  });

  it('derives generic reduction input fusion for ReLU and scalar Add producers', () => {
    const relu = requireAttempt(
      'fusion-relu-reduce-sum', REDUCTION_INPUT_FUSION_RULES,
      'embed-elementwise-producer-into-reduction',
      { producer: 'producer', consumer: 'reducer' },
    );
    const add = requireAttempt(
      'fusion-add-scalar-reduce-sum', REDUCTION_INPUT_FUSION_RULES,
      'embed-elementwise-producer-into-reduction',
      { producer: 'producer', consumer: 'reducer' },
    );
    for (const attempt of [relu, add]) {
      expect(attempt.status).toBe(LegalityStatus.APPLICABLE);
      expect(attempt.evidence).toMatchObject({
        producerSemantics: { resolvedDependencyFootprint: { kind: 'CORRESPONDING_ELEMENT' } },
        consumerSemantics: {
          partialState: { supportsIncrementalUpdate: true, supportsMerge: true },
        },
        materialization: { requirement: MaterializationRequirement.NOT_REQUIRED },
      });
    }
    expect(relu.evidence).toMatchObject({ producerOperatorId: 'relu' });
    expect(add.evidence).toMatchObject({
      producerOperatorId: 'add',
      producerSemantics: { resolvedReuse: { kind: 'ACROSS_OUTPUTS' } },
    });
  });

  it('requires materialization and rejects fusion for a shared producer', () => {
    const attempt = requireAttempt(
      'fusion-shared-producer', REDUCTION_INPUT_FUSION_RULES,
      'embed-elementwise-producer-into-reduction',
      { producer: 'producer', consumer: 'reducer' },
    );
    expect(attempt).toMatchObject({
      status: LegalityStatus.REJECTED,
      mathematicalLegality: { status: LegalityStatus.APPLICABLE },
      graphLegality: {
        status: LegalityStatus.REJECTED,
        reason: 'The producer result has multiple consumers.',
      },
      evidence: { materialization: { requirement: MaterializationRequirement.REQUIRED } },
    });
  });

  it.each([
    ['fusion-unknown-transpose', 'Permutation-index dependency mapping'],
    ['fusion-unknown-matmul', 'row/column dependency regions'],
  ] as const)('keeps unsupported footprint reasoning UNKNOWN for %s', (exampleId, reason) => {
    const attempt = requireAttempt(
      exampleId, REDUCTION_INPUT_FUSION_RULES, 'embed-elementwise-producer-into-reduction',
      { producer: 'producer', consumer: 'reducer' },
    );
    expect(attempt.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.evidence).toMatchObject({
      producerSemantics: {
        declaredDependencyFootprint: { kind: 'UNKNOWN', reason: expect.stringContaining(reason) },
      },
      materialization: { requirement: MaterializationRequirement.UNKNOWN },
    });
  });
});
