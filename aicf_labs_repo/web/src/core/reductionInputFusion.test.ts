import { describe, expect, it } from 'vitest';
import type { Graph } from '../domain/graph';
import { LegalityStatus, type TransformationAttempt } from '../domain/rewrite';
import { MaterializationRequirement } from '../domain/semantic';
import { document, edge, graph, node } from '../test/graphFixtures';
import { parseGraphDocument, serializeGraphDocument } from './documentCodec';
import { graphFingerprint, searchRewriteCandidates } from './rewriteEngine';
import { REDUCTION_INPUT_FUSION_RULES } from './rewriteRules';
import { formatTransformationAttemptsText, serializeTransformationAttempts } from './transformationTrace';
import { validateGraph } from './validateGraph';

function reluReductionGraph(
  inputParameters: Parameters<typeof node>[2] = { symbol: 'x', shape: ['m', 'n'] },
  reductionParameters: Parameters<typeof node>[2] = { axis: 1, keepDims: false },
): Graph {
  return graph(
    [
      node('x', 'input', inputParameters),
      node('producer', 'relu'),
      node('consumer', 'reduceSum', reductionParameters),
      node('unrelated', 'constant', { value: 19 }),
    ],
    [
      edge('x', 'producer', 'in-0', 'x-producer'),
      edge('producer', 'consumer', 'in-0', 'producer-consumer'),
    ],
    'ReLU reduction input fusion',
    ['consumer'],
  );
}

function addReductionGraph(): Graph {
  return graph(
    [
      node('x', 'input', { symbol: 'x', shape: ['n'] }),
      node('bias', 'constant', { value: 2 }),
      node('producer', 'add'),
      node('consumer', 'reduceSum', { axis: 'all', keepDims: false }),
    ],
    [
      edge('x', 'producer', 'in-0'),
      edge('bias', 'producer', 'in-1'),
      edge('producer', 'consumer', 'in-0'),
    ],
    'Generic elementwise reduction fusion',
    ['consumer'],
  );
}

function search(value: Graph) {
  return searchRewriteCandidates(value, REDUCTION_INPUT_FUSION_RULES);
}

function producerAttempt(value: Graph, producerId = 'producer'): TransformationAttempt {
  const found = search(value).attempts.find(({ bindings }) => bindings.producer === producerId);
  expect(found, 'expected a producer-embedding attempt').toBeDefined();
  if (!found) throw new Error('Missing producer-embedding attempt');
  return found;
}

describe('elementwise producer to reduction reasoning', () => {
  it('accepts ReLU → ReduceSum through properties, direct dataflow, and dependency evidence', () => {
    const attempt = producerAttempt(reluReductionGraph());
    expect(attempt).toMatchObject({
      ruleId: 'embed-elementwise-producer-into-reduction',
      requiredProperties: [
        'ELEMENTWISE(operator)',
        'PURE(operator)',
        'REDUCTION(consumer)',
        'PURE(consumer)',
      ],
      requiredCapabilities: ['REDUCTION_INPUT_FUSION'],
      semanticDomain: 'ABSTRACT_REAL',
      status: LegalityStatus.APPLICABLE,
    });
    expect(attempt.mathematicalLegality.reason).toBe(
      'Corresponding producer values can be generated on demand and incrementally accumulated into a mergeable reduction partial state.',
    );
    expect(attempt.evidence).toMatchObject({
      kind: 'producer-embedding',
      producerOperatorId: 'relu',
      producerProperties: expect.arrayContaining(['ELEMENTWISE', 'PURE', 'SHAPE_PRESERVING']),
      consumerOperatorId: 'reduceSum',
      consumerProperties: expect.arrayContaining(['REDUCTION', 'PURE']),
      materialization: { intermediateNodeId: 'producer', requirement: MaterializationRequirement.NOT_REQUIRED },
    });
    expect(attempt.evidence).toMatchObject({
      producerSemantics: {
        resolvedDependencyFootprint: { kind: 'CORRESPONDING_ELEMENT' },
        resolvedReuse: { kind: 'NONE' },
      },
      consumerSemantics: {
        resolvedDependencyFootprint: { kind: 'FULL_AXIS', axis: 1 },
        partialState: { supportsIncrementalUpdate: true, supportsMerge: true },
      },
      graphFacts: { directDataflow: true, exclusiveUse: true, producerIsGraphOutput: false },
    });
    expect(attempt.evidence?.materialization.reason).toContain('immediately accumulated');
  });

  it('applies the same rule to Add without producer-name checks', () => {
    const attempt = producerAttempt(addReductionGraph());
    expect(attempt.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.evidence).toMatchObject({ producerOperatorId: 'add' });
    expect(attempt.evidence).toMatchObject({
      producerSemantics: {
        resolvedDependencyFootprint: { kind: 'CORRESPONDING_ELEMENT' },
        resolvedReuse: { kind: 'ACROSS_OUTPUTS' },
      },
    });
    expect(search(addReductionGraph()).candidates).toHaveLength(1);
  });

  it('keeps an unmodeled MatMul producer footprint as UNKNOWN', () => {
    const value = graph(
      [
        node('a', 'input', { symbol: 'A', shape: ['m', 'k'] }),
        node('b', 'input', { symbol: 'B', shape: ['k', 'n'] }),
        node('producer', 'matmul'),
        node('consumer', 'reduceSum', { axis: 'all', keepDims: false }),
      ],
      [
        edge('a', 'producer', 'in-0'),
        edge('b', 'producer', 'in-1'),
        edge('producer', 'consumer', 'in-0'),
      ],
      'Non-elementwise producer',
      ['consumer'],
    );
    const attempt = producerAttempt(value);
    expect(attempt.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.mathematicalLegality.reason).toContain('current semantic model cannot establish compatibility');
    expect(attempt.evidence?.producerSemantics.declaredDependencyFootprint).toMatchObject({
      kind: 'UNKNOWN',
      reason: expect.stringContaining('row/column dependency regions'),
    });
  });

  it('rejects a non-reduction consumer at property resolution', () => {
    const value = graph(
      [
        node('x', 'input', { symbol: 'x', shape: ['n'] }),
        node('producer', 'relu'),
        node('bias', 'constant', { value: 1 }),
        node('consumer', 'add'),
      ],
      [
        edge('x', 'producer', 'in-0'),
        edge('producer', 'consumer', 'in-0'),
        edge('bias', 'consumer', 'in-1'),
      ],
      'Non-reduction consumer',
      ['consumer'],
    );
    const attempt = producerAttempt(value);
    expect(attempt.status).toBe(LegalityStatus.REJECTED);
    expect(attempt.mathematicalLegality.reason).toContain('Required REDUCTION(consumer) property is absent');
  });
});

describe('fusion graph legality and materialization', () => {
  it('rejects a producer with multiple consumers while preserving semantic applicability', () => {
    const value = reluReductionGraph();
    value.nodes.push(node('bias', 'constant', { value: 1 }), node('observer', 'add'));
    value.edges.push(
      edge('producer', 'observer', 'in-0'),
      edge('bias', 'observer', 'in-1'),
    );
    value.outputs.push('observer');
    const attempt = producerAttempt(value);
    expect(attempt.mathematicalLegality.status).toBe(LegalityStatus.APPLICABLE);
    expect(attempt.graphLegality).toMatchObject({
      status: LegalityStatus.REJECTED,
      reason: 'The producer result has multiple consumers.',
    });
    expect(attempt.evidence?.materialization.requirement).toBe(MaterializationRequirement.REQUIRED);
  });

  it('rejects a producer graph output and records required materialization', () => {
    const value = reluReductionGraph();
    value.outputs.unshift('producer');
    const attempt = producerAttempt(value);
    expect(attempt.status).toBe(LegalityStatus.REJECTED);
    expect(attempt.graphLegality.reason).toContain('explicit graph output');
    expect(attempt.evidence?.materialization.requirement).toBe(MaterializationRequirement.REQUIRED);
  });

  it('keeps unknown producer shape as UNKNOWN', () => {
    const value = reluReductionGraph({}, { axis: 'all', keepDims: false });
    expect(validateGraph(value).valid).toBe(true);
    const attempt = producerAttempt(value);
    expect(attempt.mathematicalLegality.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.graphLegality.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.evidence?.materialization.requirement).toBe(MaterializationRequirement.UNKNOWN);
  });

  it('keeps unknown reduction-axis relation as UNKNOWN', () => {
    const value = reluReductionGraph({ symbol: 'x', shape: ['n'] }, {});
    expect(validateGraph(value).valid).toBe(true);
    const attempt = producerAttempt(value);
    expect(attempt.mathematicalLegality.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.mathematicalLegality.reason).toContain('reduction axis relation is not known');
    expect(attempt.graphLegality.reason).toContain('not evaluated');
  });

  it('keeps Transpose permutation dependency semantics as UNKNOWN', () => {
    const value = graph(
      [
        node('x', 'input', { symbol: 'x', shape: ['m', 'n'] }),
        node('producer', 'transpose'),
        node('consumer', 'reduceSum', { axis: 'all', keepDims: false }),
      ],
      [edge('x', 'producer', 'in-0'), edge('producer', 'consumer', 'in-0')],
      'Unsupported permutation footprint',
      ['consumer'],
    );
    const attempt = producerAttempt(value);
    expect(attempt.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.evidence?.producerSemantics.declaredDependencyFootprint).toMatchObject({
      kind: 'UNKNOWN',
      reason: expect.stringContaining('Permutation-index dependency mapping'),
    });
    expect(attempt.evidence?.materialization.requirement).toBe(MaterializationRequirement.UNKNOWN);
  });

  it('keeps non-scalar broadcast dependency mapping as UNKNOWN', () => {
    const value = graph(
      [
        node('left', 'input', { symbol: 'left', shape: ['n', 1] }),
        node('right', 'input', { symbol: 'right', shape: [1, 'm'] }),
        node('producer', 'add'),
        node('consumer', 'reduceSum', { axis: 'all', keepDims: false }),
      ],
      [
        edge('left', 'producer', 'in-0'),
        edge('right', 'producer', 'in-1'),
        edge('producer', 'consumer', 'in-0'),
      ],
      'Unsupported non-scalar broadcast mapping',
      ['consumer'],
    );
    const attempt = producerAttempt(value);
    expect(attempt.status).toBe(LegalityStatus.UNKNOWN);
    expect(attempt.mathematicalLegality.reason).toContain('Non-scalar broadcast dependency mapping');
    expect(attempt.evidence?.producerSemantics.resolvedDependencyFootprint.kind).toBe('UNKNOWN');
    expect(attempt.evidence?.producerSemantics.resolvedReuse.kind).toBe('UNKNOWN');
  });
});

describe('generic reduction semantic region representation', () => {
  it('preserves the computation graph and adds a validated deterministic region', () => {
    const original = reluReductionGraph();
    const snapshot = JSON.stringify(original);
    const candidate = search(original).candidates[0];
    expect(candidate).toBeDefined();
    if (!candidate) throw new Error('Missing fusion candidate');
    const transformed = candidate.graph;

    expect(JSON.stringify(original)).toBe(snapshot);
    expect(transformed.nodes).toEqual(original.nodes);
    expect(transformed.edges).toEqual(original.edges);
    expect(transformed.outputs).toEqual(['consumer']);
    expect(transformed.nodes.find(({ id }) => id === 'consumer')?.parameters).toEqual({ axis: 1, keepDims: false });
    expect(transformed.nodes.find(({ id }) => id === 'unrelated')).toEqual(
      original.nodes.find(({ id }) => id === 'unrelated'),
    );
    expect(transformed.semanticRegions).toEqual([{
      id: 'reduction-input-fusion:consumer:in-0:producer',
      kind: 'reduction-input-fusion',
      inputTransformNodeId: 'producer',
      reducerNodeId: 'consumer',
      reducerInputPort: 'in-0',
      producerInputNodeIds: ['x'],
      dependencyKind: 'ELEMENTWISE_CORRESPONDING_INPUTS',
      materializationRequirement: MaterializationRequirement.NOT_REQUIRED,
    }]);
    expect(validateGraph(transformed).valid).toBe(true);
    expect(graphFingerprint(transformed)).not.toBe(graphFingerprint(original));
    expect(search(transformed).attempts.some(({ bindings }) =>
      bindings.producer === 'producer' && bindings.consumer === 'consumer')).toBe(false);
    expect(serializeTransformationAttempts(search(original).attempts)).toBe(
      serializeTransformationAttempts(search(original).attempts),
    );
  });

  it('round-trips a semantic region through the versioned graph document codec', () => {
    const candidate = search(reluReductionGraph()).candidates[0];
    if (!candidate) throw new Error('Missing fusion candidate');
    const original = document(candidate.graph);
    expect(parseGraphDocument(serializeGraphDocument(original))).toEqual({ ok: true, value: original });
  });

  it('invalidates a stale region if the embedded producer later gains another use', () => {
    const candidate = search(reluReductionGraph()).candidates[0];
    if (!candidate) throw new Error('Missing fusion candidate');
    const stale = structuredClone(candidate.graph);
    stale.nodes.push(node('observer', 'relu'));
    stale.edges.push(edge('producer', 'observer', 'in-0'));
    stale.outputs.push('observer');
    expect(validateGraph(stale).issues).toContainEqual(expect.objectContaining({
      code: 'invalid-semantic-region',
    }));
  });
});

describe('producer embedding traces', () => {
  it('serializes applicable, rejected, and unknown materialization evidence', () => {
    const applicable = producerAttempt(reluReductionGraph());
    const shared = reluReductionGraph();
    shared.nodes.push(node('other', 'relu'));
    shared.edges.push(edge('producer', 'other', 'in-0'));
    shared.outputs.push('other');
    const rejected = producerAttempt(shared);
    const unknown = producerAttempt(reluReductionGraph({}, { axis: 'all', keepDims: false }));
    const attempts = [applicable, rejected, unknown];
    const parsed = JSON.parse(serializeTransformationAttempts(attempts)) as TransformationAttempt[];

    expect(parsed.map(({ status }) => status)).toEqual([
      LegalityStatus.APPLICABLE,
      LegalityStatus.REJECTED,
      LegalityStatus.UNKNOWN,
    ]);
    expect(parsed.map(({ evidence }) => evidence?.materialization.requirement)).toEqual([
      MaterializationRequirement.NOT_REQUIRED,
      MaterializationRequirement.REQUIRED,
      MaterializationRequirement.UNKNOWN,
    ]);
    expect(parsed[0].graphFacts).toEqual([expect.objectContaining({
      scope: 'graph-instance',
      kind: 'DIRECT_DATAFLOW',
    })]);
    expect(parsed[0].evidence).toMatchObject({
      producerSemantics: {
        resolvedDependencyFootprint: { kind: 'CORRESPONDING_ELEMENT' },
      },
      consumerSemantics: {
        resolvedDependencyFootprint: { kind: 'FULL_AXIS' },
        partialState: { supportsIncrementalUpdate: true, supportsMerge: true },
      },
      materialization: {
        requirement: 'NOT_REQUIRED',
        reason: expect.stringContaining('incremental partial state'),
      },
    });
    expect(formatTransformationAttemptsText(attempts)).toContain('materialization=NOT_REQUIRED');
    expect(formatTransformationAttemptsText(attempts)).toContain('materialization=REQUIRED');
    expect(formatTransformationAttemptsText(attempts)).toContain('materialization=UNKNOWN');
  });
});
