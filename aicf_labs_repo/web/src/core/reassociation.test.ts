import { describe, expect, it } from 'vitest';
import type { Graph } from '../domain/graph';
import { LegalityStatus, SemanticDomain, type TransformationAttempt } from '../domain/rewrite';
import { TransformationCapability } from '../domain/property';
import { edge, graph, node } from '../test/graphFixtures';
import { searchRewriteCandidates } from './rewriteEngine';
import { REASSOCIATION_RULES } from './rewriteRules';
import { formatTransformationAttemptsText, serializeTransformationAttempts } from './transformationTrace';
import { validateGraph } from './validateGraph';

function associativeGraph(
  operatorId: 'add' | 'mul' = 'add',
  inputParameters: Parameters<typeof node>[2] = { symbol: 'x', shape: ['n'] },
): Graph {
  return graph(
    [
      node('a', 'input', inputParameters),
      node('b', 'constant', { value: 2 }),
      node('c', 'input', inputParameters),
      node('inner', operatorId),
      node('outer', operatorId),
      node('unrelated', 'constant', { value: 17 }),
    ],
    [
      edge('a', 'inner', 'in-0', 'a-inner'),
      edge('b', 'inner', 'in-1', 'b-inner'),
      edge('inner', 'outer', 'in-0', 'inner-outer'),
      edge('c', 'outer', 'in-1', 'c-outer'),
    ],
    `${operatorId} reassociation`,
    ['outer'],
  );
}

function unsupportedMatMulGraph(): Graph {
  return graph(
    [
      node('a', 'input', { symbol: 'A', shape: ['m', 'k'] }),
      node('b', 'input', { symbol: 'B', shape: ['k', 'n'] }),
      node('c', 'input', { symbol: 'C', shape: ['n', 'p'] }),
      node('inner', 'matmul'),
      node('outer', 'matmul'),
    ],
    [
      edge('a', 'inner', 'in-0'),
      edge('b', 'inner', 'in-1'),
      edge('inner', 'outer', 'in-0'),
      edge('c', 'outer', 'in-1'),
    ],
    'MatMul without an associative claim',
    ['outer'],
  );
}

function search(value: Graph) {
  return searchRewriteCandidates(value, REASSOCIATION_RULES);
}

function attempt(value: Graph): TransformationAttempt {
  const found = search(value).attempts[0];
  expect(found, 'expected a reassociation attempt').toBeDefined();
  if (!found) throw new Error('Missing reassociation attempt');
  return found;
}

describe('property-driven associative reasoning', () => {
  it.each(['add', 'mul'] as const)('accepts %s without operator-name logic in the rule', (operatorId) => {
    const result = attempt(associativeGraph(operatorId));
    expect(result).toMatchObject({
      ruleId: 'reassociate-associative-right',
      status: LegalityStatus.APPLICABLE,
      semanticDomain: SemanticDomain.ABSTRACT_REAL,
      requiredCapabilities: [TransformationCapability.REASSOCIATION],
      requiredProperties: ['ASSOCIATIVE(operator)', 'PURE(operator)'],
    });
    expect(result.mathematicalLegality).toMatchObject({
      status: LegalityStatus.APPLICABLE,
      reason: 'The operator declares associativity under abstract semantics.',
    });
    expect(result.mathematicalLegality.evidence?.join(' ')).toContain('declares ASSOCIATIVE(operator)');
  });

  it('records property absence as the reason an otherwise matching binary chain is rejected', () => {
    const result = attempt(unsupportedMatMulGraph());
    expect(result.status).toBe(LegalityStatus.REJECTED);
    expect(result.mathematicalLegality.reason).toBe(
      "Required ASSOCIATIVE(operator) property is absent on 'matmul'.",
    );
    expect(result.graphLegality.reason).toContain('not evaluated');
    expect(result.targetGraphId).toBeUndefined();
  });

  it('requires the inner and outer nodes to have the same operator identity', () => {
    const value = associativeGraph('add');
    const inner = value.nodes.find(({ id }) => id === 'inner');
    if (!inner) throw new Error('Missing inner node');
    inner.operatorId = 'mul';
    expect(validateGraph(value).valid).toBe(true);
    expect(search(value).attempts).toEqual([]);
  });
});

describe('reassociation graph legality', () => {
  it('accepts a private inner result that is not a graph output', () => {
    const result = attempt(associativeGraph());
    expect(result.graphLegality.status).toBe(LegalityStatus.APPLICABLE);
    expect(result.graphLegality.checkedConditions).toEqual(expect.arrayContaining([
      'inner consumer count = 1',
      'inner is not an explicit graph output',
      'operator identity = add',
    ]));
  });

  it('rejects a shared inner result', () => {
    const value = associativeGraph();
    value.nodes.push(node('observer', 'relu'));
    value.edges.push(edge('inner', 'observer', 'in-0'));
    value.outputs.push('observer');
    const result = attempt(value);
    expect(result.status).toBe(LegalityStatus.REJECTED);
    expect(result.mathematicalLegality.status).toBe(LegalityStatus.APPLICABLE);
    expect(result.graphLegality.reason).toBe('The inner result has multiple consumers.');
  });

  it('rejects an inner result that is also an explicit graph output', () => {
    const value = associativeGraph();
    value.outputs.unshift('inner');
    const result = attempt(value);
    expect(result.status).toBe(LegalityStatus.REJECTED);
    expect(result.graphLegality.reason).toContain('explicit graph output');
  });

  it('keeps insufficient shape information as UNKNOWN after mathematical acceptance', () => {
    const value = associativeGraph('add', {});
    expect(validateGraph(value).valid).toBe(true);
    const result = attempt(value);
    expect(result.mathematicalLegality.status).toBe(LegalityStatus.APPLICABLE);
    expect(result.graphLegality.status).toBe(LegalityStatus.UNKNOWN);
    expect(result.status).toBe(LegalityStatus.UNKNOWN);
    expect(result.reason).toContain('Shape information is insufficient');
  });

  it('does not reason about an invalid source graph', () => {
    const value = associativeGraph();
    value.edges = value.edges.filter(({ id }) => id !== 'c-outer');
    expect(validateGraph(value).valid).toBe(false);
    expect(search(value)).toMatchObject({ candidates: [], attempts: [], metrics: [] });
  });
});

describe('canonical immutable reassociation', () => {
  it('rewires Op(Op(a,b),c) to Op(a,Op(b,c)) and preserves graph identity data', () => {
    const original = associativeGraph();
    const snapshot = JSON.stringify(original);
    const candidate = search(original).candidates[0];
    expect(candidate).toBeDefined();
    if (!candidate) throw new Error('Missing reassociation candidate');
    const rewritten = candidate.graph;

    expect(JSON.stringify(original)).toBe(snapshot);
    expect(rewritten.nodes).toEqual(original.nodes);
    expect(rewritten.outputs).toEqual(['outer']);
    expect(rewritten.nodes.find(({ id }) => id === 'b')).toEqual(
      original.nodes.find(({ id }) => id === 'b'),
    );
    expect(rewritten.nodes.find(({ id }) => id === 'unrelated')).toEqual(
      original.nodes.find(({ id }) => id === 'unrelated'),
    );
    expect(rewritten.edges).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: 'a-inner', sourceNodeId: 'a', targetNodeId: 'outer', targetPort: 'in-0' }),
      expect.objectContaining({ id: 'b-inner', sourceNodeId: 'b', targetNodeId: 'inner', targetPort: 'in-0' }),
      expect.objectContaining({ id: 'c-outer', sourceNodeId: 'c', targetNodeId: 'inner', targetPort: 'in-1' }),
      expect.objectContaining({ id: 'inner-outer', sourceNodeId: 'inner', targetNodeId: 'outer', targetPort: 'in-1' }),
    ]));
    expect(validateGraph(rewritten).valid).toBe(true);
    expect(search(rewritten).attempts).toEqual([]);
  });
});

describe('reassociation traces', () => {
  it('serializes applicable, property-rejected, and graph-rejected paths to JSON and text', () => {
    const applicable = attempt(associativeGraph());
    const propertyRejected = attempt(unsupportedMatMulGraph());
    const shared = associativeGraph();
    shared.nodes.push(node('observer', 'relu'));
    shared.edges.push(edge('inner', 'observer', 'in-0'));
    shared.outputs.push('observer');
    const graphRejected = attempt(shared);
    const attempts = [applicable, propertyRejected, graphRejected];
    const parsed = JSON.parse(serializeTransformationAttempts(attempts)) as TransformationAttempt[];

    expect(parsed.map(({ status }) => status)).toEqual([
      LegalityStatus.APPLICABLE,
      LegalityStatus.REJECTED,
      LegalityStatus.REJECTED,
    ]);
    expect(parsed[0].targetGraphId).toBeDefined();
    expect(parsed[1].mathematicalLegality.reason).toContain('property is absent');
    expect(parsed[2].graphLegality.reason).toContain('multiple consumers');
    const text = formatTransformationAttemptsText(attempts);
    expect(text).toContain('ReassociateAssociativeRight');
    expect(text).toContain('REJECTED');
    expect(serializeTransformationAttempts(search(associativeGraph()).attempts)).toBe(
      serializeTransformationAttempts(search(associativeGraph()).attempts),
    );
  });

  it('orders multiple structural matches deterministically by match id', () => {
    const value = graph(
      [
        ...['a', 'b', 'c', 'd', 'e', 'f'].map((id) =>
          node(id, 'input', { symbol: id, shape: ['n'] })),
        node('inner-z', 'add'),
        node('outer-z', 'add'),
        node('inner-a', 'add'),
        node('outer-a', 'add'),
      ],
      [
        edge('a', 'inner-z', 'in-0'),
        edge('b', 'inner-z', 'in-1'),
        edge('inner-z', 'outer-z', 'in-0'),
        edge('c', 'outer-z', 'in-1'),
        edge('d', 'inner-a', 'in-0'),
        edge('e', 'inner-a', 'in-1'),
        edge('inner-a', 'outer-a', 'in-0'),
        edge('f', 'outer-a', 'in-1'),
      ],
      'Two reassociation matches',
      ['outer-z', 'outer-a'],
    );
    const first = search(value).attempts;
    const second = search(value).attempts;
    expect(first.map(({ bindings }) => bindings.outer)).toEqual(['outer-a', 'outer-z']);
    expect(serializeTransformationAttempts(first)).toBe(serializeTransformationAttempts(second));
  });
});
