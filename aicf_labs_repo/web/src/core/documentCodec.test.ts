import { describe, expect, it } from 'vitest';
import { document, edge, graph, node } from '../test/graphFixtures';
import { parseGraphDocument, serializeGraphDocument } from './documentCodec';

describe('graph document codec', () => {
  it('round-trips a versioned graph with layout', () => {
    const original = document(graph([node('x', 'input', { symbol: 'x', shape: ['m'] })], []));
    const result = parseGraphDocument(serializeGraphDocument(original));

    expect(result).toEqual({ ok: true, value: original });
  });

  it('rejects malformed JSON and unsupported schema versions', () => {
    expect(parseGraphDocument('{not-json').ok).toBe(false);
    expect(parseGraphDocument(JSON.stringify({ schemaVersion: 2 })).ok).toBe(false);
  });

  it('rejects unknown operators', () => {
    const original = document(graph([node('x', 'input', { symbol: 'x', shape: [] })], []));
    const raw = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
    raw.graph.nodes[0].operatorId = 'mystery';

    const result = parseGraphDocument(JSON.stringify(raw));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('mystery');
  });

  it('rejects prototype property operator ids', () => {
    const original = document(graph([node('x', 'input', { symbol: 'x', shape: [] })], []));
    const raw = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
    raw.graph.nodes[0].operatorId = '__proto__';
    expect(parseGraphDocument(JSON.stringify(raw)).ok).toBe(false);
  });

  it.each([
    ['input symbol', 'input', { symbol: 3, shape: [] }],
    ['input shape', 'input', { symbol: 'x', shape: 'm' }],
    ['constant value', 'constant', { value: '1' }],
    ['reduce axis', 'reduceSum', { axis: -1, keepDims: false }],
    ['reduce keepDims', 'reduceSum', { axis: 'all', keepDims: 'false' }],
    ['input extra field', 'input', { symbol: 'x', shape: [], extra: true }],
    ['constant extra field', 'constant', { value: 1, extra: true }],
    ['reduce extra field', 'reduceSum', { axis: 'all', keepDims: false, extra: true }],
    ['empty operator parameters', 'relu', { unexpected: true }],
  ])('rejects invalid %s parameters', (_, operatorId, parameters) => {
    const original = document(graph([node('x', 'input', { symbol: 'x', shape: [] })], []));
    const raw = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
    raw.graph.nodes[0] = { id: 'x', operatorId, parameters };
    expect(parseGraphDocument(JSON.stringify(raw)).ok).toBe(false);
  });

  it('rejects empty or duplicate node and edge ids', () => {
    const original = document(graph([node('x', 'input', { symbol: 'x', shape: [] })], []));
    const emptyId = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
    emptyId.graph.nodes[0].id = '';
    expect(parseGraphDocument(JSON.stringify(emptyId)).ok).toBe(false);

    const duplicateNodes = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
    duplicateNodes.graph.nodes.push({ ...duplicateNodes.graph.nodes[0] });
    expect(parseGraphDocument(JSON.stringify(duplicateNodes)).ok).toBe(false);

    const withEdges = document(graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('relu', 'relu')],
      [edge('x', 'relu', 'in-0', 'edge')],
    ));
    const duplicateEdges = JSON.parse(serializeGraphDocument(withEdges)) as Record<string, any>;
    duplicateEdges.graph.edges.push({ ...duplicateEdges.graph.edges[0] });
    expect(parseGraphDocument(JSON.stringify(duplicateEdges)).ok).toBe(false);
  });

  it('rejects dangling source or target ids and missing positions', () => {
    const original = document(graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('relu', 'relu')],
      [edge('x', 'relu', 'in-0')],
    ));
    for (const field of ['sourceNodeId', 'targetNodeId']) {
      const raw = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
      raw.graph.edges[0][field] = 'missing';
      expect(parseGraphDocument(JSON.stringify(raw)).ok).toBe(false);
    }
    const missingPosition = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
    delete missingPosition.layout.positions.relu;
    expect(parseGraphDocument(JSON.stringify(missingPosition)).ok).toBe(false);
  });

  it('rejects missing, duplicate, or unknown explicit outputs', () => {
    const original = document(graph([node('x', 'input', { symbol: 'x', shape: [] })], []));
    for (const outputs of [undefined, [], ['x', 'x'], ['missing']]) {
      const raw = JSON.parse(serializeGraphDocument(original)) as Record<string, any>;
      if (outputs === undefined) delete raw.graph.outputs;
      else raw.graph.outputs = outputs;
      expect(parseGraphDocument(JSON.stringify(raw)).ok).toBe(false);
    }
  });

  it('allows structurally valid but semantically incomplete work-in-progress graphs', () => {
    const incomplete = document(graph([node('add', 'add')], []));
    const result = parseGraphDocument(serializeGraphDocument(incomplete));
    expect(result.ok).toBe(true);
  });
});
