import { describe, expect, it } from 'vitest';
import { edge, graph, node } from '../test/graphFixtures';
import { canConnect, validateGraph } from './validateGraph';

describe('validateGraph', () => {
  it('accepts a complete acyclic graph and returns a topological order', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'x', shape: ['m', 'k'] }), node('w', 'input', { symbol: 'W', shape: ['k', 'n'] }), node('mm', 'matmul'), node('relu', 'relu')],
      [edge('x', 'mm', 'in-0'), edge('w', 'mm', 'in-1'), edge('mm', 'relu', 'in-0')],
    );

    const result = validateGraph(value);

    expect(result.valid).toBe(true);
    expect(result.issues).toEqual([]);
    expect(result.topologicalOrder?.at(-1)).toBe('relu');
  });

  it('reports every disconnected required input', () => {
    const result = validateGraph(graph([node('x', 'input', { symbol: 'x', shape: [] }), node('add', 'add')], [edge('x', 'add', 'in-0')]));

    expect(result.valid).toBe(false);
    expect(result.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'missing-input', nodeIds: ['add'] }),
    ]));
  });

  it('reports duplicate input occupancy and excessive arity', () => {
    const value = graph(
      [node('a', 'constant', { value: 1 }), node('b', 'constant', { value: 2 }), node('c', 'constant', { value: 3 }), node('add', 'add')],
      [edge('a', 'add', 'in-0', 'e1'), edge('b', 'add', 'in-0', 'e2'), edge('c', 'add', 'in-1', 'e3')],
    );

    const codes = validateGraph(value).issues.map(({ code }) => code);
    expect(codes).toContain('duplicate-input');
    expect(codes).toContain('invalid-arity');
  });

  it('detects an indirect cycle', () => {
    const value = graph(
      [node('a', 'relu'), node('b', 'transpose')],
      [edge('a', 'b', 'in-0'), edge('b', 'a', 'in-0')],
    );

    const result = validateGraph(value);
    expect(result.valid).toBe(false);
    expect(result.issues).toContainEqual(expect.objectContaining({ code: 'cycle', nodeIds: ['a', 'b'] }));
    expect(result.topologicalOrder).toBeUndefined();
  });

  it('does not classify a downstream node as part of the cycle', () => {
    const value = graph(
      [node('a', 'relu'), node('b', 'transpose'), node('downstream', 'relu')],
      [edge('a', 'b', 'in-0'), edge('b', 'a', 'in-0'), edge('b', 'downstream', 'in-0')],
    );
    const cycle = validateGraph(value).issues.find(({ code }) => code === 'cycle');
    expect(cycle?.nodeIds).toEqual(['a', 'b']);
  });

  it('detects a self-loop without throwing', () => {
    const value = graph([node('self', 'relu')], [edge('self', 'self', 'in-0')]);
    expect(validateGraph(value).issues).toContainEqual(expect.objectContaining({ code: 'cycle', nodeIds: ['self'] }));
  });

  it('treats unknown and prototype operator ids as validation errors', () => {
    for (const operatorId of ['mystery', '__proto__']) {
      const value = graph([node('bad', 'input', { symbol: 'x', shape: [] })], []);
      (value.nodes[0] as { operatorId: string }).operatorId = operatorId;
      expect(() => validateGraph(value)).not.toThrow();
      expect(validateGraph(value).issues).toContainEqual(expect.objectContaining({ code: 'unknown-operator' }));
    }
  });

  it('reports dangling edges and invalid ports', () => {
    const dangling = edge('missing', 'relu', 'in-0', 'dangling');
    const invalidPort = { ...edge('x', 'relu', 'in-0', 'invalid-port'), targetPort: 'in-4' as const };
    const result = validateGraph(graph([node('x', 'input', { symbol: 'x', shape: [] }), node('relu', 'relu')], [dangling, invalidPort]));

    expect(result.issues.map(({ code }) => code)).toEqual(expect.arrayContaining(['dangling-edge', 'invalid-port', 'missing-input']));
  });

  it('reports a missing target node as a dangling edge', () => {
    const result = validateGraph(graph(
      [node('x', 'input', { symbol: 'x', shape: [] })],
      [edge('x', 'missing-target', 'in-0', 'missing-target-edge')],
    ));
    expect(result.issues).toContainEqual(expect.objectContaining({ code: 'dangling-edge', edgeIds: ['missing-target-edge'] }));
  });

  it('requires at least one explicit output and validates each output id', () => {
    const value = graph([node('x', 'input', { symbol: 'x', shape: [] })], []);
    (value as unknown as { outputs: string[] }).outputs = [];
    expect(validateGraph(value).issues).toContainEqual(expect.objectContaining({ code: 'missing-output' }));

    (value as unknown as { outputs: string[] }).outputs = ['missing'];
    expect(validateGraph(value).issues).toContainEqual(expect.objectContaining({ code: 'invalid-output' }));
  });
});

describe('canConnect', () => {
  const base = graph(
    [node('x', 'input', { symbol: 'x', shape: [] }), node('a', 'relu'), node('b', 'relu')],
    [edge('x', 'a', 'in-0')],
  );

  it('accepts a free port that preserves acyclicity', () => {
    expect(canConnect(base, { sourceNodeId: 'a', targetNodeId: 'b', targetPort: 'in-0' })).toEqual({ allowed: true });
  });

  it('rejects occupied ports and cycle-producing connections', () => {
    expect(canConnect(base, { sourceNodeId: 'b', targetNodeId: 'a', targetPort: 'in-0' }).allowed).toBe(false);
    const cycleBase = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('add', 'add'), node('relu', 'relu')],
      [edge('x', 'add', 'in-0'), edge('add', 'relu', 'in-0')],
    );
    expect(canConnect(cycleBase, { sourceNodeId: 'relu', targetNodeId: 'add', targetPort: 'in-1' })).toEqual({
      allowed: false,
      reason: '이 연결은 순환을 만듭니다.',
    });
  });
});
