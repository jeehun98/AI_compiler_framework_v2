import { describe, expect, it } from 'vitest';
import { edge, graph, node } from '../test/graphFixtures';
import { graphToLatex } from './graphToLatex';

describe('graphToLatex', () => {
  it('generates a stable expression using input-port order', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('two', 'constant', { value: 2 }), node('mul', 'mul'), node('zero', 'constant', { value: 0 }), node('add', 'add')],
      [edge('two', 'mul', 'in-1'), edge('x', 'mul', 'in-0'), edge('mul', 'add', 'in-0'), edge('zero', 'add', 'in-1')],
    );

    expect(graphToLatex(value).combinedLatex).toBe('y &= \\left(\\left(x \\cdot 2\\right) + 0\\right)');
  });

  it('supports MatMul, ReLU, Transpose, and ReduceSum notation', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'X', shape: ['m', 'k'] }), node('w', 'input', { symbol: 'W', shape: ['k', 'n'] }), node('mm', 'matmul'), node('relu', 'relu'), node('t', 'transpose'), node('sum', 'reduceSum', { axis: 1, keepDims: false })],
      [edge('x', 'mm', 'in-0'), edge('w', 'mm', 'in-1'), edge('mm', 'relu', 'in-0'), edge('relu', 't', 'in-0'), edge('t', 'sum', 'in-0')],
    );

    const latex = graphToLatex(value).combinedLatex;
    expect(latex).toContain('\\sum_{i_{1}}');
    expect(latex).toContain('\\operatorname{ReLU}');
    expect(latex).toContain('^{\\mathsf T}');
    expect(latex).toContain('X\\,W');
  });

  it('emits one labelled expression per sink', () => {
    const result = graphToLatex(graph([
      node('x', 'input', { symbol: 'x', shape: [] }),
      node('c', 'constant', { value: 4 }),
    ], []));

    expect(result.expressions).toHaveLength(2);
    expect(result.combinedLatex).toContain('y_{1}');
    expect(result.combinedLatex).toContain('y_{2}');
  });

  it('uses explicit outputs even when an output has downstream consumers', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('relu', 'relu')],
      [edge('x', 'relu', 'in-0')],
    );
    (value as unknown as { outputs: string[] }).outputs = ['x'];
    const result = graphToLatex(value);
    expect(result.expressions).toEqual([{ nodeId: 'x', label: 'y', latex: 'x' }]);
  });

  it('renders multiple explicit outputs in their declared order', () => {
    const value = graph(
      [node('x', 'input', { symbol: 'x', shape: [] }), node('relu', 'relu')],
      [edge('x', 'relu', 'in-0')],
    );
    (value as unknown as { outputs: string[] }).outputs = ['relu', 'x'];
    const result = graphToLatex(value);
    expect(result.expressions.map(({ nodeId }) => nodeId)).toEqual(['relu', 'x']);
  });

  it('refuses to recursively render a cyclic graph', () => {
    const result = graphToLatex(graph(
      [node('a', 'relu'), node('b', 'transpose')],
      [edge('a', 'b', 'in-0'), edge('b', 'a', 'in-0')],
    ));

    expect(result.expressions).toEqual([]);
    expect(result.combinedLatex).toBe('\\operatorname{invalid\\ DAG}');
  });

  it('fails safely instead of rendering malformed graphs', () => {
    const malformedGraphs = [
      graph(
        [node('x', 'input', { symbol: 'x', shape: [] })],
        [edge('x', 'missing', 'in-0')],
      ),
      graph(
        [node('x', 'input', { symbol: 'x', shape: [] }), node('a', 'constant', { value: 1 }), node('b', 'constant', { value: 2 }), node('add', 'add')],
        [edge('x', 'add', 'in-0', 'x-edge'), edge('a', 'add', 'in-1', 'a-edge'), edge('b', 'add', 'in-1', 'b-edge')],
      ),
    ];
    const unknown = graph([node('bad', 'input', { symbol: 'x', shape: [] })], []);
    (unknown.nodes[0] as { operatorId: string }).operatorId = '__proto__';
    malformedGraphs.push(unknown);

    for (const value of malformedGraphs) {
      expect(() => graphToLatex(value)).not.toThrow();
      const result = graphToLatex(value);
      expect(result.expressions).toEqual([]);
      expect(result.issues.length).toBeGreaterThan(0);
    }
  });
});
