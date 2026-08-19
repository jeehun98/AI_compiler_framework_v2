import { describe, expect, it } from 'vitest';
import { OPERATORS, OPERATOR_MAP } from './operators';

describe('operator catalog', () => {
  it('contains exactly the eight MVP operators', () => {
    expect(OPERATORS.map(({ id }) => id)).toEqual([
      'input', 'constant', 'add', 'mul', 'matmul', 'relu', 'transpose', 'reduceSum',
    ]);
  });

  it('keeps fixed arity metadata addressable by id', () => {
    expect(OPERATOR_MAP.input.arity).toBe(0);
    expect(OPERATOR_MAP.add.arity).toBe(2);
    expect(OPERATOR_MAP.transpose.arity).toBe(1);
  });
});
