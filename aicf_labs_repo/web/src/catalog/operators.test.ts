import { describe, expect, it } from 'vitest';
import { matchesMask, OperatorMask, summarizeMasks } from '../domain/operator';
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

  it('indexes shared features without replacing operator identity', () => {
    const required = OperatorMask.ELEMENTWISE | OperatorMask.PURE;
    const matched = OPERATORS.filter(({ mask }) => matchesMask(mask, required));

    expect(matched.map(({ id }) => id)).toEqual(['add', 'mul', 'relu']);
    expect(matched.every(({ category }) => category === 'elementwise')).toBe(true);
  });

  it('summarizes a region with only common and present bit sets', () => {
    const summary = summarizeMasks([OPERATOR_MAP.add.mask, OPERATOR_MAP.relu.mask]);

    expect(summary.common).toBe(OperatorMask.ELEMENTWISE | OperatorMask.PURE);
    expect(matchesMask(summary.present, OperatorMask.BROADCAST)).toBe(true);
  });
});
