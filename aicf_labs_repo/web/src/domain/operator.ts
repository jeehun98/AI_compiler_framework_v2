export type OperatorId =
  | 'input'
  | 'constant'
  | 'add'
  | 'mul'
  | 'matmul'
  | 'relu'
  | 'transpose'
  | 'reduceSum';

export type OperatorCategory =
  | 'source'
  | 'elementwise'
  | 'matrix'
  | 'layout'
  | 'reduction';

import type { FreedomProfile } from './freedom';

// Positive, cheap screening features. Absence is not proof of the opposite.
export const OperatorMask = {
  NONE: 0,
  ELEMENTWISE: 1 << 0,
  REDUCTION: 1 << 1,
  COMMUTATIVE: 1 << 2,
  PURE: 1 << 3,
  SHAPE_PRESERVING: 1 << 4,
  PERMUTATION: 1 << 5,
  BROADCAST: 1 << 6,
} as const;

export type OperatorMask = number;

export interface MaskSummary {
  common: OperatorMask;
  present: OperatorMask;
}

export function matchesMask(mask: OperatorMask, required: OperatorMask): boolean {
  return (mask & required) === required;
}

export function summarizeMasks(masks: readonly OperatorMask[]): MaskSummary {
  if (masks.length === 0) return { common: OperatorMask.NONE, present: OperatorMask.NONE };
  return {
    common: masks.reduce((common, mask) => common & mask),
    present: masks.reduce((present, mask) => present | mask, OperatorMask.NONE),
  };
}

export interface InputPortDefinition {
  id: `in-${number}`;
  label: string;
  required: true;
}

export interface ShapeRule {
  notation: string;
  constraints: string[];
}

export interface Operator {
  id: OperatorId;
  name: string;
  symbol: string;
  category: OperatorCategory;
  arity: number;
  mask: OperatorMask;
  inputPorts: InputPortDefinition[];
  meaning: string;
  latexTemplate: string;
  shapeRule: ShapeRule;
  algebraicProperties: string[];
  numericalNotes: string[];
  applicableRewriteRuleIds: string[];
  freedom: FreedomProfile;
  accent: string;
}
