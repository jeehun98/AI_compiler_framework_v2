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
