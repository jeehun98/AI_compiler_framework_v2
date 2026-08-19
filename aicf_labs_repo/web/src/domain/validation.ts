export type ValidationIssueCode =
  | 'missing-input'
  | 'invalid-arity'
  | 'duplicate-input'
  | 'cycle'
  | 'dangling-edge'
  | 'unknown-operator'
  | 'invalid-port'
  | 'duplicate-node-id'
  | 'duplicate-edge-id'
  | 'missing-output'
  | 'invalid-output'
  | 'duplicate-output';

export interface ValidationIssue {
  code: ValidationIssueCode;
  severity: 'error' | 'warning';
  message: string;
  nodeIds: string[];
  edgeIds: string[];
}

export interface ValidationResult {
  valid: boolean;
  issues: ValidationIssue[];
  topologicalOrder?: string[];
}
