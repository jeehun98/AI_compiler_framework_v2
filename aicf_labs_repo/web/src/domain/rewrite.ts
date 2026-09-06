import type { FreedomProfile } from './freedom';
import type { Graph, GraphNode } from './graph';
import type { OperatorMask } from './operator';
import type {
  PropertyClaim,
  PropertyKind,
  PropertyScope,
  TransformationCapability,
} from './property';
import type {
  OperatorSemanticFacts,
  TransformationEvidence,
  TransformationFact,
} from './semantic';

export type RewriteExactness = 'exact' | 'conditionally-exact' | 'approximate';

export const SemanticDomain = {
  ABSTRACT_REAL: 'ABSTRACT_REAL',
  STRICT_IEEE: 'STRICT_IEEE',
  TOLERANCE_BASED: 'TOLERANCE_BASED',
} as const;

export type SemanticDomain = typeof SemanticDomain[keyof typeof SemanticDomain];

export const LegalityStatus = {
  APPLICABLE: 'APPLICABLE',
  REJECTED: 'REJECTED',
  UNKNOWN: 'UNKNOWN',
} as const;

export type LegalityStatus = typeof LegalityStatus[keyof typeof LegalityStatus];

export interface LegalityResult {
  status: LegalityStatus;
  reason: string;
  checkedConditions?: readonly string[];
  evidence?: readonly string[];
}

export type PropertyRequirementScope =
  | { kind: 'operator' }
  | { kind: 'input'; inputPortBinding: string }
  | { kind: 'output' };

export interface PropertyRequirement {
  kind: PropertyKind;
  operatorBinding: string;
  scope: PropertyRequirementScope;
  missingPropertyStatus?: typeof LegalityStatus.REJECTED | typeof LegalityStatus.UNKNOWN;
}

export interface ResolvedPropertyRequirement {
  operatorNodeId: string;
  operatorId: string;
  scope: PropertyScope;
  requirement: PropertyRequirement;
  claim: PropertyClaim;
}

export interface RuleEvaluationContext {
  semanticDomain: SemanticDomain;
  resolvedProperties: readonly ResolvedPropertyRequirement[];
  operatorSemanticFacts: readonly ResolvedOperatorSemanticFacts[];
}

export interface ResolvedOperatorSemanticFacts {
  binding: string;
  operatorNodeId: string;
  operatorId: string;
  facts: OperatorSemanticFacts;
}

export interface RewriteMatch {
  id: string;
  ruleId: string;
  rootNodeId: string;
  nodeIds: string[];
  bindings: Record<string, string>;
  summary: string;
  facts?: readonly TransformationFact[];
}

export interface RewriteRule {
  id: string;
  name: string;
  exactness: RewriteExactness;
  description: string;
  conditions: string[];
  freedom: FreedomProfile;
  requiredMask: OperatorMask;
  semanticDomain: SemanticDomain;
  requiredCapabilities: readonly TransformationCapability[];
  missingCapabilityStatus?: typeof LegalityStatus.REJECTED | typeof LegalityStatus.UNKNOWN;
  requiredProperties: readonly PropertyRequirement[];
  capabilityOperatorBindings?: readonly string[];
  justification: string;
  /** Finds graph structure only. Property and legality decisions are later phases. */
  matchStructure(graph: Graph, candidates: readonly GraphNode[]): RewriteMatch[];
  checkMathematicalLegality(
    graph: Graph,
    match: RewriteMatch,
    context: RuleEvaluationContext,
  ): LegalityResult;
  checkGraphLegality(
    graph: Graph,
    match: RewriteMatch,
    context: RuleEvaluationContext,
  ): LegalityResult;
  collectEvidence?(
    graph: Graph,
    match: RewriteMatch,
    context: RuleEvaluationContext,
    mathematicalLegality: LegalityResult,
    graphLegality: LegalityResult,
  ): TransformationEvidence;
  apply(graph: Graph, match: RewriteMatch): Graph;
}

export interface TransformationAttempt {
  id: string;
  sourceGraphId: string;
  ruleId: string;
  ruleName: string;
  matchId: string;
  bindings: Readonly<Record<string, string>>;
  graphFacts?: readonly TransformationFact[];
  requiredProperties: readonly string[];
  requiredCapabilities: readonly TransformationCapability[];
  semanticDomain: SemanticDomain;
  mathematicalLegality: LegalityResult;
  graphLegality: LegalityResult;
  evidence?: TransformationEvidence;
  status: LegalityStatus;
  targetGraphId?: string;
  reason?: string;
}

export interface RewriteCandidate {
  id: string;
  ruleId: string;
  ruleName: string;
  description: string;
  exactness: RewriteExactness;
  conditions: string[];
  summary: string;
  affectedNodeIds: string[];
  freedom: FreedomProfile;
  attemptId: string;
  targetGraphId: string;
  semanticDomain: SemanticDomain;
  justification: string;
  graphFacts?: readonly TransformationFact[];
  evidence?: TransformationEvidence;
  graph: Graph;
}
