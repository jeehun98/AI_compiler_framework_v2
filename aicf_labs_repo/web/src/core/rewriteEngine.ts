import { getOperator } from '../catalog/operators';
import type { Graph, GraphNode, InputPortId } from '../domain/graph';
import { matchesMask, type OperatorMask } from '../domain/operator';
import {
  deriveTransformationCapabilities,
  formatPropertyClaim,
  type PropertyScope,
  type TransformationCapability,
} from '../domain/property';
import {
  LegalityStatus,
  type LegalityResult,
  type PropertyRequirement,
  type ResolvedPropertyRequirement,
  type RewriteCandidate,
  type RewriteMatch,
  type RewriteRule,
  type TransformationAttempt,
} from '../domain/rewrite';
import { REWRITE_RULES } from './rewriteRules';
import { validateGraph } from './validateGraph';

export interface RewriteSearchOptions {
  useMask?: boolean;
}

export interface RuleSearchMetrics {
  ruleId: string;
  nodesScanned: number;
  maskAccepted: number;
  structureMatched: number;
  mathematicalAccepted: number;
  graphAccepted: number;
  conditionAccepted: number;
  rewritesApplied: number;
  maskPassRate: number;
  conditionPassRate: number;
  rewriteRate: number;
  candidateReduction: number;
}

export interface RewriteSearchResult {
  maskEnabled: boolean;
  candidates: RewriteCandidate[];
  attempts: TransformationAttempt[];
  metrics: RuleSearchMetrics[];
}

interface PropertyEvaluation {
  legality: LegalityResult;
  resolved: ResolvedPropertyRequirement[];
  requiredLabels: string[];
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, stableValue(item)]));
  }
  return value;
}

export function graphFingerprint(graph: Graph): string {
  return JSON.stringify({
    nodes: graph.nodes
      .map(({ id, operatorId, parameters }) => ({ id, operatorId, parameters: stableValue(parameters) }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    edges: graph.edges
      .map(({ id, sourceNodeId, sourcePort, targetNodeId, targetPort }) => ({ id, sourceNodeId, sourcePort, targetNodeId, targetPort }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    outputs: [...graph.outputs],
  });
}

export function nodesMatchingMask(graph: Graph, required: OperatorMask): GraphNode[] {
  return graph.nodes.filter((node) => {
    const operator = getOperator(node.operatorId);
    return operator !== undefined && matchesMask(operator.mask, required);
  });
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

function ruleMetrics(
  ruleId: string,
  nodesScanned: number,
  maskAccepted: number,
  structureMatched: number,
  mathematicalAccepted: number,
  graphAccepted: number,
  rewritesApplied: number,
): RuleSearchMetrics {
  return {
    ruleId,
    nodesScanned,
    maskAccepted,
    structureMatched,
    mathematicalAccepted,
    graphAccepted,
    conditionAccepted: graphAccepted,
    rewritesApplied,
    maskPassRate: ratio(maskAccepted, nodesScanned),
    conditionPassRate: ratio(graphAccepted, maskAccepted),
    rewriteRate: ratio(rewritesApplied, nodesScanned),
    candidateReduction: nodesScanned === 0 ? 0 : 1 - ratio(maskAccepted, nodesScanned),
  };
}

function resolveScope(requirement: PropertyRequirement, match: RewriteMatch): PropertyScope | undefined {
  switch (requirement.scope.kind) {
    case 'operator': return { kind: 'operator' };
    case 'output': return { kind: 'output', outputPort: 'out' };
    case 'input': {
      const inputPort = match.bindings[requirement.scope.inputPortBinding];
      return /^in-[0-9]+$/.test(inputPort ?? '')
        ? { kind: 'input', inputPort: inputPort as InputPortId }
        : undefined;
    }
  }
}

function scopesEqual(left: PropertyScope, right: PropertyScope): boolean {
  if (left.kind !== right.kind) return false;
  if (left.kind === 'input' && right.kind === 'input') return left.inputPort === right.inputPort;
  if (left.kind === 'output' && right.kind === 'output') return left.outputPort === right.outputPort;
  return left.kind === 'operator' && right.kind === 'operator';
}

function evaluatePropertiesAndCapabilities(
  graph: Graph,
  rule: RewriteRule,
  match: RewriteMatch,
): PropertyEvaluation {
  const resolved: ResolvedPropertyRequirement[] = [];
  const requiredLabels: string[] = [];
  const evidence: string[] = [];

  for (const requirement of rule.requiredProperties) {
    const operatorNodeId = match.bindings[requirement.operatorBinding];
    const operatorNode = graph.nodes.find(({ id }) => id === operatorNodeId);
    const scope = resolveScope(requirement, match);
    if (!operatorNode || !scope) {
      return {
        resolved,
        requiredLabels,
        legality: {
          status: LegalityStatus.UNKNOWN,
          reason: `Cannot resolve property binding '${requirement.operatorBinding}'.`,
          evidence,
        },
      };
    }
    const label = formatPropertyClaim({ kind: requirement.kind, scope });
    requiredLabels.push(label);
    const operator = getOperator(operatorNode.operatorId);
    if (!operator) {
      return {
        resolved,
        requiredLabels,
        legality: {
          status: LegalityStatus.UNKNOWN,
          reason: `Operator metadata for '${operatorNode.operatorId}' is unavailable.`,
          evidence,
        },
      };
    }
    const claim = operator.propertyClaims.find((candidate) =>
      candidate.kind === requirement.kind && scopesEqual(candidate.scope, scope));
    if (!claim) {
      return {
        resolved,
        requiredLabels,
        legality: {
          status: LegalityStatus.REJECTED,
          reason: `Required ${label} property is absent on '${operatorNode.operatorId}'.`,
          evidence,
        },
      };
    }
    resolved.push({ operatorNodeId, operatorId: operatorNode.operatorId, scope, requirement, claim });
    evidence.push(`${operatorNode.operatorId} declares ${label}${claim.justification ? `: ${claim.justification}` : '.'}`);
  }

  const derivedCapabilities = new Set<TransformationCapability>(
    deriveTransformationCapabilities(resolved.map(({ claim }) => claim)),
  );
  const missingCapability = rule.requiredCapabilities.find((capability) => !derivedCapabilities.has(capability));
  if (missingCapability) {
    return {
      resolved,
      requiredLabels,
      legality: {
        status: LegalityStatus.REJECTED,
        reason: `Required ${missingCapability} capability cannot be derived from the matched properties.`,
        evidence,
      },
    };
  }

  return {
    resolved,
    requiredLabels,
    legality: {
      status: LegalityStatus.APPLICABLE,
      reason: 'Required scoped properties and derived capabilities are present.',
      evidence,
    },
  };
}

function notEvaluated(reason: string): LegalityResult {
  return { status: LegalityStatus.UNKNOWN, reason };
}

function mergeMathematicalLegality(propertyResult: LegalityResult, ruleResult: LegalityResult): LegalityResult {
  return {
    ...ruleResult,
    evidence: [...(propertyResult.evidence ?? []), ...(ruleResult.evidence ?? [])],
  };
}

function attemptStatus(mathematical: LegalityResult, graph: LegalityResult): LegalityResult['status'] {
  if (mathematical.status === LegalityStatus.REJECTED || graph.status === LegalityStatus.REJECTED) {
    return LegalityStatus.REJECTED;
  }
  if (mathematical.status === LegalityStatus.UNKNOWN || graph.status === LegalityStatus.UNKNOWN) {
    return LegalityStatus.UNKNOWN;
  }
  return LegalityStatus.APPLICABLE;
}

function transformedGraphId(graph: Graph, rule: RewriteRule, match: RewriteMatch): string {
  return `${graph.id}::${rule.id}::${match.id}`;
}

export function searchRewriteCandidates(
  graph: Graph,
  rules: readonly RewriteRule[] = REWRITE_RULES,
  options: RewriteSearchOptions = {},
): RewriteSearchResult {
  const maskEnabled = options.useMask ?? true;
  if (!validateGraph(graph).valid) {
    return { maskEnabled, candidates: [], attempts: [], metrics: [] };
  }

  const candidates: RewriteCandidate[] = [];
  const attempts: TransformationAttempt[] = [];
  const metrics: RuleSearchMetrics[] = [];
  const fingerprints = new Set<string>();

  for (const rule of rules) {
    const nodesScanned = graph.nodes.length;
    const screenedNodes = maskEnabled ? nodesMatchingMask(graph, rule.requiredMask) : graph.nodes;
    const matches = rule.matchStructure(graph, screenedNodes)
      .sort((left, right) => left.id.localeCompare(right.id));
    let mathematicalAccepted = 0;
    let graphAccepted = 0;
    let rewritesApplied = 0;

    for (const match of matches) {
      const propertyEvaluation = evaluatePropertiesAndCapabilities(graph, rule, match);
      const context = {
        semanticDomain: rule.semanticDomain,
        resolvedProperties: propertyEvaluation.resolved,
      };
      let mathematicalLegality = propertyEvaluation.legality;
      if (mathematicalLegality.status === LegalityStatus.APPLICABLE) {
        mathematicalLegality = mergeMathematicalLegality(
          mathematicalLegality,
          rule.checkMathematicalLegality(graph, match, context),
        );
      }
      if (mathematicalLegality.status === LegalityStatus.APPLICABLE) mathematicalAccepted += 1;

      let graphLegality = mathematicalLegality.status === LegalityStatus.APPLICABLE
        ? rule.checkGraphLegality(graph, match, context)
        : notEvaluated('Graph legality was not evaluated because mathematical legality is not APPLICABLE.');
      let candidateGraph: Graph | undefined;
      if (graphLegality.status === LegalityStatus.APPLICABLE) {
        try {
          const rewritten = rule.apply(graph, match);
          const validation = validateGraph(rewritten);
          if (validation.valid) {
            candidateGraph = rewritten;
            graphAccepted += 1;
          } else {
            graphLegality = {
              status: LegalityStatus.REJECTED,
              reason: `Rewritten graph failed validation: ${validation.issues.map(({ code }) => code).join(', ')}.`,
            };
          }
        } catch (error) {
          graphLegality = {
            status: LegalityStatus.REJECTED,
            reason: `Rewrite failed: ${error instanceof Error ? error.message : String(error)}.`,
          };
        }
      }

      const status = attemptStatus(mathematicalLegality, graphLegality);
      const targetGraphId = status === LegalityStatus.APPLICABLE
        ? transformedGraphId(graph, rule, match)
        : undefined;
      const attempt: TransformationAttempt = {
        id: `${rule.id}:${match.id}`,
        sourceGraphId: graph.id,
        ruleId: rule.id,
        ruleName: rule.name,
        matchId: match.id,
        bindings: { ...match.bindings },
        requiredProperties: propertyEvaluation.requiredLabels,
        requiredCapabilities: [...rule.requiredCapabilities],
        semanticDomain: rule.semanticDomain,
        mathematicalLegality,
        graphLegality,
        status,
        ...(targetGraphId ? { targetGraphId } : {}),
        ...(status === LegalityStatus.APPLICABLE
          ? {}
          : { reason: mathematicalLegality.status !== LegalityStatus.APPLICABLE ? mathematicalLegality.reason : graphLegality.reason }),
      };
      attempts.push(attempt);

      if (!candidateGraph || !targetGraphId) continue;
      const fingerprint = graphFingerprint(candidateGraph);
      if (fingerprints.has(fingerprint)) continue;
      fingerprints.add(fingerprint);
      rewritesApplied += 1;
      candidates.push({
        id: match.id,
        ruleId: rule.id,
        ruleName: rule.name,
        description: rule.description,
        exactness: rule.exactness,
        conditions: [...rule.conditions],
        summary: match.summary,
        affectedNodeIds: [...new Set(match.nodeIds.filter(Boolean))],
        freedom: rule.freedom,
        attemptId: attempt.id,
        targetGraphId,
        semanticDomain: rule.semanticDomain,
        justification: rule.justification,
        graph: candidateGraph,
      });
    }

    metrics.push(ruleMetrics(
      rule.id,
      nodesScanned,
      screenedNodes.length,
      matches.length,
      mathematicalAccepted,
      graphAccepted,
      rewritesApplied,
    ));
  }

  return { maskEnabled, candidates, attempts, metrics };
}

export function findRewriteCandidates(
  graph: Graph,
  rules: readonly RewriteRule[] = REWRITE_RULES,
): RewriteCandidate[] {
  return searchRewriteCandidates(graph, rules).candidates;
}

export function findTransformationAttempts(
  graph: Graph,
  rules: readonly RewriteRule[] = REWRITE_RULES,
): TransformationAttempt[] {
  return searchRewriteCandidates(graph, rules).attempts;
}
