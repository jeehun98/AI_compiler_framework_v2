import { getOperator } from '../catalog/operators';
import type { Graph, GraphNode } from '../domain/graph';
import { matchesMask, type OperatorMask } from '../domain/operator';
import type { RewriteCandidate, RewriteRule } from '../domain/rewrite';
import { REWRITE_RULES } from './rewriteRules';
import { validateGraph } from './validateGraph';

export interface RewriteSearchOptions {
  useMask?: boolean;
}

export interface RuleSearchMetrics {
  ruleId: string;
  nodesScanned: number;
  maskAccepted: number;
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
  metrics: RuleSearchMetrics[];
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
  conditionAccepted: number,
  rewritesApplied: number,
): RuleSearchMetrics {
  return {
    ruleId,
    nodesScanned,
    maskAccepted,
    conditionAccepted,
    rewritesApplied,
    maskPassRate: ratio(maskAccepted, nodesScanned),
    conditionPassRate: ratio(conditionAccepted, maskAccepted),
    rewriteRate: ratio(rewritesApplied, nodesScanned),
    candidateReduction: nodesScanned === 0 ? 0 : 1 - ratio(maskAccepted, nodesScanned),
  };
}

export function searchRewriteCandidates(
  graph: Graph,
  rules: readonly RewriteRule[] = REWRITE_RULES,
  options: RewriteSearchOptions = {},
): RewriteSearchResult {
  const maskEnabled = options.useMask ?? true;
  if (!validateGraph(graph).valid) {
    return { maskEnabled, candidates: [], metrics: [] };
  }

  const candidates: RewriteCandidate[] = [];
  const metrics: RuleSearchMetrics[] = [];
  const fingerprints = new Set<string>();

  for (const rule of rules) {
    const nodesScanned = graph.nodes.length;
    // This remains a permissive superset; findMatches owns detailed legality.
    const screenedNodes = maskEnabled
      ? nodesMatchingMask(graph, rule.requiredMask)
      : graph.nodes;
    const matches = rule.findMatches(graph, screenedNodes)
      .sort((left, right) => left.id.localeCompare(right.id));
    let rewritesApplied = 0;

    for (const match of matches) {
      const candidateGraph = rule.apply(graph, match);
      if (!validateGraph(candidateGraph).valid) continue;
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
        graph: candidateGraph,
      });
    }

    metrics.push(ruleMetrics(
      rule.id,
      nodesScanned,
      screenedNodes.length,
      matches.length,
      rewritesApplied,
    ));
  }

  return { maskEnabled, candidates, metrics };
}

export function findRewriteCandidates(
  graph: Graph,
  rules: readonly RewriteRule[] = REWRITE_RULES,
): RewriteCandidate[] {
  return searchRewriteCandidates(graph, rules).candidates;
}
