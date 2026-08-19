import type { Graph } from '../domain/graph';
import type { RewriteCandidate, RewriteRule } from '../domain/rewrite';
import { REWRITE_RULES } from './rewriteRules';
import { validateGraph } from './validateGraph';

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

export function findRewriteCandidates(
  graph: Graph,
  rules: readonly RewriteRule[] = REWRITE_RULES,
): RewriteCandidate[] {
  if (!validateGraph(graph).valid) return [];
  const fingerprints = new Set<string>();
  return rules.flatMap((rule) => rule.findMatches(graph)
    .sort((left, right) => left.id.localeCompare(right.id))
    .flatMap((match): RewriteCandidate[] => {
      const candidateGraph = rule.apply(graph, match);
      if (!validateGraph(candidateGraph).valid) return [];
      const fingerprint = graphFingerprint(candidateGraph);
      if (fingerprints.has(fingerprint)) return [];
      fingerprints.add(fingerprint);
      return [{
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
      }];
    }));
}
