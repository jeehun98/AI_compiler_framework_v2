import type { FreedomProfile } from './freedom';
import type { Graph } from './graph';

export type RewriteExactness = 'exact' | 'conditionally-exact' | 'approximate';

export interface RewriteMatch {
  id: string;
  ruleId: string;
  rootNodeId: string;
  nodeIds: string[];
  bindings: Record<string, string>;
  summary: string;
}

export interface RewriteRule {
  id: string;
  name: string;
  exactness: RewriteExactness;
  description: string;
  conditions: string[];
  freedom: FreedomProfile;
  findMatches(graph: Graph): RewriteMatch[];
  apply(graph: Graph, match: RewriteMatch): Graph;
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
  graph: Graph;
}
