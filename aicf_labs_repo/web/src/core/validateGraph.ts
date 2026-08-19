import { getOperator } from '../catalog/operators';
import type { Graph, GraphEdge, InputPortId } from '../domain/graph';
import type { ValidationIssue, ValidationResult } from '../domain/validation';
import { topologicalSort } from './topology';

function issue(
  code: ValidationIssue['code'],
  message: string,
  nodeIds: string[] = [],
  edgeIds: string[] = [],
): ValidationIssue {
  return { code, severity: 'error', message, nodeIds, edgeIds };
}

function duplicates(values: string[]): string[] {
  const seen = new Set<string>();
  const duplicateValues = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) duplicateValues.add(value);
    seen.add(value);
  }
  return [...duplicateValues].sort();
}

export function validateGraph(graph: Graph): ValidationResult {
  const issues: ValidationIssue[] = [];
  const duplicateNodeIds = duplicates(graph.nodes.map(({ id }) => id));
  const duplicateEdgeIds = duplicates(graph.edges.map(({ id }) => id));
  const outputs = Array.isArray(graph.outputs) ? graph.outputs : [];
  const duplicateOutputIds = duplicates(outputs);

  for (const id of duplicateNodeIds) {
    issues.push(issue('duplicate-node-id', `노드 ID '${id}'가 중복되었습니다.`, [id]));
  }
  for (const id of duplicateEdgeIds) {
    issues.push(issue('duplicate-edge-id', `엣지 ID '${id}'가 중복되었습니다.`, [], [id]));
  }

  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const validEdges: GraphEdge[] = [];

  if (outputs.length === 0) {
    issues.push(issue('missing-output', '그래프에 하나 이상의 명시적 output이 필요합니다.'));
  }
  for (const outputId of duplicateOutputIds) {
    issues.push(issue('duplicate-output', `output '${outputId}'가 중복되었습니다.`, [outputId]));
  }
  for (const outputId of outputs) {
    if (!nodes.has(outputId)) {
      issues.push(issue('invalid-output', `output '${outputId}'가 존재하지 않는 노드를 참조합니다.`, [outputId]));
    }
  }

  for (const node of graph.nodes) {
    if (!getOperator(node.operatorId)) {
      issues.push(issue('unknown-operator', `알 수 없는 연산자 '${String(node.operatorId)}'입니다.`, [node.id]));
    }
  }

  for (const edge of graph.edges) {
    const source = nodes.get(edge.sourceNodeId);
    const target = nodes.get(edge.targetNodeId);
    if (!source || !target) {
      issues.push(issue('dangling-edge', `엣지 '${edge.id}'가 존재하지 않는 노드를 참조합니다.`, [edge.sourceNodeId, edge.targetNodeId], [edge.id]));
      continue;
    }
    const operator = getOperator(target.operatorId);
    const validTargetPorts = operator?.inputPorts.map(({ id }) => id) ?? [];
    if (edge.sourcePort !== 'out' || !validTargetPorts.includes(edge.targetPort)) {
      issues.push(issue('invalid-port', `엣지 '${edge.id}'의 포트가 ${operator?.name ?? '연산자'} 계약과 맞지 않습니다.`, [source.id, target.id], [edge.id]));
      continue;
    }
    validEdges.push(edge);
  }

  for (const node of graph.nodes) {
    const operator = getOperator(node.operatorId);
    if (!operator) continue;
    const incoming = validEdges.filter(({ targetNodeId }) => targetNodeId === node.id);
    if (incoming.length > operator.arity) {
      issues.push(issue('invalid-arity', `${operator.name} 노드는 입력 ${operator.arity}개가 필요하지만 ${incoming.length}개가 연결되었습니다.`, [node.id], incoming.map(({ id }) => id)));
    }
    for (const port of operator.inputPorts) {
      const portEdges = incoming.filter(({ targetPort }) => targetPort === port.id);
      if (portEdges.length === 0) {
        issues.push(issue('missing-input', `${operator.name}의 필수 입력 '${port.label}'이 연결되지 않았습니다.`, [node.id]));
      } else if (portEdges.length > 1) {
        issues.push(issue('duplicate-input', `${operator.name}의 입력 '${port.label}'에 여러 엣지가 연결되었습니다.`, [node.id], portEdges.map(({ id }) => id)));
      }
    }
  }

  const topology = topologicalSort({ ...graph, edges: validEdges });
  if (topology.cycleNodeIds.length > 0) {
    const cycleEdges = validEdges.filter(
      ({ sourceNodeId, targetNodeId }) => topology.cycleNodeIds.includes(sourceNodeId) && topology.cycleNodeIds.includes(targetNodeId),
    );
    issues.push(issue('cycle', `순환 연결이 감지되었습니다: ${topology.cycleNodeIds.join(' → ')}`, topology.cycleNodeIds, cycleEdges.map(({ id }) => id)));
  }

  return {
    valid: issues.length === 0,
    issues,
    ...(topology.cycleNodeIds.length === 0 ? { topologicalOrder: topology.order } : {}),
  };
}

export interface ConnectRequest {
  sourceNodeId: string;
  targetNodeId: string;
  targetPort: InputPortId;
}

export interface ConnectCheck {
  allowed: boolean;
  reason?: string;
}

export function canConnect(graph: Graph, request: ConnectRequest): ConnectCheck {
  if (request.sourceNodeId === request.targetNodeId) {
    return { allowed: false, reason: '노드를 자기 자신에 연결할 수 없습니다.' };
  }
  const source = graph.nodes.find(({ id }) => id === request.sourceNodeId);
  const target = graph.nodes.find(({ id }) => id === request.targetNodeId);
  if (!source || !target) return { allowed: false, reason: '연결할 노드를 찾을 수 없습니다.' };
  const operator = getOperator(target.operatorId);
  if (!operator) return { allowed: false, reason: '대상 노드의 연산자를 확인할 수 없습니다.' };
  if (!operator.inputPorts.some(({ id }) => id === request.targetPort)) {
    return { allowed: false, reason: '대상 입력 포트가 유효하지 않습니다.' };
  }
  if (graph.edges.some((edge) => edge.targetNodeId === request.targetNodeId && edge.targetPort === request.targetPort)) {
    return { allowed: false, reason: '필수 입력 포트에는 하나의 엣지만 연결할 수 있습니다.' };
  }
  const trial: Graph = {
    ...graph,
    edges: [...graph.edges, {
      id: '__connection-check__', sourceNodeId: request.sourceNodeId, sourcePort: 'out',
      targetNodeId: request.targetNodeId, targetPort: request.targetPort,
    }],
  };
  if (topologicalSort(trial).cycleNodeIds.length > 0) {
    return { allowed: false, reason: '이 연결은 순환을 만듭니다.' };
  }
  return { allowed: true };
}
