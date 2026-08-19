import { getOperator } from '../catalog/operators';
import type {
  GraphDocument,
  GraphEdge,
  GraphNode,
  GraphPosition,
  GraphViewport,
  NodeParameters,
} from '../domain/graph';
import type { OperatorId } from '../domain/operator';

export type DocumentParseResult =
  | { ok: true; value: GraphDocument }
  | { ok: false; error: string };

type ValueResult<T> = { ok: true; value: T } | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function duplicateValue(values: string[]): string | undefined {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) return value;
    seen.add(value);
  }
  return undefined;
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: readonly string[]): boolean {
  const allowedKeys = new Set(allowed);
  return Object.keys(value).every((key) => allowedKeys.has(key));
}

function parseParameters(operatorId: OperatorId, value: unknown): ValueResult<NodeParameters> {
  if (!isRecord(value)) return { ok: false, error: `${operatorId} parameters는 객체여야 합니다.` };
  switch (operatorId) {
    case 'input': {
      if (!hasOnlyKeys(value, ['symbol', 'shape'])) return { ok: false, error: 'Input.parameters에 알 수 없는 필드가 있습니다.' };
      if (!isNonEmptyString(value.symbol)) return { ok: false, error: 'Input.symbol은 비어 있지 않은 문자열이어야 합니다.' };
      if (!Array.isArray(value.shape) || !value.shape.every((dimension) =>
        (typeof dimension === 'string' && dimension.trim().length > 0)
        || (Number.isInteger(dimension) && (dimension as number) >= 0))) {
        return { ok: false, error: 'Input.shape은 음이 아닌 정수 또는 비어 있지 않은 기호의 배열이어야 합니다.' };
      }
      return { ok: true, value: { symbol: value.symbol, shape: [...value.shape] as Array<number | string> } };
    }
    case 'constant':
      if (!hasOnlyKeys(value, ['value'])) return { ok: false, error: 'Constant.parameters에 알 수 없는 필드가 있습니다.' };
      return isFiniteNumber(value.value)
        ? { ok: true, value: { value: value.value } }
        : { ok: false, error: 'Constant.value는 유한한 숫자여야 합니다.' };
    case 'reduceSum': {
      if (!hasOnlyKeys(value, ['axis', 'keepDims'])) return { ok: false, error: 'ReduceSum.parameters에 알 수 없는 필드가 있습니다.' };
      const validAxis = value.axis === 'all' || (Number.isInteger(value.axis) && (value.axis as number) >= 0);
      if (!validAxis) return { ok: false, error: "ReduceSum.axis는 'all' 또는 음이 아닌 정수여야 합니다." };
      if (typeof value.keepDims !== 'boolean') return { ok: false, error: 'ReduceSum.keepDims는 boolean이어야 합니다.' };
      return { ok: true, value: { axis: value.axis as number | 'all', keepDims: value.keepDims } };
    }
    default:
      return Object.keys(value).length === 0
        ? { ok: true, value: {} }
        : { ok: false, error: `${operatorId} parameters에는 추가 필드가 허용되지 않습니다.` };
  }
}

function parseNode(value: unknown, index: number): ValueResult<GraphNode> {
  if (!isRecord(value)) return { ok: false, error: `nodes[${index}]는 객체여야 합니다.` };
  if (!isNonEmptyString(value.id)) return { ok: false, error: `nodes[${index}].id는 비어 있지 않은 문자열이어야 합니다.` };
  const operator = getOperator(value.operatorId);
  if (!operator) return { ok: false, error: `알 수 없는 연산자 '${String(value.operatorId)}'입니다.` };
  const parameters = parseParameters(operator.id, value.parameters);
  if (!parameters.ok) return parameters;
  return { ok: true, value: { id: value.id, operatorId: operator.id, parameters: parameters.value } };
}

function parseEdge(
  value: unknown,
  index: number,
  nodes: ReadonlyMap<string, GraphNode>,
): ValueResult<GraphEdge> {
  if (!isRecord(value)) return { ok: false, error: `edges[${index}]는 객체여야 합니다.` };
  if (!isNonEmptyString(value.id)) return { ok: false, error: `edges[${index}].id는 비어 있지 않은 문자열이어야 합니다.` };
  if (!isNonEmptyString(value.sourceNodeId) || !nodes.has(value.sourceNodeId)) {
    return { ok: false, error: `edges[${index}]의 source node '${String(value.sourceNodeId)}'가 존재하지 않습니다.` };
  }
  if (!isNonEmptyString(value.targetNodeId) || !nodes.has(value.targetNodeId)) {
    return { ok: false, error: `edges[${index}]의 target node '${String(value.targetNodeId)}'가 존재하지 않습니다.` };
  }
  if (value.sourcePort !== 'out') return { ok: false, error: `edges[${index}].sourcePort는 'out'이어야 합니다.` };
  if (typeof value.targetPort !== 'string') return { ok: false, error: `edges[${index}].targetPort가 필요합니다.` };
  const target = nodes.get(value.targetNodeId);
  const operator = getOperator(target?.operatorId);
  const port = operator?.inputPorts.find(({ id }) => id === value.targetPort);
  if (!port) return { ok: false, error: `edges[${index}]의 target port '${value.targetPort}'가 유효하지 않습니다.` };
  return {
    ok: true,
    value: {
      id: value.id,
      sourceNodeId: value.sourceNodeId,
      sourcePort: 'out',
      targetNodeId: value.targetNodeId,
      targetPort: port.id,
    },
  };
}

function parsePosition(value: unknown, nodeId: string): ValueResult<GraphPosition> {
  if (!isRecord(value) || !isFiniteNumber(value.x) || !isFiniteNumber(value.y)) {
    return { ok: false, error: `노드 '${nodeId}'의 position은 유한한 x, y를 가져야 합니다.` };
  }
  return { ok: true, value: { x: value.x, y: value.y } };
}

function parseViewport(value: unknown): ValueResult<GraphViewport | undefined> {
  if (value === undefined) return { ok: true, value: undefined };
  if (!isRecord(value) || !isFiniteNumber(value.x) || !isFiniteNumber(value.y) || !isFiniteNumber(value.zoom) || value.zoom <= 0) {
    return { ok: false, error: 'viewport는 유한한 x, y와 양수 zoom을 가져야 합니다.' };
  }
  return { ok: true, value: { x: value.x, y: value.y, zoom: value.zoom } };
}

export function serializeGraphDocument(document: GraphDocument): string {
  return JSON.stringify(document, null, 2);
}

export function parseGraphDocument(json: string): DocumentParseResult {
  let root: unknown;
  try {
    root = JSON.parse(json);
  } catch {
    return { ok: false, error: '올바른 JSON 문서가 아닙니다.' };
  }
  if (!isRecord(root)) return { ok: false, error: 'JSON 루트는 객체여야 합니다.' };
  if (root.schemaVersion !== 1) return { ok: false, error: '지원하지 않는 schemaVersion입니다. 현재 버전은 1입니다.' };
  if (!isRecord(root.graph)) return { ok: false, error: 'graph 객체가 필요합니다.' };
  const graphValue = root.graph;
  if (!isNonEmptyString(graphValue.id) || !isNonEmptyString(graphValue.name)) {
    return { ok: false, error: 'graph.id와 graph.name은 비어 있지 않은 문자열이어야 합니다.' };
  }
  if (!Array.isArray(graphValue.nodes) || !Array.isArray(graphValue.edges)) {
    return { ok: false, error: 'graph.nodes와 graph.edges는 배열이어야 합니다.' };
  }

  const nodes: GraphNode[] = [];
  for (let index = 0; index < graphValue.nodes.length; index += 1) {
    const parsed = parseNode(graphValue.nodes[index], index);
    if (!parsed.ok) return parsed;
    nodes.push(parsed.value);
  }
  const duplicateNodeId = duplicateValue(nodes.map(({ id }) => id));
  if (duplicateNodeId) return { ok: false, error: `노드 ID '${duplicateNodeId}'가 중복되었습니다.` };
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));

  if (!Array.isArray(graphValue.outputs) || graphValue.outputs.length === 0 || !graphValue.outputs.every(isNonEmptyString)) {
    return { ok: false, error: 'graph.outputs는 하나 이상의 비어 있지 않은 node ID 배열이어야 합니다.' };
  }
  const outputs = [...graphValue.outputs];
  const duplicateOutputId = duplicateValue(outputs);
  if (duplicateOutputId) return { ok: false, error: `output '${duplicateOutputId}'가 중복되었습니다.` };
  const missingOutput = outputs.find((nodeId) => !nodeMap.has(nodeId));
  if (missingOutput) return { ok: false, error: `output '${missingOutput}'가 존재하지 않는 노드를 참조합니다.` };

  const edges: GraphEdge[] = [];
  for (let index = 0; index < graphValue.edges.length; index += 1) {
    const parsed = parseEdge(graphValue.edges[index], index, nodeMap);
    if (!parsed.ok) return parsed;
    edges.push(parsed.value);
  }
  const duplicateEdgeId = duplicateValue(edges.map(({ id }) => id));
  if (duplicateEdgeId) return { ok: false, error: `엣지 ID '${duplicateEdgeId}'가 중복되었습니다.` };
  const occupiedPorts = new Set<string>();
  for (const edge of edges) {
    const key = `${edge.targetNodeId}\u0000${edge.targetPort}`;
    if (occupiedPorts.has(key)) return { ok: false, error: `노드 '${edge.targetNodeId}'의 '${edge.targetPort}'에 여러 엣지가 연결되었습니다.` };
    occupiedPorts.add(key);
  }

  if (!isRecord(root.layout) || !isRecord(root.layout.positions)) {
    return { ok: false, error: 'layout.positions 객체가 필요합니다.' };
  }
  const positions: Record<string, GraphPosition> = {};
  for (const node of nodes) {
    const parsed = parsePosition(root.layout.positions[node.id], node.id);
    if (!parsed.ok) return parsed;
    positions[node.id] = parsed.value;
  }
  const extraPosition = Object.keys(root.layout.positions).find((nodeId) => !nodeMap.has(nodeId));
  if (extraPosition) return { ok: false, error: `layout.positions가 존재하지 않는 노드 '${extraPosition}'를 참조합니다.` };
  const viewport = parseViewport(root.layout.viewport);
  if (!viewport.ok) return viewport;

  return {
    ok: true,
    value: {
      schemaVersion: 1,
      graph: { id: graphValue.id, name: graphValue.name, nodes, edges, outputs },
      layout: { positions, ...(viewport.value ? { viewport: viewport.value } : {}) },
    },
  };
}
