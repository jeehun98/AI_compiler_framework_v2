import type { FreedomAxis, FreedomProfile } from '../domain/freedom';
import type { Graph, GraphEdge, GraphNode, InputPortId } from '../domain/graph';
import type { RewriteMatch, RewriteRule } from '../domain/rewrite';

const unboundImplementation: FreedomAxis = {
  status: 'unknown',
  summary: '의미 그래프의 변경이며 구체 실행 구현은 아직 선택되지 않았습니다.',
  constraints: ['향후 backend binding 결과를 별도로 비교해야 합니다.'],
};

function ruleFreedom(
  algebraic: FreedomAxis,
  numerical: FreedomAxis,
  structural: FreedomAxis,
): FreedomProfile {
  return { algebraic, numerical, structural, implementation: unboundImplementation };
}

const conditionalNumerical: FreedomAxis = {
  status: 'conditional',
  summary: '수학적 값은 보존하지만 IEEE 특수값 또는 target dtype 정책을 확인해야 합니다.',
  constraints: ['NaN, Inf, signed zero, signaling behavior'],
};

const exactNumerical: FreedomAxis = {
  status: 'available',
  summary: '값의 산술 재계산 없이 구조만 제거합니다.',
  constraints: [],
};

function incomingEdge(graph: Graph, nodeId: string, targetPort: InputPortId): GraphEdge | undefined {
  return graph.edges.find((edge) => edge.targetNodeId === nodeId && edge.targetPort === targetPort);
}

function nodeById(graph: Graph, nodeId: string | undefined): GraphNode | undefined {
  return nodeId ? graph.nodes.find(({ id }) => id === nodeId) : undefined;
}

function constantValue(node: GraphNode | undefined): number | undefined {
  if (!node || node.operatorId !== 'constant' || !('value' in node.parameters)) return undefined;
  return Number.isFinite(node.parameters.value) ? node.parameters.value : undefined;
}

function inputBindings(graph: Graph, rootNodeId: string): { left?: GraphNode; right?: GraphNode } {
  const left = nodeById(graph, incomingEdge(graph, rootNodeId, 'in-0')?.sourceNodeId);
  const right = nodeById(graph, incomingEdge(graph, rootNodeId, 'in-1')?.sourceNodeId);
  return { left, right };
}

function replaceOutput(graph: Graph, removedNodeId: string, replacementNodeId: string): string[] {
  return [...new Set(graph.outputs.map((nodeId) => nodeId === removedNodeId ? replacementNodeId : nodeId))];
}

function bypassRoot(graph: Graph, rootNodeId: string, replacementNodeId: string): Graph {
  const rootOutgoing = graph.edges.filter(({ sourceNodeId }) => sourceNodeId === rootNodeId);
  const retainedEdges = graph.edges.filter(({ sourceNodeId, targetNodeId }) => sourceNodeId !== rootNodeId && targetNodeId !== rootNodeId);
  const rewired = rootOutgoing.map((edge) => ({ ...edge, sourceNodeId: replacementNodeId }));
  return {
    ...graph,
    nodes: graph.nodes.filter(({ id }) => id !== rootNodeId),
    edges: [...retainedEdges, ...rewired],
    outputs: replaceOutput(graph, rootNodeId, replacementNodeId),
  };
}

function replaceWithConstant(graph: Graph, rootNodeId: string, value: number): Graph {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => node.id === rootNodeId ? { ...node, operatorId: 'constant', parameters: { value } } : node),
    edges: graph.edges.filter(({ targetNodeId }) => targetNodeId !== rootNodeId),
  };
}

function identityRule(
  id: 'add-zero' | 'mul-one',
  operatorId: 'add' | 'mul',
  identity: number,
  name: string,
): RewriteRule {
  return {
    id,
    name,
    exactness: 'conditionally-exact',
    description: `${operatorId === 'add' ? '덧셈' : '곱셈'} 항등원을 제거합니다.`,
    conditions: operatorId === 'add'
      ? ['signed zero가 관찰 가능한 의미가 아니어야 합니다.']
      : ['signaling NaN과 NaN payload 변화가 관찰 가능한 의미가 아니어야 합니다.'],
    freedom: ruleFreedom(
      { status: 'available', summary: '항등원 노드와 연산 노드를 제거합니다.', constraints: [] },
      conditionalNumerical,
      { status: 'available', summary: '두 노드를 우회해 소비자를 원래 값에 직접 연결합니다.', constraints: ['공유된 피연산자는 보존합니다.'] },
    ),
    findMatches(graph) {
      const matches: RewriteMatch[] = [];
      for (const root of graph.nodes.filter((node) => node.operatorId === operatorId)) {
        const { left, right } = inputBindings(graph, root.id);
        const leftIdentity = constantValue(left) === identity;
        const rightIdentity = constantValue(right) === identity;
        if (leftIdentity && right) {
          matches.push({ id: `${id}:${root.id}:left`, ruleId: id, rootNodeId: root.id, nodeIds: [root.id, left?.id ?? '', right.id], bindings: { value: right.id, identity: left?.id ?? '' }, summary: `${root.id}에서 왼쪽 항등원 ${identity} 제거` });
        } else if (rightIdentity && left) {
          matches.push({ id: `${id}:${root.id}:right`, ruleId: id, rootNodeId: root.id, nodeIds: [root.id, left.id, right?.id ?? ''], bindings: { value: left.id, identity: right?.id ?? '' }, summary: `${root.id}에서 오른쪽 항등원 ${identity} 제거` });
        }
      }
      return matches;
    },
    apply(graph, match) {
      return bypassRoot(graph, match.rootNodeId, match.bindings.value);
    },
  };
}

const addZeroRule = identityRule('add-zero', 'add', 0, 'x + 0 → x');
const mulOneRule = identityRule('mul-one', 'mul', 1, 'x × 1 → x');

const mulZeroRule: RewriteRule = {
  id: 'mul-zero',
  name: 'x × 0 → 0',
  exactness: 'conditionally-exact',
  description: '곱셈의 소거원 0으로 전체 결과를 대체합니다.',
  conditions: ['0이 아닌 피연산자가 유한해야 합니다.', 'NaN, Inf, signed zero 정책이 변형을 허용해야 합니다.'],
  freedom: ruleFreedom(
    { status: 'available', summary: '0 소거원으로 곱셈 전체를 대체합니다.', constraints: [] },
    conditionalNumerical,
    { status: 'available', summary: '사용되지 않는 반대쪽 피연산자 가지를 제거할 수 있습니다.', constraints: ['다른 소비자가 있는 공유 가지는 보존합니다.'] },
  ),
  findMatches(graph) {
    const matches: RewriteMatch[] = [];
    for (const root of graph.nodes.filter(({ operatorId }) => operatorId === 'mul')) {
      const { left, right } = inputBindings(graph, root.id);
      const zero = constantValue(left) === 0 ? left : constantValue(right) === 0 ? right : undefined;
      const discarded = zero?.id === left?.id ? right : left;
      if (zero && discarded) {
        matches.push({ id: `mul-zero:${root.id}`, ruleId: 'mul-zero', rootNodeId: root.id, nodeIds: [root.id, zero.id, discarded.id], bindings: { zero: zero.id, discarded: discarded.id }, summary: `${root.id}을 상수 0으로 대체` });
      }
    }
    return matches;
  },
  apply(graph, match) {
    return bypassRoot(graph, match.rootNodeId, match.bindings.zero);
  },
};

function evaluateConstant(operatorId: GraphNode['operatorId'], values: number[]): number | undefined {
  switch (operatorId) {
    case 'add': return values[0] + values[1];
    case 'mul': return values[0] * values[1];
    case 'relu': return Math.max(values[0], 0);
    default: return undefined;
  }
}

const constantFoldRule: RewriteRule = {
  id: 'constant-fold',
  name: 'Constant folding',
  exactness: 'conditionally-exact',
  description: '스칼라 Constant만을 입력으로 받는 Add, Mul, ReLU를 미리 계산합니다.',
  conditions: ['JavaScript number 평가가 target dtype의 반올림·특수값 의미와 같아야 합니다.'],
  freedom: ruleFreedom(
    { status: 'available', summary: '알려진 상수 식을 하나의 상수로 축약합니다.', constraints: [] },
    conditionalNumerical,
    { status: 'available', summary: '연산 노드와 전용 상수 입력을 단일 노드로 합칩니다.', constraints: ['공유 Constant는 제거하지 않습니다.'] },
  ),
  findMatches(graph) {
    const matches: RewriteMatch[] = [];
    for (const root of graph.nodes.filter(({ operatorId }) => ['add', 'mul', 'relu'].includes(operatorId))) {
      const arity = root.operatorId === 'relu' ? 1 : 2;
      const inputNodes = Array.from({ length: arity }, (_, index) => nodeById(graph, incomingEdge(graph, root.id, `in-${index}`)?.sourceNodeId));
      const values = inputNodes.map(constantValue);
      if (inputNodes.every(Boolean) && values.every((value) => value !== undefined)) {
        const result = evaluateConstant(root.operatorId, values as number[]);
        if (result !== undefined && Number.isFinite(result)) {
          const inputIds = inputNodes.map((node) => node?.id ?? '');
          matches.push({ id: `constant-fold:${root.id}`, ruleId: 'constant-fold', rootNodeId: root.id, nodeIds: [root.id, ...inputIds], bindings: { inputIds: inputIds.join(','), value: String(result) }, summary: `${root.id}을 상수 ${result}로 접기` });
        }
      }
    }
    return matches;
  },
  apply(graph, match) {
    return replaceWithConstant(graph, match.rootNodeId, Number(match.bindings.value));
  },
};

function commutativeRule(operatorId: 'add' | 'mul'): RewriteRule {
  const id = `${operatorId}-commute`;
  return {
    id,
    name: `${operatorId === 'add' ? 'Add' : 'Mul'} 입력 교환`,
    exactness: 'conditionally-exact',
    description: '교환법칙을 사용해 두 입력 포트를 바꿉니다.',
    conditions: ['NaN payload와 operand evaluation order가 관찰 가능한 의미가 아니어야 합니다.'],
    freedom: ruleFreedom(
      { status: 'available', summary: '교환법칙에 따라 입력 순서를 바꿉니다.', constraints: [] },
      conditionalNumerical,
      { status: 'available', summary: '노드 수를 바꾸지 않고 두 입력 엣지의 포트만 교환합니다.', constraints: [] },
    ),
    findMatches(graph) {
      return graph.nodes
        .filter((node) => node.operatorId === operatorId)
        .flatMap((root): RewriteMatch[] => {
          const left = incomingEdge(graph, root.id, 'in-0');
          const right = incomingEdge(graph, root.id, 'in-1');
          if (!left || !right || left.sourceNodeId === right.sourceNodeId) return [];
          return [{ id: `${id}:${root.id}`, ruleId: id, rootNodeId: root.id, nodeIds: [root.id, left.sourceNodeId, right.sourceNodeId], bindings: { leftEdge: left.id, rightEdge: right.id }, summary: `${root.id}의 두 입력 순서 교환` }];
        });
    },
    apply(graph, match) {
      return {
        ...graph,
        edges: graph.edges.map((edge) => {
          if (edge.id === match.bindings.leftEdge) return { ...edge, targetPort: 'in-1' };
          if (edge.id === match.bindings.rightEdge) return { ...edge, targetPort: 'in-0' };
          return edge;
        }),
      };
    },
  };
}

const doubleTransposeRule: RewriteRule = {
  id: 'double-transpose',
  name: 'Transpose(Transpose(x)) → x',
  exactness: 'exact',
  description: '동일한 마지막 두 축을 두 번 교환하는 연속 Transpose를 제거합니다.',
  conditions: ['두 Transpose가 모두 마지막 두 축 교환으로 정의되어야 합니다.'],
  freedom: ruleFreedom(
    { status: 'available', summary: 'Transpose의 대합 성질을 적용합니다.', constraints: [] },
    exactNumerical,
    { status: 'available', summary: '두 layout 노드를 우회합니다.', constraints: ['공유 inner Transpose는 보존합니다.'] },
  ),
  findMatches(graph) {
    const matches: RewriteMatch[] = [];
    for (const outer of graph.nodes.filter(({ operatorId }) => operatorId === 'transpose')) {
      const inner = nodeById(graph, incomingEdge(graph, outer.id, 'in-0')?.sourceNodeId);
      const value = inner?.operatorId === 'transpose' ? nodeById(graph, incomingEdge(graph, inner.id, 'in-0')?.sourceNodeId) : undefined;
      if (inner && value) {
        matches.push({ id: `double-transpose:${outer.id}`, ruleId: 'double-transpose', rootNodeId: outer.id, nodeIds: [outer.id, inner.id, value.id], bindings: { inner: inner.id, value: value.id }, summary: `${outer.id}과 ${inner.id} 제거` });
      }
    }
    return matches;
  },
  apply(graph, match) {
    const bypassed = bypassRoot(graph, match.rootNodeId, match.bindings.value);
    const innerId = match.bindings.inner;
    const innerIsStillUsed = bypassed.outputs.includes(innerId)
      || bypassed.edges.some(({ sourceNodeId }) => sourceNodeId === innerId);
    if (innerIsStillUsed) return bypassed;
    return {
      ...bypassed,
      nodes: bypassed.nodes.filter(({ id }) => id !== innerId),
      edges: bypassed.edges.filter(({ sourceNodeId, targetNodeId }) => sourceNodeId !== innerId && targetNodeId !== innerId),
    };
  },
};

export const REWRITE_RULES: readonly RewriteRule[] = [
  addZeroRule,
  mulOneRule,
  mulZeroRule,
  constantFoldRule,
  commutativeRule('add'),
  commutativeRule('mul'),
  doubleTransposeRule,
];

export const REWRITE_RULE_MAP: Readonly<Record<string, RewriteRule>> = Object.fromEntries(
  REWRITE_RULES.map((rule) => [rule.id, rule]),
);
