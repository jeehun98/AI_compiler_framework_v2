import type { FreedomAxis, FreedomProfile } from '../domain/freedom';
import { getOperator } from '../catalog/operators';
import type { Graph, GraphEdge, GraphNode, InputPortId } from '../domain/graph';
import { OperatorMask } from '../domain/operator';
import { PropertyKind, TransformationCapability } from '../domain/property';
import {
  LegalityStatus,
  SemanticDomain,
  type LegalityResult,
  type PropertyRequirement,
  type RewriteMatch,
  type RewriteRule,
  type RuleEvaluationContext,
} from '../domain/rewrite';
import {
  getConsumerCount,
  getConsumers,
  getIncomingEdge,
  inferNodeShape,
  isGraphOutput,
} from './graphFacts';
import { validateGraph } from './validateGraph';

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

function nodeById(graph: Graph, nodeId: string | undefined): GraphNode | undefined {
  return nodeId ? graph.nodes.find(({ id }) => id === nodeId) : undefined;
}

function constantValue(node: GraphNode | undefined): number | undefined {
  if (!node || node.operatorId !== 'constant' || !('value' in node.parameters)) return undefined;
  return Number.isFinite(node.parameters.value) ? node.parameters.value : undefined;
}

function inputBindings(graph: Graph, rootNodeId: string): { left?: GraphNode; right?: GraphNode } {
  const left = nodeById(graph, getIncomingEdge(graph, rootNodeId, 'in-0')?.sourceNodeId);
  const right = nodeById(graph, getIncomingEdge(graph, rootNodeId, 'in-1')?.sourceNodeId);
  return { left, right };
}

function applicable(reason: string, checkedConditions: readonly string[] = []): LegalityResult {
  return { status: LegalityStatus.APPLICABLE, reason, checkedConditions };
}

function wholeProperty(kind: PropertyRequirement['kind']): PropertyRequirement {
  return { kind, operatorBinding: 'operator', scope: { kind: 'operator' } };
}

function standardReasoning(
  requiredProperties: readonly PropertyRequirement[],
  justification: string,
): Pick<
  RewriteRule,
  | 'semanticDomain'
  | 'requiredCapabilities'
  | 'requiredProperties'
  | 'justification'
  | 'checkMathematicalLegality'
  | 'checkGraphLegality'
> {
  return {
    semanticDomain: SemanticDomain.ABSTRACT_REAL,
    requiredCapabilities: [],
    requiredProperties,
    justification,
    checkMathematicalLegality() {
      return applicable('The declared algebraic relation is valid over abstract real-number semantics.');
    },
    checkGraphLegality() {
      return applicable('The existing immutable rewrite preserves graph references; final DAG validation remains authoritative.');
    },
  };
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
    requiredMask: OperatorMask.ELEMENTWISE | OperatorMask.PURE,
    ...standardReasoning(
      [wholeProperty(PropertyKind.ELEMENTWISE), wholeProperty(PropertyKind.PURE)],
      `${operatorId}의 항등원 성질을 사용합니다.`,
    ),
    conditions: operatorId === 'add'
      ? ['signed zero가 관찰 가능한 의미가 아니어야 합니다.']
      : ['signaling NaN과 NaN payload 변화가 관찰 가능한 의미가 아니어야 합니다.'],
    freedom: ruleFreedom(
      { status: 'available', summary: '항등원 노드와 연산 노드를 제거합니다.', constraints: [] },
      conditionalNumerical,
      { status: 'available', summary: '두 노드를 우회해 소비자를 원래 값에 직접 연결합니다.', constraints: ['공유된 피연산자는 보존합니다.'] },
    ),
    matchStructure(graph, candidates) {
      const matches: RewriteMatch[] = [];
      for (const root of candidates.filter((node) => node.operatorId === operatorId)) {
        const { left, right } = inputBindings(graph, root.id);
        const leftIdentity = constantValue(left) === identity;
        const rightIdentity = constantValue(right) === identity;
        if (leftIdentity && right) {
          matches.push({ id: `${id}:${root.id}:left`, ruleId: id, rootNodeId: root.id, nodeIds: [root.id, left?.id ?? '', right.id], bindings: { operator: root.id, value: right.id, identity: left?.id ?? '' }, summary: `${root.id}에서 왼쪽 항등원 ${identity} 제거` });
        } else if (rightIdentity && left) {
          matches.push({ id: `${id}:${root.id}:right`, ruleId: id, rootNodeId: root.id, nodeIds: [root.id, left.id, right?.id ?? ''], bindings: { operator: root.id, value: left.id, identity: right?.id ?? '' }, summary: `${root.id}에서 오른쪽 항등원 ${identity} 제거` });
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
  requiredMask: OperatorMask.ELEMENTWISE | OperatorMask.PURE,
  ...standardReasoning(
    [wholeProperty(PropertyKind.ELEMENTWISE), wholeProperty(PropertyKind.PURE)],
    '0 is the multiplicative annihilator over abstract real-number semantics.',
  ),
  conditions: ['0이 아닌 피연산자가 유한해야 합니다.', 'NaN, Inf, signed zero 정책이 변형을 허용해야 합니다.'],
  freedom: ruleFreedom(
    { status: 'available', summary: '0 소거원으로 곱셈 전체를 대체합니다.', constraints: [] },
    conditionalNumerical,
    { status: 'available', summary: '사용되지 않는 반대쪽 피연산자 가지를 제거할 수 있습니다.', constraints: ['다른 소비자가 있는 공유 가지는 보존합니다.'] },
  ),
  matchStructure(graph, candidates) {
    const matches: RewriteMatch[] = [];
    for (const root of candidates.filter(({ operatorId }) => operatorId === 'mul')) {
      const { left, right } = inputBindings(graph, root.id);
      const zero = constantValue(left) === 0 ? left : constantValue(right) === 0 ? right : undefined;
      const discarded = zero?.id === left?.id ? right : left;
      if (zero && discarded) {
        matches.push({ id: `mul-zero:${root.id}`, ruleId: 'mul-zero', rootNodeId: root.id, nodeIds: [root.id, zero.id, discarded.id], bindings: { operator: root.id, zero: zero.id, discarded: discarded.id }, summary: `${root.id}을 상수 0으로 대체` });
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
  requiredMask: OperatorMask.ELEMENTWISE | OperatorMask.PURE,
  ...standardReasoning(
    [wholeProperty(PropertyKind.ELEMENTWISE), wholeProperty(PropertyKind.PURE)],
    'A closed finite scalar expression can be evaluated without graph inputs.',
  ),
  conditions: ['JavaScript number 평가가 target dtype의 반올림·특수값 의미와 같아야 합니다.'],
  freedom: ruleFreedom(
    { status: 'available', summary: '알려진 상수 식을 하나의 상수로 축약합니다.', constraints: [] },
    conditionalNumerical,
    { status: 'available', summary: '연산 노드와 전용 상수 입력을 단일 노드로 합칩니다.', constraints: ['공유 Constant는 제거하지 않습니다.'] },
  ),
  matchStructure(graph, candidates) {
    const matches: RewriteMatch[] = [];
    for (const root of candidates.filter(({ operatorId }) => ['add', 'mul', 'relu'].includes(operatorId))) {
      const arity = root.operatorId === 'relu' ? 1 : 2;
      const inputNodes = Array.from({ length: arity }, (_, index) => nodeById(graph, getIncomingEdge(graph, root.id, `in-${index}`)?.sourceNodeId));
      const values = inputNodes.map(constantValue);
      if (inputNodes.every(Boolean) && values.every((value) => value !== undefined)) {
        const result = evaluateConstant(root.operatorId, values as number[]);
        if (result !== undefined && Number.isFinite(result)) {
          const inputIds = inputNodes.map((node) => node?.id ?? '');
          matches.push({ id: `constant-fold:${root.id}`, ruleId: 'constant-fold', rootNodeId: root.id, nodeIds: [root.id, ...inputIds], bindings: { operator: root.id, inputIds: inputIds.join(','), value: String(result) }, summary: `${root.id}을 상수 ${result}로 접기` });
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
    requiredMask: OperatorMask.COMMUTATIVE | OperatorMask.PURE,
    ...standardReasoning(
      [wholeProperty(PropertyKind.COMMUTATIVE), wholeProperty(PropertyKind.PURE)],
      `${operatorId} declares commutativity over abstract real-number semantics.`,
    ),
    conditions: ['NaN payload와 operand evaluation order가 관찰 가능한 의미가 아니어야 합니다.'],
    freedom: ruleFreedom(
      { status: 'available', summary: '교환법칙에 따라 입력 순서를 바꿉니다.', constraints: [] },
      conditionalNumerical,
      { status: 'available', summary: '노드 수를 바꾸지 않고 두 입력 엣지의 포트만 교환합니다.', constraints: [] },
    ),
    matchStructure(graph, candidates) {
      return candidates
        .filter((node) => node.operatorId === operatorId)
        .flatMap((root): RewriteMatch[] => {
          const left = getIncomingEdge(graph, root.id, 'in-0');
          const right = getIncomingEdge(graph, root.id, 'in-1');
          if (!left || !right || left.sourceNodeId === right.sourceNodeId) return [];
          return [{ id: `${id}:${root.id}`, ruleId: id, rootNodeId: root.id, nodeIds: [root.id, left.sourceNodeId, right.sourceNodeId], bindings: { operator: root.id, leftEdge: left.id, rightEdge: right.id }, summary: `${root.id}의 두 입력 순서 교환` }];
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
  requiredMask: OperatorMask.PERMUTATION | OperatorMask.PURE,
  ...standardReasoning(
    [wholeProperty(PropertyKind.PERMUTATION), wholeProperty(PropertyKind.PURE)],
    'Applying the same final-two-axis permutation twice is the identity.',
  ),
  conditions: ['두 Transpose가 모두 마지막 두 축 교환으로 정의되어야 합니다.'],
  freedom: ruleFreedom(
    { status: 'available', summary: 'Transpose의 대합 성질을 적용합니다.', constraints: [] },
    exactNumerical,
    { status: 'available', summary: '두 layout 노드를 우회합니다.', constraints: ['공유 inner Transpose는 보존합니다.'] },
  ),
  matchStructure(graph, candidates) {
    const matches: RewriteMatch[] = [];
    for (const outer of candidates.filter(({ operatorId }) => operatorId === 'transpose')) {
      const inner = nodeById(graph, getIncomingEdge(graph, outer.id, 'in-0')?.sourceNodeId);
      const value = inner?.operatorId === 'transpose' ? nodeById(graph, getIncomingEdge(graph, inner.id, 'in-0')?.sourceNodeId) : undefined;
      if (inner && value) {
        matches.push({ id: `double-transpose:${outer.id}`, ruleId: 'double-transpose', rootNodeId: outer.id, nodeIds: [outer.id, inner.id, value.id], bindings: { operator: outer.id, inner: inner.id, value: value.id }, summary: `${outer.id}과 ${inner.id} 제거` });
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

function scaleStructureMatches(
  graph: Graph,
  candidates: readonly GraphNode[],
  ruleId: string,
): RewriteMatch[] {
  const matches: RewriteMatch[] = [];
  for (const operator of candidates) {
    const inputEdges = graph.edges
      .filter(({ targetNodeId }) => targetNodeId === operator.id)
      .sort((left, right) => left.targetPort.localeCompare(right.targetPort));
    for (const scaleToOperator of inputEdges) {
      const scale = nodeById(graph, scaleToOperator.sourceNodeId);
      if (scale?.operatorId !== 'mul') continue;
      const leftEdge = getIncomingEdge(graph, scale.id, 'in-0');
      const rightEdge = getIncomingEdge(graph, scale.id, 'in-1');
      if (!leftEdge || !rightEdge) continue;
      const leftAlpha = constantValue(nodeById(graph, leftEdge.sourceNodeId));
      const rightAlpha = constantValue(nodeById(graph, rightEdge.sourceNodeId));
      const alphaEdge = rightAlpha !== undefined ? rightEdge : leftAlpha !== undefined ? leftEdge : undefined;
      const alpha = rightAlpha !== undefined ? rightAlpha : leftAlpha;
      if (!alphaEdge || alpha === undefined) continue;
      const valueEdge = alphaEdge.id === leftEdge.id ? rightEdge : leftEdge;
      matches.push({
        id: `${ruleId}:${operator.id}:${scaleToOperator.targetPort}:${scale.id}`,
        ruleId,
        rootNodeId: operator.id,
        nodeIds: [scale.id, operator.id, valueEdge.sourceNodeId, alphaEdge.sourceNodeId],
        bindings: {
          operator: operator.id,
          scale: scale.id,
          value: valueEdge.sourceNodeId,
          alpha: alphaEdge.sourceNodeId,
          alphaValue: String(alpha),
          targetInputPort: scaleToOperator.targetPort,
          scaleValueEdge: valueEdge.id,
          alphaEdge: alphaEdge.id,
          scaleTargetEdge: scaleToOperator.id,
        },
        summary: `${scale.id}을 ${operator.id}의 ${scaleToOperator.targetPort} 앞에서 뒤로 이동`,
      });
    }
  }
  return matches;
}

function checkLinearScaleMathematics(
  graph: Graph,
  match: RewriteMatch,
  context: RuleEvaluationContext,
): LegalityResult {
  const linear = context.resolvedProperties.find(({ claim }) => claim.kind === PropertyKind.LINEAR);
  if (!linear) {
    return { status: LegalityStatus.UNKNOWN, reason: 'The scoped LINEAR claim could not be resolved.' };
  }
  const checkedConditions: string[] = ['semantic domain is ABSTRACT_REAL'];
  for (const condition of linear.claim.conditions) {
    if (condition.kind !== 'input-fixed') continue;
    const fixedInput = getIncomingEdge(graph, match.bindings.operator, condition.inputPort);
    if (!fixedInput) {
      return {
        status: LegalityStatus.UNKNOWN,
        reason: `Cannot confirm fixed input ${condition.inputPort}.`,
        checkedConditions,
      };
    }
    checkedConditions.push(`${condition.inputPort} remains connected to ${fixedInput.sourceNodeId}`);
  }
  return applicable(
    'Scale propagation follows from the scoped LINEAR property over abstract real-number semantics.',
    checkedConditions,
  );
}

function checkPositiveScaleMathematics(
  _graph: Graph,
  match: RewriteMatch,
  _context: RuleEvaluationContext,
): LegalityResult {
  const alpha = Number(match.bindings.alphaValue);
  if (!Number.isFinite(alpha)) {
    return { status: LegalityStatus.UNKNOWN, reason: 'The scale value is not a known finite scalar.' };
  }
  if (alpha < 0) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'Positive homogeneity requires alpha >= 0.',
      checkedConditions: [`alpha = ${alpha}`, 'alpha >= 0 is false'],
    };
  }
  return applicable(
    'Positive homogeneity permits moving a non-negative scale through the operator.',
    [`alpha = ${alpha}`, 'alpha >= 0 is true', 'semantic domain is ABSTRACT_REAL'],
  );
}

function checkScaleGraphLegality(
  graph: Graph,
  match: RewriteMatch,
  context: RuleEvaluationContext,
): LegalityResult {
  const scaleId = match.bindings.scale;
  const operatorId = match.bindings.operator;
  const consumers = getConsumers(graph, scaleId);
  const outgoingUses = graph.edges.filter(({ sourceNodeId }) => sourceNodeId === scaleId);
  if (getConsumerCount(graph, scaleId) !== 1 || consumers[0]?.id !== operatorId || outgoingUses.length !== 1) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'Scale propagation requires exactly one consumer and one outgoing use, both targeting the matched operator.',
      checkedConditions: [`consumer count = ${getConsumerCount(graph, scaleId)}`, `outgoing use count = ${outgoingUses.length}`],
    };
  }
  if (isGraphOutput(graph, scaleId)) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'The scale node is an explicit graph output and cannot be moved.',
      checkedConditions: ['scale is an explicit graph output'],
    };
  }
  const scopedClaim = context.resolvedProperties.find(({ scope }) =>
    scope.kind === 'input' && scope.inputPort === match.bindings.targetInputPort);
  if (!scopedClaim) {
    return {
      status: LegalityStatus.UNKNOWN,
      reason: 'The matched target input port cannot be tied to a scoped property claim.',
    };
  }
  const scaleShape = inferNodeShape(graph, scaleId);
  const operatorShape = inferNodeShape(graph, operatorId);
  if (scaleShape === undefined || operatorShape === undefined) {
    return {
      status: LegalityStatus.UNKNOWN,
      reason: 'Shape information is insufficient to confirm the moved scale and target output shapes.',
      checkedConditions: [
        `scale shape = ${scaleShape === undefined ? 'unknown' : JSON.stringify(scaleShape)}`,
        `operator shape = ${operatorShape === undefined ? 'unknown' : JSON.stringify(operatorShape)}`,
      ],
    };
  }
  return applicable(
    'The scale has one use, is not a graph output, matches the claimed input port, and all required shapes are known.',
    [
      'scale consumer count = 1',
      'scale is not an explicit graph output',
      `property scope matches ${match.bindings.targetInputPort}`,
      `scale shape = ${JSON.stringify(scaleShape)}`,
      `operator output shape = ${JSON.stringify(operatorShape)}`,
      'dtype is abstracted by the ABSTRACT_REAL semantic domain',
    ],
  );
}

function moveScaleAfterOperator(graph: Graph, match: RewriteMatch): Graph {
  const scaleValueEdgeId = match.bindings.scaleValueEdge;
  const scaleTargetEdgeId = match.bindings.scaleTargetEdge;
  const scaleId = match.bindings.scale;
  const operatorId = match.bindings.operator;
  const valueId = match.bindings.value;
  const targetInputPort = match.bindings.targetInputPort as InputPortId;
  if (!graph.edges.some(({ id }) => id === scaleValueEdgeId)
    || !graph.edges.some(({ id }) => id === scaleTargetEdgeId)) {
    throw new Error('Scale rewrite bindings do not reference the matched graph edges');
  }
  return {
    ...graph,
    edges: graph.edges.map((edge): GraphEdge => {
      if (edge.id === scaleValueEdgeId) {
        return { ...edge, sourceNodeId: operatorId };
      }
      if (edge.id === scaleTargetEdgeId) {
        return { ...edge, sourceNodeId: valueId, targetNodeId: operatorId, targetPort: targetInputPort };
      }
      if (edge.sourceNodeId === operatorId) {
        return { ...edge, sourceNodeId: scaleId };
      }
      return edge;
    }),
    outputs: graph.outputs.map((nodeId) => nodeId === operatorId ? scaleId : nodeId),
  };
}

const scopedInputLinearity: PropertyRequirement = {
  kind: PropertyKind.LINEAR,
  operatorBinding: 'operator',
  scope: { kind: 'input', inputPortBinding: 'targetInputPort' },
};

const scopedInputPositiveHomogeneity: PropertyRequirement = {
  kind: PropertyKind.POSITIVE_HOMOGENEOUS,
  operatorBinding: 'operator',
  scope: { kind: 'input', inputPortBinding: 'targetInputPort' },
};

const scaleThroughLinearRule: RewriteRule = {
  id: 'scale-through-linear',
  name: 'ScaleThroughLinear',
  exactness: 'exact',
  description: 'LINEAR input property를 근거로 scalar scale을 operator 뒤로 이동합니다.',
  conditions: ['alpha is a finite scalar', 'the non-scaled inputs remain fixed', 'semantic domain is ABSTRACT_REAL'],
  freedom: ruleFreedom(
    { status: 'available', summary: 'Scoped linearity permits scale propagation.', constraints: ['other inputs remain fixed'] },
    { status: 'available', summary: 'The MVP reasons over abstract real numbers.', constraints: ['not a strict IEEE bitwise claim'] },
    { status: 'conditional', summary: 'The scale node moves after its sole consumer.', constraints: ['known shapes', 'single scale use', 'scale is not an output'] },
  ),
  requiredMask: OperatorMask.PURE,
  semanticDomain: SemanticDomain.ABSTRACT_REAL,
  requiredCapabilities: [TransformationCapability.SCALE_PROPAGATION],
  requiredProperties: [scopedInputLinearity, wholeProperty(PropertyKind.PURE)],
  justification: 'A linear map F satisfies F(alpha × x) = alpha × F(x) while its other inputs are fixed.',
  matchStructure(graph, candidates) {
    return scaleStructureMatches(graph, candidates, 'scale-through-linear');
  },
  checkMathematicalLegality: checkLinearScaleMathematics,
  checkGraphLegality: checkScaleGraphLegality,
  apply: moveScaleAfterOperator,
};

const scaleThroughPositiveHomogeneousRule: RewriteRule = {
  id: 'scale-through-positive-homogeneous',
  name: 'ScaleThroughPositiveHomogeneous',
  exactness: 'exact',
  description: 'POSITIVE_HOMOGENEOUS input property를 근거로 non-negative scale을 operator 뒤로 이동합니다.',
  conditions: ['alpha is a finite scalar', 'alpha >= 0', 'semantic domain is ABSTRACT_REAL'],
  freedom: ruleFreedom(
    { status: 'conditional', summary: 'Positive homogeneity permits non-negative scale propagation.', constraints: ['alpha >= 0'] },
    { status: 'available', summary: 'The MVP reasons over abstract real numbers.', constraints: ['not a strict IEEE bitwise claim'] },
    { status: 'conditional', summary: 'The scale node moves after its sole consumer.', constraints: ['known shapes', 'single scale use', 'scale is not an output'] },
  ),
  requiredMask: OperatorMask.PURE,
  semanticDomain: SemanticDomain.ABSTRACT_REAL,
  requiredCapabilities: [TransformationCapability.POSITIVE_SCALE_PROPAGATION],
  requiredProperties: [scopedInputPositiveHomogeneity, wholeProperty(PropertyKind.PURE)],
  justification: 'A positively homogeneous map F satisfies F(alpha × x) = alpha × F(x) for alpha >= 0.',
  matchStructure(graph, candidates) {
    return scaleStructureMatches(graph, candidates, 'scale-through-positive-homogeneous');
  },
  checkMathematicalLegality: checkPositiveScaleMathematics,
  checkGraphLegality: checkScaleGraphLegality,
  apply: moveScaleAfterOperator,
};

function associativeStructureMatches(
  graph: Graph,
  candidates: readonly GraphNode[],
): RewriteMatch[] {
  const matches: RewriteMatch[] = [];
  for (const outer of candidates) {
    if (getOperator(outer.operatorId)?.arity !== 2) continue;
    const innerToOuter = getIncomingEdge(graph, outer.id, 'in-0');
    const outerRight = getIncomingEdge(graph, outer.id, 'in-1');
    const inner = nodeById(graph, innerToOuter?.sourceNodeId);
    if (!innerToOuter || !outerRight || !inner || inner.operatorId !== outer.operatorId) continue;
    if (getOperator(inner.operatorId)?.arity !== 2) continue;
    const innerLeft = getIncomingEdge(graph, inner.id, 'in-0');
    const innerRight = getIncomingEdge(graph, inner.id, 'in-1');
    if (!innerLeft || !innerRight) continue;
    matches.push({
      id: `reassociate-associative-right:${outer.id}:${inner.id}`,
      ruleId: 'reassociate-associative-right',
      rootNodeId: outer.id,
      nodeIds: [
        outer.id,
        inner.id,
        innerLeft.sourceNodeId,
        innerRight.sourceNodeId,
        outerRight.sourceNodeId,
      ],
      bindings: {
        operator: outer.id,
        outer: outer.id,
        inner: inner.id,
        a: innerLeft.sourceNodeId,
        b: innerRight.sourceNodeId,
        c: outerRight.sourceNodeId,
        innerLeftEdge: innerLeft.id,
        innerRightEdge: innerRight.id,
        innerToOuterEdge: innerToOuter.id,
        outerRightEdge: outerRight.id,
      },
      summary: `${outer.operatorId}((${innerLeft.sourceNodeId}, ${innerRight.sourceNodeId}), ${outerRight.sourceNodeId})을 오른쪽 결합형으로 정규화`,
    });
  }
  return matches;
}

function reassociateAssociativeRight(graph: Graph, match: RewriteMatch): Graph {
  const {
    innerLeftEdge,
    innerRightEdge,
    innerToOuterEdge,
    outerRightEdge,
    inner,
    outer,
  } = match.bindings;
  const requiredEdgeIds = [innerLeftEdge, innerRightEdge, innerToOuterEdge, outerRightEdge];
  if (!requiredEdgeIds.every((edgeId) => graph.edges.some(({ id }) => id === edgeId))) {
    throw new Error('Reassociation bindings do not reference all matched graph edges');
  }
  return {
    ...graph,
    edges: graph.edges.map((edge): GraphEdge => {
      if (edge.id === innerLeftEdge) {
        return { ...edge, targetNodeId: outer, targetPort: 'in-0' };
      }
      if (edge.id === innerRightEdge) {
        return { ...edge, targetNodeId: inner, targetPort: 'in-0' };
      }
      if (edge.id === outerRightEdge) {
        return { ...edge, targetNodeId: inner, targetPort: 'in-1' };
      }
      if (edge.id === innerToOuterEdge) {
        return { ...edge, sourceNodeId: inner, targetNodeId: outer, targetPort: 'in-1' };
      }
      return edge;
    }),
  };
}

function checkAssociativeMathematics(
  _graph: Graph,
  _match: RewriteMatch,
  context: RuleEvaluationContext,
): LegalityResult {
  const associative = context.resolvedProperties.find(({ claim }) => claim.kind === PropertyKind.ASSOCIATIVE);
  if (!associative) {
    return { status: LegalityStatus.UNKNOWN, reason: 'The ASSOCIATIVE property could not be resolved.' };
  }
  return applicable(
    'The operator declares associativity under abstract semantics.',
    ['ASSOCIATIVE(operator) is declared', 'semantic domain is ABSTRACT_REAL'],
  );
}

function checkAssociativeGraphLegality(
  graph: Graph,
  match: RewriteMatch,
  context: RuleEvaluationContext,
): LegalityResult {
  const innerId = match.bindings.inner;
  const outerId = match.bindings.outer;
  const inner = nodeById(graph, innerId);
  const outer = nodeById(graph, outerId);
  const consumers = getConsumers(graph, innerId);
  const outgoingUses = graph.edges.filter(({ sourceNodeId }) => sourceNodeId === innerId);

  if (getConsumerCount(graph, innerId) > 1) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'The inner result has multiple consumers.',
      checkedConditions: [
        `inner consumer count = ${getConsumerCount(graph, innerId)}`,
        `inner outgoing use count = ${outgoingUses.length}`,
      ],
    };
  }
  if (getConsumerCount(graph, innerId) !== 1 || consumers[0]?.id !== outerId || outgoingUses.length !== 1) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'Reassociation requires exactly one inner-result use targeting the matched outer operator.',
      checkedConditions: [
        `inner consumer count = ${getConsumerCount(graph, innerId)}`,
        `inner outgoing use count = ${outgoingUses.length}`,
      ],
    };
  }
  if (isGraphOutput(graph, innerId)) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'The inner result is an explicit graph output and cannot be reassociated.',
      checkedConditions: ['inner is an explicit graph output'],
    };
  }
  const pure = context.resolvedProperties.some(({ claim }) => claim.kind === PropertyKind.PURE);
  if (!inner || !outer || inner.operatorId !== outer.operatorId || !pure) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'Both matched nodes must be instances of the same pure operator.',
      checkedConditions: [
        `same operator identity = ${Boolean(inner && outer && inner.operatorId === outer.operatorId)}`,
        `PURE(operator) is declared = ${pure}`,
      ],
    };
  }
  if (JSON.stringify(inner.parameters) !== JSON.stringify(outer.parameters)) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'Associative operator instances with different attributes cannot be safely reassociated.',
    };
  }

  const originalShape = inferNodeShape(graph, outerId);
  const reassociated = reassociateAssociativeRight(graph, match);
  const reassociatedShape = inferNodeShape(reassociated, outerId);
  const reassociatedValidation = validateGraph(reassociated);
  if (!reassociatedValidation.valid) {
    return {
      status: LegalityStatus.REJECTED,
      reason: `The reassociated graph would be invalid: ${reassociatedValidation.issues.map(({ code }) => code).join(', ')}.`,
    };
  }
  if (originalShape === undefined || reassociatedShape === undefined) {
    return {
      status: LegalityStatus.UNKNOWN,
      reason: 'Shape information is insufficient to confirm the reassociated output shape.',
      checkedConditions: [
        `original output shape = ${originalShape === undefined ? 'unknown' : JSON.stringify(originalShape)}`,
        `reassociated output shape = ${reassociatedShape === undefined ? 'unknown' : JSON.stringify(reassociatedShape)}`,
      ],
    };
  }
  if (JSON.stringify(originalShape) !== JSON.stringify(reassociatedShape)) {
    return {
      status: LegalityStatus.REJECTED,
      reason: 'Reassociation would change the inferred output shape.',
      checkedConditions: [
        `original output shape = ${JSON.stringify(originalShape)}`,
        `reassociated output shape = ${JSON.stringify(reassociatedShape)}`,
      ],
    };
  }
  return applicable(
    'The inner node has one consumer, is not a graph output, and the rewrite preserves the DAG and output shape.',
    [
      'inner consumer count = 1',
      'inner is not an explicit graph output',
      `operator identity = ${outer.operatorId}`,
      'PURE(operator) is declared',
      'rewritten graph passes DAG/arity/port/output validation',
      `output shape = ${JSON.stringify(originalShape)}`,
    ],
  );
}

const reassociateAssociativeRightRule: RewriteRule = {
  id: 'reassociate-associative-right',
  name: 'ReassociateAssociativeRight',
  exactness: 'exact',
  description: 'ASSOCIATIVE property를 근거로 Op(Op(a, b), c)를 Op(a, Op(b, c))로 정규화합니다.',
  conditions: [
    'only the left-associated to right-associated canonical direction is generated',
    'the inner result has one outgoing use and is not a graph output',
    'semantic domain is ABSTRACT_REAL',
  ],
  freedom: ruleFreedom(
    { status: 'available', summary: 'Declared associativity permits one-way reassociation.', constraints: ['ABSTRACT_REAL only'] },
    { status: 'available', summary: 'The MVP makes an abstract-real equality claim.', constraints: ['not a strict IEEE bitwise claim'] },
    { status: 'conditional', summary: 'Existing inner and outer nodes are rewired without changing node identity.', constraints: ['same operator', 'private inner result', 'shape preservation'] },
  ),
  requiredMask: OperatorMask.PURE,
  semanticDomain: SemanticDomain.ABSTRACT_REAL,
  requiredCapabilities: [TransformationCapability.REASSOCIATION],
  requiredProperties: [wholeProperty(PropertyKind.ASSOCIATIVE), wholeProperty(PropertyKind.PURE)],
  justification: 'For an associative operator, Op(Op(a, b), c) = Op(a, Op(b, c)); the rule emits only the right-associated canonical form.',
  matchStructure: associativeStructureMatches,
  checkMathematicalLegality: checkAssociativeMathematics,
  checkGraphLegality: checkAssociativeGraphLegality,
  apply: reassociateAssociativeRight,
};

export const SCALE_PROPAGATION_RULES: readonly RewriteRule[] = [
  scaleThroughLinearRule,
  scaleThroughPositiveHomogeneousRule,
];

export const REASSOCIATION_RULES: readonly RewriteRule[] = [
  reassociateAssociativeRightRule,
];

export const REWRITE_RULES: readonly RewriteRule[] = [
  addZeroRule,
  mulOneRule,
  mulZeroRule,
  constantFoldRule,
  commutativeRule('add'),
  commutativeRule('mul'),
  doubleTransposeRule,
  ...SCALE_PROPAGATION_RULES,
  ...REASSOCIATION_RULES,
];

export const REWRITE_RULE_MAP: Readonly<Record<string, RewriteRule>> = Object.fromEntries(
  REWRITE_RULES.map((rule) => [rule.id, rule]),
);
