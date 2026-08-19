import type { FreedomProfile } from '../domain/freedom';
import type { Operator, OperatorId } from '../domain/operator';

const implementationUnbound = {
  status: 'unknown',
  summary: '브라우저 MVP는 특정 실행 구현에 결합되지 않습니다.',
  constraints: ['향후 backend가 dtype과 implementation variant를 제공해야 합니다.'],
} as const;

function freedom(
  algebraic: FreedomProfile['algebraic'],
  numerical: FreedomProfile['numerical'],
  structural: FreedomProfile['structural'],
): FreedomProfile {
  return { algebraic, numerical, structural, implementation: implementationUnbound };
}

const fixedAlgebra = { status: 'fixed', summary: '소스 노드 자체에는 대수적 재작성 자유도가 없습니다.', constraints: [] } as const;
const valueDependent = { status: 'conditional', summary: '구체 dtype과 특수값 정책에 따라 달라집니다.', constraints: ['NaN, Inf, signed zero를 고려해야 합니다.'] } as const;
const sourceStructure = { status: 'fixed', summary: '입력이 없는 그래프 경계 노드입니다.', constraints: [] } as const;

export const OPERATORS: readonly Operator[] = [
  {
    id: 'input', name: 'Input', symbol: 'x', category: 'source', arity: 0, inputPorts: [],
    meaning: '계산 그래프에 외부 텐서를 도입합니다.', latexTemplate: 'x',
    shapeRule: { notation: '사용자가 지정한 shape', constraints: ['shape는 의미 주석이며 MVP에서 실행하지 않습니다.'] },
    algebraicProperties: ['그래프의 자유 변수'], numericalNotes: ['dtype과 값 범위는 외부 입력 계약에 따릅니다.'], applicableRewriteRuleIds: [],
    freedom: freedom(fixedAlgebra, valueDependent, sourceStructure), accent: '#67e8f9',
  },
  {
    id: 'constant', name: 'Constant', symbol: 'c', category: 'source', arity: 0, inputPorts: [],
    meaning: '그래프 내부의 스칼라 상수입니다.', latexTemplate: 'c',
    shapeRule: { notation: 'scalar', constraints: ['MVP Constant는 유한한 JavaScript number입니다.'] },
    algebraicProperties: ['항등원·소거원 패턴에 참여'], numericalNotes: ['target dtype으로 변환될 때 반올림될 수 있습니다.'], applicableRewriteRuleIds: ['constant-fold'],
    freedom: freedom({ status: 'available', summary: '값에 따라 항등원·소거원 규칙에 참여합니다.', constraints: [] }, valueDependent, sourceStructure), accent: '#fbbf24',
  },
  {
    id: 'add', name: 'Add', symbol: '+', category: 'elementwise', arity: 2,
    inputPorts: [{ id: 'in-0', label: 'left', required: true }, { id: 'in-1', label: 'right', required: true }],
    meaning: '두 값을 원소별로 더합니다.', latexTemplate: 'a + b',
    shapeRule: { notation: 'broadcast(A, B)', constraints: ['두 입력 shape는 broadcast 가능해야 합니다.'] },
    algebraicProperties: ['교환법칙', '0은 항등원'], numericalNotes: ['부동소수점 덧셈은 signed zero와 NaN payload에 민감합니다.'],
    applicableRewriteRuleIds: ['add-zero', 'add-commute', 'constant-fold'],
    freedom: freedom({ status: 'available', summary: '입력 교환과 0 제거가 가능합니다.', constraints: [] }, valueDependent, { status: 'conditional', summary: '두 고정 입력 포트를 유지해야 합니다.', constraints: ['broadcast 계약을 유지해야 합니다.'] }), accent: '#fb7185',
  },
  {
    id: 'mul', name: 'Mul', symbol: '×', category: 'elementwise', arity: 2,
    inputPorts: [{ id: 'in-0', label: 'left', required: true }, { id: 'in-1', label: 'right', required: true }],
    meaning: '두 값을 원소별로 곱합니다.', latexTemplate: 'a \\cdot b',
    shapeRule: { notation: 'broadcast(A, B)', constraints: ['두 입력 shape는 broadcast 가능해야 합니다.'] },
    algebraicProperties: ['교환법칙', '1은 항등원', '0은 소거원'], numericalNotes: ['0 곱 제거는 NaN과 Inf에서 동치가 아닐 수 있습니다.'],
    applicableRewriteRuleIds: ['mul-one', 'mul-zero', 'mul-commute', 'constant-fold'],
    freedom: freedom({ status: 'available', summary: '입력 교환, 1 제거, 0 소거가 가능합니다.', constraints: [] }, valueDependent, { status: 'conditional', summary: '두 고정 입력 포트를 유지해야 합니다.', constraints: ['broadcast 계약을 유지해야 합니다.'] }), accent: '#c084fc',
  },
  {
    id: 'matmul', name: 'MatMul', symbol: '@', category: 'matrix', arity: 2,
    inputPorts: [{ id: 'in-0', label: 'left', required: true }, { id: 'in-1', label: 'right', required: true }],
    meaning: '두 행렬의 축약 곱을 계산합니다.', latexTemplate: 'AB',
    shapeRule: { notation: '[m, k] @ [k, n] → [m, n]', constraints: ['왼쪽 마지막 축과 오른쪽 끝에서 두 번째 축이 같아야 합니다.'] },
    algebraicProperties: ['일반적으로 비가환', '수학적으로 결합법칙 성립'], numericalNotes: ['축약 순서와 누적 precision에 민감합니다.'], applicableRewriteRuleIds: [],
    freedom: freedom({ status: 'fixed', summary: 'MVP에는 MatMul 재결합 규칙이 없습니다.', constraints: [] }, valueDependent, { status: 'fixed', summary: '축약 축과 입력 순서가 고정됩니다.', constraints: ['[m,k]와 [k,n] 계약'] }), accent: '#60a5fa',
  },
  {
    id: 'relu', name: 'ReLU', symbol: '↗', category: 'elementwise', arity: 1,
    inputPorts: [{ id: 'in-0', label: 'input', required: true }], meaning: '각 원소를 max(x, 0)으로 변환합니다.', latexTemplate: '\\operatorname{ReLU}(x)',
    shapeRule: { notation: 'shape(Y) = shape(X)', constraints: ['rank와 shape를 보존합니다.'] },
    algebraicProperties: ['멱등: ReLU(ReLU(x)) = ReLU(x)'], numericalNotes: ['NaN 처리 방식은 구체 max 구현에 따라 달라질 수 있습니다.'], applicableRewriteRuleIds: ['constant-fold'],
    freedom: freedom({ status: 'conditional', summary: '멱등성이 있으나 MVP rewrite에는 포함하지 않습니다.', constraints: [] }, valueDependent, { status: 'fixed', summary: '단항 shape-preserving 연산입니다.', constraints: [] }), accent: '#34d399',
  },
  {
    id: 'transpose', name: 'Transpose', symbol: 'T', category: 'layout', arity: 1,
    inputPorts: [{ id: 'in-0', label: 'input', required: true }], meaning: '마지막 두 축을 서로 교환합니다.', latexTemplate: 'x^{\\mathsf T}',
    shapeRule: { notation: '[…, m, n] → […, n, m]', constraints: ['rank가 2 이상이어야 합니다.'] },
    algebraicProperties: ['대합: T(T(x)) = x'], numericalNotes: ['값을 재계산하지 않지만 구현에서 layout materialization이 필요할 수 있습니다.'], applicableRewriteRuleIds: ['double-transpose'],
    freedom: freedom({ status: 'available', summary: '연속 두 Transpose를 제거할 수 있습니다.', constraints: [] }, { status: 'available', summary: '순수한 축 교환은 값의 반올림을 일으키지 않습니다.', constraints: [] }, { status: 'available', summary: '연속 축 교환 구조를 제거할 수 있습니다.', constraints: ['동일한 마지막 두 축 Transpose 정의'] }), accent: '#f472b6',
  },
  {
    id: 'reduceSum', name: 'ReduceSum', symbol: 'Σ', category: 'reduction', arity: 1,
    inputPorts: [{ id: 'in-0', label: 'input', required: true }], meaning: '선택한 축의 원소를 합산합니다.', latexTemplate: '\\sum_i x_i',
    shapeRule: { notation: '지정 축 제거 또는 크기 1 유지', constraints: ['axis는 입력 rank 범위 안이어야 합니다.'] },
    algebraicProperties: ['선형 연산'], numericalNotes: ['합산 순서에 따라 부동소수점 결과가 달라질 수 있습니다.'], applicableRewriteRuleIds: [],
    freedom: freedom({ status: 'fixed', summary: 'MVP에는 reduction 재결합 규칙이 없습니다.', constraints: [] }, valueDependent, { status: 'fixed', summary: 'axis와 keepDims가 출력 구조를 결정합니다.', constraints: [] }), accent: '#a3e635',
  },
] as const;

export const OPERATOR_MAP: Readonly<Record<OperatorId, Operator>> = Object.fromEntries(
  OPERATORS.map((operator) => [operator.id, operator]),
) as Record<OperatorId, Operator>;

export function getOperator(operatorId: unknown): Operator | undefined {
  if (typeof operatorId !== 'string' || !Object.hasOwn(OPERATOR_MAP, operatorId)) return undefined;
  return OPERATOR_MAP[operatorId as OperatorId];
}
