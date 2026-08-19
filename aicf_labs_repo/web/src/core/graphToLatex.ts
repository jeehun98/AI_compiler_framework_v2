import { getOperator } from '../catalog/operators';
import type { Graph, GraphNode, ReduceSumParameters } from '../domain/graph';
import type { ValidationIssue } from '../domain/validation';
import { validateGraph } from './validateGraph';

export interface GeneratedExpression {
  nodeId: string;
  label: string;
  latex: string;
}

export interface ExpressionResult {
  ok: boolean;
  expressions: GeneratedExpression[];
  combinedLatex: string;
  issues: ValidationIssue[];
}

function inputSymbol(node: GraphNode): string {
  const symbol = 'symbol' in node.parameters && typeof node.parameters.symbol === 'string'
    ? node.parameters.symbol.trim()
    : 'x';
  if (/^[A-Za-z]$/.test(symbol)) return symbol;
  if (/^[A-Za-z][A-Za-z0-9]*$/.test(symbol)) return `\\mathrm{${symbol}}`;
  return '\\mathrm{input}';
}

function constantValue(node: GraphNode): string {
  const value = 'value' in node.parameters && typeof node.parameters.value === 'number'
    ? node.parameters.value
    : 0;
  if (Object.is(value, -0)) return '-0';
  return String(value);
}

function expressionFor(
  graph: Graph,
  node: GraphNode,
  memo: Map<string, string>,
  active: Set<string>,
): string {
  const cached = memo.get(node.id);
  if (cached) return cached;
  if (active.has(node.id)) return '\\operatorname{cycle}';
  active.add(node.id);
  const operator = getOperator(node.operatorId);
  if (!operator) return '\\operatorname{unknown}';
  const inputs = operator.inputPorts.map((port) => {
    const edge = graph.edges.find(({ targetNodeId, targetPort }) => targetNodeId === node.id && targetPort === port.id);
    const source = edge ? graph.nodes.find(({ id }) => id === edge.sourceNodeId) : undefined;
    return source ? expressionFor(graph, source, memo, active) : '\\color{#fb7185}{\\square}';
  });

  let latex: string;
  switch (node.operatorId) {
    case 'input': latex = inputSymbol(node); break;
    case 'constant': latex = constantValue(node); break;
    case 'add': latex = `\\left(${inputs[0]} + ${inputs[1]}\\right)`; break;
    case 'mul': latex = `\\left(${inputs[0]} \\cdot ${inputs[1]}\\right)`; break;
    case 'matmul': latex = `\\left(${inputs[0]}\\,${inputs[1]}\\right)`; break;
    case 'relu': latex = `\\operatorname{ReLU}\\left(${inputs[0]}\\right)`; break;
    case 'transpose': latex = `\\left(${inputs[0]}\\right)^{\\mathsf T}`; break;
    case 'reduceSum': {
      const parameters = node.parameters as ReduceSumParameters;
      const subscript = parameters.axis === 'all' ? '' : `_{i_{${parameters.axis}}}`;
      latex = `\\sum${subscript} \\left(${inputs[0]}\\right)`;
      break;
    }
  }

  active.delete(node.id);
  memo.set(node.id, latex);
  return latex;
}

export function graphToLatex(graph: Graph): ExpressionResult {
  const validation = validateGraph(graph);
  if (!validation.valid) {
    return { ok: false, expressions: [], combinedLatex: '\\operatorname{invalid\\ DAG}', issues: validation.issues };
  }

  const memo = new Map<string, string>();
  const expressions = graph.outputs.map((nodeId, index) => {
    const node = graph.nodes.find(({ id }) => id === nodeId);
    const label = graph.outputs.length === 1 ? 'y' : `y_{${index + 1}}`;
    return {
      nodeId,
      label,
      latex: node ? expressionFor(graph, node, memo, new Set()) : '\\square',
    };
  });
  const rows = expressions.map(({ label, latex }) => `${label} &= ${latex}`);
  return {
    ok: true,
    expressions,
    combinedLatex: rows.length <= 1 ? (rows[0] ?? '\\varnothing') : `\\begin{aligned}${rows.join(' \\\\ ')}\\end{aligned}`,
    issues: validation.issues,
  };
}
