import { getOperator } from '../catalog/operators';
import type { Graph, GraphNode } from '../domain/graph';
import type {
  LayerObservation,
  LayerObservationStatus,
  TransformationTraceStep,
  WorkspaceId,
} from '../domain/layerObservation';
import { LegalityStatus, type RewriteCandidate, type TransformationAttempt } from '../domain/rewrite';
import { selectTransformationAttemptReason } from './transformationAttemptSummary';

const PLACEHOLDER_REASON = 'Backend instrumentation is not connected in this UI milestone.';

function attemptStatus(status: TransformationAttempt['status']): LayerObservationStatus {
  if (status === LegalityStatus.REJECTED) return 'rejected';
  if (status === LegalityStatus.UNKNOWN) return 'unknown';
  return 'valid';
}

function placeholder(
  id: string,
  workspace: WorkspaceId,
  category: string,
  node: GraphNode,
  title: string,
  summary: string,
  status: LayerObservationStatus = 'unknown',
  evidence: LayerObservation['evidence'] = [],
  metrics: LayerObservation['metrics'] = [],
): LayerObservation {
  return {
    id,
    workspace,
    category,
    title,
    summary,
    status,
    evidence,
    metrics,
    relatedNodeIds: [node.id],
    source: 'placeholder',
    placeholderReason: PLACEHOLDER_REASON,
  };
}

function nodePlaceholders(node: GraphNode): LayerObservation[] {
  const operator = getOperator(node.operatorId);
  const name = operator?.name ?? node.operatorId;
  const isMatMul = node.operatorId === 'matmul';
  const isSource = operator?.arity === 0;
  return [
    placeholder(`capture:graph-breaks:${node.id}`, 'graph', 'Capture / Graph Breaks', node, `Capture boundary · ${name}`, 'No graph break is projected around this node.', 'unknown', [
      { label: 'Capture region', value: 'Awaiting frontend trace' },
    ]),
    placeholder(`capture:alias:${node.id}`, 'graph', 'Capture / Alias & Mutation', node, `Alias contract · ${name}`, 'Alias and mutation analysis will be populated by the graph capture adapter.'),
    placeholder(`capture:dynamic-shape:${node.id}`, 'graph', 'Capture / Dynamic Shapes', node, `Shape guards · ${name}`, 'Dynamic dimension constraints are not yet captured.'),
    placeholder(`plan:regions:${node.id}`, 'graph', 'Plan / Regions', node, `Region containing ${node.id}`, isSource ? 'Source value enters the first executable region.' : `${name} is grouped with adjacent compatible operators.`, 'changed', [
      { label: 'Projected region', value: isMatMul ? 'linear-main' : 'elementwise-main' },
    ]),
    placeholder(`plan:materialization:${node.id}`, 'graph', 'Plan / Materialization', node, `Materialization · ${name}`, isMatMul ? 'The MatMul result is projected to stay in-register for its epilogue.' : 'Value materialization has not been scheduled.', isMatMul ? 'embedded' : 'unknown'),
    placeholder(`plan:guards:${node.id}`, 'graph', 'Plan / Guards', node, `Guards · ${name}`, isMatMul ? 'K dimension alignment and contiguous-stride guards are projected.' : 'Runtime guards will appear after planning.'),
    placeholder(`lowering:kernel:${node.id}`, 'kernel', 'Kernel Mapping', node, isMatMul ? 'Tiled GEMM kernel' : `${name} kernel mapping`, isMatMul ? 'Maps to a GEMM mainloop with an epilogue slot.' : 'A backend kernel mapping has not been selected.', isMatMul ? 'embedded' : 'unknown', [
      { label: 'Backend', value: isMatMul ? 'CUDA GEMM candidate' : 'Unassigned' },
    ]),
    placeholder(`lowering:tile:${node.id}`, 'kernel', 'Tiling', node, `Tile / threads · ${name}`, isMatMul ? 'Projected CTA tile 128 × 128 × 32 with 256 threads.' : 'Launch geometry is awaiting lowering.', 'unknown', [], isMatMul ? [{ label: 'CTA tile', value: '128 × 128 × 32' }, { label: 'Threads', value: '256' }] : []),
    placeholder(`lowering:layout:${node.id}`, 'kernel', 'Memory Layout', node, `Memory layout · ${name}`, isMatMul ? 'Row-major inputs with an epilogue accumulator layout are projected.' : 'Physical layout is not assigned.'),
    placeholder(`runtime:launches:${node.id}`, 'runtime', 'Launches', node, `${name} launch`, isSource ? 'Source nodes do not launch kernels.' : isMatMul ? 'Projected as one fused GEMM launch.' : 'Launch count is awaiting a runtime trace.', isMatMul ? 'improved' : 'unknown', [], isMatMul ? [{ label: 'Kernel launches', value: '1', delta: '2 → 1 projected' }] : []),
    placeholder(`runtime:buffers:${node.id}`, 'runtime', 'Buffers', node, `Buffers · ${name}`, isMatMul ? 'Projected epilogue fusion removes one intermediate buffer.' : 'Allocation events will appear after execution.'),
    placeholder(`runtime:timeline:${node.id}`, 'runtime', 'Timeline', node, `Timeline event · ${name}`, 'No runtime event has been recorded.'),
    placeholder(`hardware:dram:${node.id}`, 'hardware', 'DRAM Traffic', node, `DRAM traffic · ${name}`, isMatMul ? 'Projected fusion avoids an intermediate write and read.' : 'Hardware counters are not available.', isMatMul ? 'improved' : 'unknown', [], isMatMul ? [{ label: 'Intermediate traffic', value: '0 B', delta: 'projected' }] : []),
    placeholder(`hardware:registers:${node.id}`, 'hardware', 'Registers', node, `Registers · ${name}`, isMatMul ? 'Projected register pressure remains within the target occupancy budget.' : 'Register allocation is unavailable.', 'unknown', [], isMatMul ? [{ label: 'Registers / thread', value: '64', delta: 'estimate' }] : []),
    placeholder(`hardware:occupancy:${node.id}`, 'hardware', 'Occupancy', node, `Occupancy · ${name}`, isMatMul ? 'Projected occupancy is 50% for the candidate tile.' : 'Occupancy requires a compiled kernel.', 'unknown', [], isMatMul ? [{ label: 'Occupancy', value: '50%', delta: 'estimate' }] : []),
    placeholder(`hardware:stalls:${node.id}`, 'hardware', 'Stall Reasons', node, `Stall reasons · ${name}`, 'Profiler stall samples are not connected.'),
  ];
}

export interface CrossLayerModel {
  observations: readonly LayerObservation[];
  trace: readonly TransformationTraceStep[];
}

export function buildCrossLayerModel(
  graph: Graph,
  candidates: readonly RewriteCandidate[],
  attempts: readonly TransformationAttempt[],
  selectedCandidateId: string | null,
  selectedAttemptId: string | null,
): CrossLayerModel {
  const nodeIds = graph.nodes.map(({ id }) => id);
  const observations: LayerObservation[] = [
    {
      id: 'model:definition', workspace: 'model', category: 'Definition', title: graph.name,
      summary: `${graph.nodes.length} nodes and ${graph.edges.length} data-flow edges.`, status: 'valid',
      metrics: [{ label: 'Nodes', value: String(graph.nodes.length) }, { label: 'Outputs', value: String(graph.outputs.length) }],
      relatedNodeIds: nodeIds, source: 'real', provenance: 'graph-document',
    },
    {
      id: 'model:shapes', workspace: 'model', category: 'Shapes', title: 'Symbolic shape inventory',
      summary: 'Shapes shown here come from graph node parameters; no backend inference is implied.', status: 'valid',
      relatedNodeIds: nodeIds, source: 'derived', derivedFrom: 'graph-structure',
    },
    {
      id: 'model:properties', workspace: 'model', category: 'Operator Properties', title: 'Semantic operator contracts',
      summary: 'Algebraic, numerical, structural, and implementation freedom comes from the operator catalog.', status: 'valid',
      relatedNodeIds: nodeIds, source: 'real', provenance: 'graph-document',
    },
    {
      id: 'graph:nodes', workspace: 'graph', category: 'Graph / Nodes', title: 'Operator nodes',
      summary: `${graph.nodes.length} operator nodes from the active GraphDocument.`, status: 'valid',
      metrics: [{ label: 'Nodes', value: String(graph.nodes.length) }, { label: 'Outputs', value: String(graph.outputs.length) }],
      relatedNodeIds: nodeIds, source: 'real', provenance: 'graph-document',
    },
    {
      id: 'graph:edges', workspace: 'graph', category: 'Graph / Edges', title: 'Data-flow dependencies',
      summary: `${graph.edges.length} directed edges preserve input-port ordering.`, status: 'valid',
      metrics: [{ label: 'Edges', value: String(graph.edges.length) }], relatedNodeIds: nodeIds,
      source: 'real', provenance: 'graph-document',
    },
    {
      id: 'graph:operator-properties', workspace: 'graph', category: 'Operator Properties', title: 'Operator property index',
      summary: 'LINEAR, ASSOCIATIVE, PURE, ELEMENTWISE, and REDUCTION claims are derived from the operator catalog.', status: 'valid',
      relatedNodeIds: nodeIds, source: 'derived', derivedFrom: 'graph-structure',
    },
    ...graph.nodes.flatMap(nodePlaceholders),
  ];

  for (const candidate of candidates) {
    observations.push({
      id: `semantic:candidate:${candidate.id}`,
      workspace: 'graph',
      category: 'Transformations / Rewrite Candidates',
      title: candidate.ruleName,
      summary: candidate.description,
      status: 'valid',
      evidence: [
        { label: 'Exactness', value: candidate.exactness },
        { label: 'Conditions', value: candidate.conditions.join(' · ') || 'None' },
        { label: 'Justification', value: candidate.justification },
      ],
      metrics: [{ label: 'Affected nodes', value: String(candidate.affectedNodeIds.length) }],
      relatedNodeIds: candidate.affectedNodeIds,
      relatedTransformationAttemptIds: [candidate.attemptId],
      source: 'real',
      provenance: 'rewrite-engine',
    });
  }

  for (const attempt of attempts) {
    const relatedNodeIds = [...new Set(Object.values(attempt.bindings).filter((id) => graph.nodes.some((node) => node.id === id)))];
    observations.push({
      id: `semantic:attempt:${attempt.id}`,
      workspace: 'graph',
      category: 'Transformations / Transformation Attempts',
      title: attempt.ruleName,
      summary: selectTransformationAttemptReason(attempt),
      status: attemptStatus(attempt.status),
      evidence: [
        { label: 'Mathematical legality', value: attempt.mathematicalLegality.status },
        { label: 'Graph legality', value: attempt.graphLegality.status },
        { label: 'Semantic domain', value: attempt.semanticDomain },
      ],
      relatedNodeIds,
      relatedTransformationAttemptIds: [attempt.id],
      source: 'real',
      provenance: 'rewrite-engine',
    });
  }

  observations.push({
    id: 'semantic:legality-summary', workspace: 'graph', category: 'Transformations / Legality Evidence', title: 'Legality evidence summary',
    summary: attempts.length === 0 ? 'No structural matches have produced legality evidence.' : `${attempts.length} transformation attempts were evaluated.`,
    status: attempts.some(({ status }) => status === LegalityStatus.REJECTED) ? 'warning' : attempts.length > 0 ? 'valid' : 'unknown',
    metrics: [
      { label: 'Applicable', value: String(attempts.filter(({ status }) => status === LegalityStatus.APPLICABLE).length) },
      { label: 'Rejected', value: String(attempts.filter(({ status }) => status === LegalityStatus.REJECTED).length) },
      { label: 'Unknown', value: String(attempts.filter(({ status }) => status === LegalityStatus.UNKNOWN).length) },
    ],
    relatedNodeIds: nodeIds, relatedTransformationAttemptIds: attempts.map(({ id }) => id), source: 'real', provenance: 'rewrite-engine',
  });

  const selectedCandidate = candidates.find(({ id }) => id === selectedCandidateId);
  const selectedAttempt = attempts.find(({ id }) => id === (selectedAttemptId ?? selectedCandidate?.attemptId))
    ?? attempts.find(({ ruleName }) => ruleName === 'ScaleThroughLinear')
    ?? attempts[0];
  const contextName = selectedCandidate?.ruleName ?? selectedAttempt?.ruleName ?? 'Transformation context';
  const related = selectedCandidate?.affectedNodeIds
    ?? (selectedAttempt ? [...new Set(Object.values(selectedAttempt.bindings).filter((id) => nodeIds.includes(id)))] : nodeIds);
  const semanticStatus = selectedAttempt ? attemptStatus(selectedAttempt.status) : 'unknown';
  const downstream = [
    { workspace: 'kernel' as const, status: 'embedded' as const, title: 'Rewrite embedded', summary: contextName === 'ScaleThroughLinear' ? 'Scale is projected into the MatMul epilogue.' : 'The rewrite is projected into the selected kernel mapping.' },
    { workspace: 'runtime' as const, status: 'improved' as const, title: '2 kernels → 1', summary: 'A projected intermediate launch and buffer are removed.' },
    { workspace: 'hardware' as const, status: 'improved' as const, title: '−12% latency', summary: 'Illustrative hardware impact; profiling is not connected.' },
  ];
  for (const item of downstream) {
    observations.push({
      id: `trace:${item.workspace}`,
      workspace: item.workspace,
      category: 'Transformation Trace',
      title: item.title,
      summary: item.summary,
      status: selectedAttempt ? item.status : 'unknown',
      relatedNodeIds: related,
      relatedTransformationAttemptIds: selectedAttempt ? [selectedAttempt.id] : [],
      source: 'placeholder',
      placeholderReason: PLACEHOLDER_REASON,
    });
  }

  const semanticObservationId = selectedCandidate
    ? `semantic:candidate:${selectedCandidate.id}`
    : selectedAttempt
      ? `semantic:attempt:${selectedAttempt.id}`
      : 'semantic:legality-summary';
  const trace: TransformationTraceStep[] = [
    {
      id: 'trace-step-graph', observationId: semanticObservationId, workspace: 'graph', title: contextName,
      summary: selectedAttempt ? selectTransformationAttemptReason(selectedAttempt) : 'Select a rewrite candidate to establish a cross-layer context.',
      status: semanticStatus, source: selectedAttempt ? 'real' : 'derived',
    },
    ...downstream.map((item) => ({
      id: `trace-step-${item.workspace}`,
      observationId: `trace:${item.workspace}`,
      workspace: item.workspace,
      title: item.title,
      summary: item.summary,
      status: selectedAttempt ? item.status : 'unknown' as LayerObservationStatus,
      source: 'placeholder' as const,
    })),
  ];

  return { observations, trace };
}

export function observationForSelection(
  observations: readonly LayerObservation[],
  workspace: WorkspaceId,
  selectedObservationId: string | null,
  selectedNodeId: string | null,
): LayerObservation | undefined {
  const explicit = observations.find((item) => item.id === selectedObservationId && item.workspace === workspace);
  if (explicit) return explicit;
  return observations.find((item) => item.workspace === workspace && selectedNodeId !== null && item.relatedNodeIds?.includes(selectedNodeId))
    ?? observations.find((item) => item.workspace === workspace);
}
