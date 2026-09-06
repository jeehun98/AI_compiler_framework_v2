import { useCallback, useMemo, useState, type DragEvent } from 'react';
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  type ReactFlowInstance,
} from '@xyflow/react';
import katex from 'katex';
import { getOperator } from '../catalog/operators';
import { GraphBottomPanel, GraphInspectorDetails } from '../components/CrossLayerPanels';
import { OperatorNode, type GraphFlowNode } from '../components/OperatorNode';
import { graphToLatex } from '../core/graphToLatex';
import { buildCrossLayerModel } from '../core/layerObservations';
import type { Graph, GraphDocument } from '../domain/graph';
import type { OperatorId } from '../domain/operator';
import { toConnectRequest, useGraphStore } from '../store/graphStore';

const nodeTypes = { operator: OperatorNode };
function nodeSubtitle(operatorId: OperatorId, parameters: Graph['nodes'][number]['parameters']): string {
  if (operatorId === 'input' && 'symbol' in parameters) {
    const shape = parameters.shape.length > 0 ? `[${parameters.shape.join(', ')}]` : 'scalar';
    return `${parameters.symbol} · ${shape}`;
  }
  if (operatorId === 'constant' && 'value' in parameters) return String(parameters.value);
  if (operatorId === 'reduceSum' && 'axis' in parameters) return `axis ${parameters.axis}`;
  return getOperator(operatorId)?.category ?? 'unknown';
}

function flowNodes(
  graph: Graph,
  positions: GraphDocument['layout']['positions'],
  selectedNodeId: string | null,
  issueNodeIds: ReadonlySet<string>,
  contextualNodeIds: ReadonlySet<string>,
): GraphFlowNode[] {
  return graph.nodes.map((node, index) => ({
    id: node.id,
    type: 'operator',
    position: positions[node.id] ?? { x: 80 + index * 190, y: 100 },
    selected: selectedNodeId === node.id,
    data: {
      operatorId: node.operatorId,
      subtitle: nodeSubtitle(node.operatorId, node.parameters),
      isOutput: graph.outputs.includes(node.id),
      hasError: issueNodeIds.has(node.id),
      isContextual: contextualNodeIds.has(node.id),
    },
  }));
}

function flowEdges(graph: Graph): Edge[] {
  return graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.sourceNodeId,
    target: edge.targetNodeId,
    sourceHandle: edge.sourcePort,
    targetHandle: edge.targetPort,
    markerEnd: { type: MarkerType.ArrowClosed },
  }));
}

function Latex({ value, compact = false }: { value: string; compact?: boolean }) {
  const markup = useMemo(() => katex.renderToString(value, { throwOnError: false, displayMode: !compact }), [compact, value]);
  return <span className={compact ? 'latex latex--compact' : 'latex'} dangerouslySetInnerHTML={{ __html: markup }} />;
}

function GraphPreview({ title, graph, positions }: { title: string; graph: Graph; positions: GraphDocument['layout']['positions'] }) {
  const expression = graphToLatex(graph);
  return (
    <section className="graph-preview">
      <div className="graph-preview__heading"><strong>{title}</strong><span>{graph.nodes.length} nodes · {graph.edges.length} edges</span></div>
      <div className="graph-preview__canvas"><ReactFlow nodes={flowNodes(graph, positions, null, new Set(), new Set())} edges={flowEdges(graph)} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.25 }} nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} zoomOnScroll={false} panOnDrag={false} proOptions={{ hideAttribution: true }}><Background gap={22} size={1} color="rgba(148, 163, 184, 0.12)" /></ReactFlow></div>
      <div className="graph-preview__formula">{expression.ok ? expression.expressions.map(({ nodeId, label, latex }) => <Latex key={nodeId} value={`${label} = ${latex}`} compact />) : <span>유효하지 않은 후보</span>}</div>
    </section>
  );
}

export function GraphWorkspace() {
  const store = useGraphStore();
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance<GraphFlowNode, Edge> | null>(null);
  const selectedCandidate = store.rewriteCandidates.find(({ id }) => id === store.selectedRewriteCandidateId);
  const model = useMemo(() => buildCrossLayerModel(store.document.graph, store.rewriteCandidates, store.transformationAttempts, store.selectedRewriteCandidateId, store.selectedTransformationAttemptId), [store.document.graph, store.rewriteCandidates, store.transformationAttempts, store.selectedRewriteCandidateId, store.selectedTransformationAttemptId]);
  const expression = useMemo(() => graphToLatex(store.document.graph), [store.document.graph]);
  const issueNodeIds = useMemo(() => new Set(store.validation.issues.flatMap(({ nodeIds }) => nodeIds)), [store.validation.issues]);
  const contextualNodeIds = useMemo(() => {
    const selectedObservation = model.observations.find(({ id }) => id === store.selectedLayerObservationId);
    return new Set(store.probeContext.graphNodeIds.length > 0 ? store.probeContext.graphNodeIds : selectedCandidate?.affectedNodeIds ?? selectedObservation?.relatedNodeIds ?? []);
  }, [model.observations, selectedCandidate, store.probeContext.graphNodeIds, store.selectedLayerObservationId]);
  const nodes = useMemo(() => flowNodes(store.document.graph, store.document.layout.positions, store.selectedNodeId, issueNodeIds, contextualNodeIds), [contextualNodeIds, issueNodeIds, store.document, store.selectedNodeId]);
  const edges = useMemo(() => flowEdges(store.document.graph), [store.document.graph]);

  const onNodesChange = useCallback((changes: NodeChange<GraphFlowNode>[]) => {
    for (const change of changes) {
      if (change.type === 'position' && change.position) store.moveNode(change.id, change.position);
      if (change.type === 'remove') store.deleteNode(change.id);
      if (change.type === 'select' && change.selected) store.selectNode(change.id);
    }
  }, [store]);
  const onEdgesChange = useCallback((changes: EdgeChange<Edge>[]) => {
    for (const change of changes) if (change.type === 'remove') store.deleteEdge(change.id);
  }, [store]);
  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target || !connection.targetHandle) return;
    const request = toConnectRequest(connection.source, connection.target, connection.targetHandle);
    if (request) store.connectEdge(request);
  }, [store]);
  const onDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const operatorId = event.dataTransfer.getData('application/aicf-operator') as OperatorId;
    if (!getOperator(operatorId) || !flowInstance) return;
    store.addOperator(operatorId, flowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  }, [flowInstance, store]);

  const formula = expression.ok
    ? <div className="formula-preview">{expression.expressions.map(({ nodeId, label, latex }) => <Latex key={nodeId} value={`${label} = ${latex}`} compact />)}</div>
    : <span className="formula-error">{store.validation.issues[0]?.message ?? '유효한 수식을 만들 수 없습니다.'}</span>;

  return (
    <>
      <section className={`canvas-column workspace-main ${selectedCandidate ? 'has-comparison' : ''}`}>
        <div className="canvas-header"><div><span className="eyebrow">GRAPH / {store.document.graph.id.toUpperCase()}</span><strong>{store.document.graph.name}</strong></div><div className="canvas-legend"><span>{store.document.graph.nodes.length} nodes</span><span>{store.document.graph.edges.length} edges</span><span className={store.validation.valid ? 'valid-pill' : 'invalid-pill'}>{store.validation.valid ? 'Valid' : 'Invalid'}</span></div></div>
        <div className="canvas-surface" aria-label="계산 그래프 캔버스" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.45} maxZoom={1.8} onInit={setFlowInstance} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={(_, node) => store.selectNode(node.id)} onPaneClick={() => store.selectNode(null)} deleteKeyCode={['Backspace', 'Delete']} proOptions={{ hideAttribution: true }}>
            <Background gap={26} size={1} color="rgba(148, 163, 184, 0.16)" /><Controls showInteractive={false} /><MiniMap pannable zoomable nodeColor="#334155" maskColor="rgba(5, 10, 20, .72)" />
          </ReactFlow>
          {(store.connectionError || store.documentError || store.validation.issues.length > 0) && <section className="validation-overlay" aria-label="그래프 검증 결과"><strong>{store.documentError ? '문서 오류' : store.connectionError ? '연결 오류' : 'Validation'}</strong>{store.documentError && <p>{store.documentError}</p>}{store.connectionError && <p>{store.connectionError}</p>}{!store.documentError && !store.connectionError && <ul>{store.validation.issues.slice(0, 4).map((item, index) => <li key={`${item.code}-${index}`}>{item.message}</li>)}</ul>}{(store.documentError || store.connectionError) && <button type="button" onClick={store.clearMessages}>닫기</button>}</section>}
        </div>
        {selectedCandidate && <section className="comparison-drawer" aria-label="원본과 변형 후보 비교"><GraphPreview title="원본 그래프" graph={store.document.graph} positions={store.document.layout.positions} /><div className="comparison-arrow"><span>→</span><button type="button" onClick={store.applySelectedRewrite}>이 후보 적용</button></div><GraphPreview title="변형 후보" graph={selectedCandidate.graph} positions={store.document.layout.positions} /></section>}
      </section>

      <aside className="panel graph-inspector workspace-inspector">
        <div className="panel-heading"><span>03</span><div><strong>GRAPH INSPECTOR</strong><small>capture · properties · transformations · plan</small></div></div>
        <GraphInspectorDetails observations={model.observations} />
      </aside>

      <GraphBottomPanel model={model} formula={formula} />
    </>
  );
}
