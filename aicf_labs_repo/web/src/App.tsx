import { useCallback, useMemo, useRef, useState, type CSSProperties, type DragEvent } from 'react';
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
import { getOperator, OPERATORS } from './catalog/operators';
import { OperatorNode, type GraphFlowNode } from './components/OperatorNode';
import { serializeGraphDocument } from './core/documentCodec';
import { graphToLatex } from './core/graphToLatex';
import { REWRITE_RULES } from './core/rewriteRules';
import type { FreedomProfile } from './domain/freedom';
import type { Graph, GraphDocument, GraphPosition } from './domain/graph';
import type { OperatorId } from './domain/operator';
import { EXAMPLE_DOCUMENTS } from './examples/graphExamples';
import { toConnectRequest, useGraphStore } from './store/graphStore';

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
      <div className="graph-preview__canvas">
        <ReactFlow
          nodes={flowNodes(graph, positions, null, new Set())}
          edges={flowEdges(graph)}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          zoomOnScroll={false}
          panOnDrag={false}
          proOptions={{ hideAttribution: true }}
        ><Background gap={22} size={1} color="rgba(148, 163, 184, 0.12)" /></ReactFlow>
      </div>
      <div className="graph-preview__formula">{expression.ok ? <Latex value={expression.combinedLatex} compact /> : <span>유효하지 않은 후보</span>}</div>
    </section>
  );
}

const freedomLabels: Record<keyof FreedomProfile, string> = {
  algebraic: 'Algebraic',
  numerical: 'Numerical',
  structural: 'Structural',
  implementation: 'Implementation',
};

export function App() {
  const store = useGraphStore();
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance<GraphFlowNode, Edge> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedNode = store.document.graph.nodes.find(({ id }) => id === store.selectedNodeId);
  const selectedOperator = getOperator(selectedNode?.operatorId);
  const selectedCandidate = store.rewriteCandidates.find(({ id }) => id === store.selectedRewriteCandidateId);
  const expression = useMemo(() => graphToLatex(store.document.graph), [store.document.graph]);
  const issueNodeIds = useMemo(() => new Set(store.validation.issues.flatMap(({ nodeIds }) => nodeIds)), [store.validation.issues]);
  const nodes = useMemo(
    () => flowNodes(store.document.graph, store.document.layout.positions, store.selectedNodeId, issueNodeIds),
    [issueNodeIds, store.document, store.selectedNodeId],
  );
  const edges = useMemo(() => flowEdges(store.document.graph), [store.document.graph]);

  const addOperator = useCallback((operatorId: OperatorId, position?: GraphPosition) => {
    store.addOperator(operatorId, position);
  }, [store]);

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
    addOperator(operatorId, flowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  }, [addOperator, flowInstance]);

  const exportDocument = useCallback(() => {
    const blob = new Blob([serializeGraphDocument(store.document)], { type: 'application/json' });
    const href = URL.createObjectURL(blob);
    const anchor = window.document.createElement('a');
    anchor.href = href;
    anchor.download = `${store.document.graph.id || 'graph'}.json`;
    anchor.style.display = 'none';
    window.document.body.append(anchor);
    anchor.click();
    window.setTimeout(() => {
      anchor.remove();
      URL.revokeObjectURL(href);
    }, 0);
  }, [store.document]);

  const importDocument = useCallback(async (file: File | undefined) => {
    if (!file) return;
    store.importJson(await file.text());
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [store]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block"><span className="brand-mark">AICF</span><div><h1>Graph Lab</h1><p>의미를 연결하고, 동치 변형의 경계를 탐색하세요.</p></div></div>
        <div className={`topbar__status ${store.validation.valid ? '' : 'is-invalid'}`}><span className="status-dot" />{store.validation.valid ? '유효한 DAG' : `${store.validation.issues.length}개 문제`}</div>
        <div className="topbar__actions">
          <select aria-label="예제 그래프" value="" onChange={(event) => event.target.value && store.loadExample(event.target.value)}>
            <option value="">예제 선택</option>
            {EXAMPLE_DOCUMENTS.map((example) => <option key={example.id} value={example.id}>{example.label}</option>)}
          </select>
          <button type="button" onClick={() => fileInputRef.current?.click()}>JSON 불러오기</button>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={(event) => void importDocument(event.target.files?.[0])} />
          <button type="button" onClick={store.reset}>초기화</button>
          <button type="button" className="button-primary" onClick={exportDocument}>저장</button>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel palette-panel">
          <div className="panel-heading"><span>01</span><div><strong>연산자 팔레트</strong><small>클릭 또는 캔버스로 끌기</small></div></div>
          <div className="palette-list">
            {OPERATORS.map((operator) => (
              <button
                type="button"
                className="palette-item"
                key={operator.id}
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.setData('application/aicf-operator', operator.id);
                  event.dataTransfer.effectAllowed = 'copy';
                }}
                onClick={() => addOperator(operator.id)}
                aria-label={`${operator.name} 노드 추가`}
              >
                <span className="palette-symbol" style={{ background: operator.accent }}>{operator.symbol}</span>
                <span><strong>{operator.name}</strong><small>{operator.arity === 0 ? 'source' : `${operator.arity} inputs`}</small></span>
              </button>
            ))}
          </div>
        </aside>

        <section className={`canvas-column ${selectedCandidate ? 'has-comparison' : ''}`}>
          <div className="canvas-header">
            <div><span className="eyebrow">GRAPH / {store.document.graph.id.toUpperCase()}</span><strong>{store.documentName}</strong></div>
            <div className="canvas-legend"><span>{store.document.graph.nodes.length} nodes</span><span>{store.document.graph.edges.length} edges</span><span className={store.validation.valid ? 'valid-pill' : 'invalid-pill'}>{store.validation.valid ? 'Valid' : 'Invalid'}</span></div>
          </div>
          <div className="canvas-surface" aria-label="계산 그래프 캔버스" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              fitView
              minZoom={0.45}
              maxZoom={1.8}
              onInit={setFlowInstance}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={(_, node) => store.selectNode(node.id)}
              onPaneClick={() => store.selectNode(null)}
              deleteKeyCode={['Backspace', 'Delete']}
              proOptions={{ hideAttribution: true }}
            >
              <Background gap={26} size={1} color="rgba(148, 163, 184, 0.16)" />
              <Controls showInteractive={false} />
              <MiniMap pannable zoomable nodeColor="#334155" maskColor="rgba(5, 10, 20, .72)" />
            </ReactFlow>
            {(store.connectionError || store.documentError || store.validation.issues.length > 0) && (
              <section className="validation-overlay" aria-label="그래프 검증 결과">
                <strong>{store.documentError ? '문서 오류' : store.connectionError ? '연결 오류' : 'Validation'}</strong>
                {store.documentError && <p>{store.documentError}</p>}
                {store.connectionError && <p>{store.connectionError}</p>}
                {!store.documentError && !store.connectionError && <ul>{store.validation.issues.slice(0, 4).map((item, index) => <li key={`${item.code}-${index}`}>{item.message}</li>)}</ul>}
                {(store.documentError || store.connectionError) && <button type="button" onClick={store.clearMessages}>닫기</button>}
              </section>
            )}
          </div>

          {selectedCandidate && (
            <section className="comparison-drawer" aria-label="원본과 변형 후보 비교">
              <GraphPreview title="원본 그래프" graph={store.document.graph} positions={store.document.layout.positions} />
              <div className="comparison-arrow"><span>→</span><button type="button" onClick={store.applySelectedRewrite}>이 후보 적용</button></div>
              <GraphPreview title="변형 후보" graph={selectedCandidate.graph} positions={store.document.layout.positions} />
            </section>
          )}

          <div className="formula-strip">
            <span className="eyebrow">MATHEMATICAL FORM</span>
            <div className="formula-preview">
              {expression.ok
                ? expression.expressions.map(({ nodeId, label, latex }) => <Latex key={nodeId} value={`${label} = ${latex}`} compact />)
                : <span className="formula-error">유효한 수식을 만들 수 없습니다.</span>}
            </div>
            <span className="formula-note">{expression.ok ? `${expression.expressions.length} outputs · 위상 순서 계산 완료` : store.validation.issues[0]?.message}</span>
          </div>
        </section>

        <aside className="panel inspector-panel">
          <div className="panel-heading"><span>02</span><div><strong>의미 검사기</strong><small>선택한 연산자의 계약</small></div></div>
          {selectedOperator && selectedNode ? (
            <>
              <div className="operator-hero" style={{ '--hero-accent': selectedOperator.accent } as CSSProperties}><span>{selectedOperator.symbol}</span><div><p>{selectedOperator.category}</p><h2>{selectedOperator.name}</h2></div></div>
              <button className="output-toggle" type="button" aria-pressed={store.document.graph.outputs.includes(selectedNode.id)} onClick={() => store.toggleOutput(selectedNode.id)}>
                {store.document.graph.outputs.includes(selectedNode.id) ? '현재 output에서 제거' : '그래프 output으로 지정'}
              </button>
              <section className="inspector-section"><h3>수학적 의미</h3><p>{selectedOperator.meaning}</p></section>
              <section className="inspector-grid"><div><span>ARITY</span><strong>{selectedOperator.arity}</strong></div><div><span>SHAPE RULE</span><strong>{selectedOperator.shapeRule.notation}</strong></div></section>
              <section className="inspector-section detail-list"><h3>Shape 제약</h3><ul>{selectedOperator.shapeRule.constraints.map((item) => <li key={item}>{item}</li>)}</ul></section>
              <section className="inspector-section detail-list"><h3>대수적 성질</h3><ul>{selectedOperator.algebraicProperties.map((item) => <li key={item}>{item}</li>)}</ul></section>
              <section className="inspector-section detail-list"><h3>수치적 주의점</h3><ul>{selectedOperator.numericalNotes.map((item) => <li key={item}>{item}</li>)}</ul></section>
              <section className="inspector-section detail-list"><h3>적용 가능한 규칙</h3><ul>{selectedOperator.applicableRewriteRuleIds.length > 0
                ? REWRITE_RULES.filter(({ id }) => selectedOperator.applicableRewriteRuleIds.includes(id)).map((rule) => <li key={rule.id}>{rule.name} · {rule.exactness}</li>)
                : <li>현재 등록된 규칙 없음</li>}</ul></section>
              <section className="inspector-section"><h3>변형 자유도</h3><div className="freedom-list">
                {(Object.keys(freedomLabels) as Array<keyof FreedomProfile>).map((axis) => (
                  <div key={axis}><span className={`axis-dot ${axis}`} /><strong>{freedomLabels[axis]}</strong><small>{selectedOperator.freedom[axis].summary}</small></div>
                ))}
              </div></section>
            </>
          ) : (
            <section className="inspector-empty"><strong>노드를 선택하세요</strong><p>팔레트에서 노드를 추가하거나 캔버스의 노드를 선택하면 의미 계약을 확인할 수 있습니다.</p></section>
          )}

          <section className="rewrite-section">
            <div className="rewrite-section__heading"><span>REWRITE CANDIDATES</span><strong>{store.rewriteCandidates.length}개</strong></div>
            {store.rewriteCandidates.length === 0 ? <p className="rewrite-empty">현재 유효하고 방문하지 않은 변형 후보가 없습니다.</p> : (
              <div className="candidate-list">
                {store.rewriteCandidates.map((candidate) => (
                  <button type="button" className={candidate.id === store.selectedRewriteCandidateId ? 'candidate-card is-selected' : 'candidate-card'} key={candidate.id} onClick={() => store.selectRewriteCandidate(candidate.id)}>
                    <span className={`exactness exactness--${candidate.exactness}`}>{candidate.exactness}</span>
                    <strong>{candidate.ruleName}</strong>
                    <small>{candidate.description}</small>
                    <small>조건: {candidate.conditions.join(' · ') || '없음'}</small>
                    <small>변경 노드: {candidate.affectedNodeIds.join(', ')}</small>
                  </button>
                ))}
              </div>
            )}
          </section>
        </aside>
      </section>
    </main>
  );
}
