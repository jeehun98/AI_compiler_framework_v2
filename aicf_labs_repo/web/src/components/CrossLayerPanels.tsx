import type { CSSProperties, ReactNode } from 'react';
import { getOperator } from '../catalog/operators';
import { observationForSelection, type CrossLayerModel } from '../core/layerObservations';
import { REWRITE_RULES } from '../core/rewriteRules';
import type { FreedomProfile } from '../domain/freedom';
import type { GraphNode } from '../domain/graph';
import {
  WORKSPACE_LABELS,
  type LayerObservation,
  type LayerObservationStatus,
} from '../domain/layerObservation';
import { useGraphStore } from '../store/graphStore';
import { BottomPanelFrame, SourcePill } from './WorkspaceChrome';

const freedomLabels: Record<keyof FreedomProfile, string> = {
  algebraic: 'Algebraic',
  numerical: 'Numerical',
  structural: 'Structural',
  implementation: 'Implementation',
};

export function StatusBadge({ status }: { status?: LayerObservationStatus }) {
  if (!status) return null;
  return <span className={`status-badge status-badge--${status}`}>{status.toUpperCase()}</span>;
}

export function ObservationCard({ observation, selected = false }: { observation: LayerObservation; selected?: boolean }) {
  return (
    <article className={selected ? 'observation-card is-selected' : 'observation-card'}>
      <div className="observation-card__heading"><StatusBadge status={observation.status} /><SourcePill source={observation.source} /></div>
      <h3>{observation.title}</h3>
      {observation.summary && <p>{observation.summary}</p>}
      {observation.metrics && observation.metrics.length > 0 && (
        <div className="observation-metrics">{observation.metrics.map((metric) => <div key={metric.label}><span>{metric.label}</span><strong>{metric.value}</strong>{metric.delta && <small>{metric.delta}</small>}</div>)}</div>
      )}
      {observation.evidence && observation.evidence.length > 0 && (
        <dl className="evidence-list">{observation.evidence.map((item) => <div key={`${item.label}-${item.value}`}><dt>{item.label}</dt><dd>{item.value}</dd></div>)}</dl>
      )}
      {observation.source === 'placeholder' && <p className="placeholder-note">PLACEHOLDER · {observation.placeholderReason}</p>}
    </article>
  );
}

function SemanticNodeContract({ node }: { node: GraphNode }) {
  const store = useGraphStore();
  const operator = getOperator(node.operatorId);
  if (!operator) return null;
  return (
    <>
      <div className="operator-hero" style={{ '--hero-accent': operator.accent } as CSSProperties}><span>{operator.symbol}</span><div><p>{operator.category}</p><h2>{operator.name}</h2><small>{node.id}</small></div></div>
      <button className="output-toggle" type="button" aria-pressed={store.document.graph.outputs.includes(node.id)} onClick={() => store.toggleOutput(node.id)}>
        {store.document.graph.outputs.includes(node.id) ? '현재 output에서 제거' : '그래프 output으로 지정'}
      </button>
      <section className="inspector-section"><h3>수학적 의미</h3><p>{operator.meaning}</p></section>
      <section className="inspector-grid"><div><span>ARITY</span><strong>{operator.arity}</strong></div><div><span>SHAPE RULE</span><strong>{operator.shapeRule.notation}</strong></div></section>
      <section className="inspector-section detail-list"><h3>Shape 제약</h3><ul>{operator.shapeRule.constraints.map((item) => <li key={item}>{item}</li>)}</ul></section>
      <section className="inspector-section detail-list"><h3>대수적 성질</h3><ul>{operator.algebraicProperties.map((item) => <li key={item}>{item}</li>)}</ul></section>
      <section className="inspector-section detail-list"><h3>수치적 주의점</h3><ul>{operator.numericalNotes.map((item) => <li key={item}>{item}</li>)}</ul></section>
      <section className="inspector-section detail-list"><h3>적용 가능한 규칙</h3><ul>{operator.applicableRewriteRuleIds.length > 0
        ? REWRITE_RULES.filter(({ id }) => operator.applicableRewriteRuleIds.includes(id)).map((rule) => <li key={rule.id}>{rule.name} · {rule.exactness}</li>)
        : <li>현재 등록된 규칙 없음</li>}</ul></section>
      <section className="inspector-section"><h3>변형 자유도</h3><div className="freedom-list">{(Object.keys(freedomLabels) as Array<keyof FreedomProfile>).map((axis) => (
        <div key={axis}><span className={`axis-dot ${axis}`} /><strong>{freedomLabels[axis]}</strong><small>{operator.freedom[axis].summary}</small></div>
      ))}</div></section>
    </>
  );
}

export function GraphInspectorDetails({ observations }: { observations: readonly LayerObservation[] }) {
  const store = useGraphStore();
  const selectedNode = store.document.graph.nodes.find(({ id }) => id === store.selectedNodeId);
  const focused = observationForSelection(observations, 'graph', store.selectedLayerObservationId, store.selectedNodeId);
  const relevant = observations.filter((item) => item.workspace === 'graph' && (!store.selectedNodeId || item.relatedNodeIds?.includes(store.selectedNodeId)))
    .filter((item, index) => item.id === focused?.id || index < 4);
  return (
    <div className="inspector-details">
      <header className="details-context"><span>GRAPH WORKSPACE</span><strong>{selectedNode ? `${getOperator(selectedNode.operatorId)?.name ?? selectedNode.operatorId} · ${selectedNode.id}` : 'Graph overview'}</strong></header>
      {selectedNode ? <SemanticNodeContract node={selectedNode} /> : <section className="inspector-empty"><strong>노드를 선택하세요</strong><p>Graph canvas에서 노드를 선택하면 operator contract와 transformation evidence를 확인할 수 있습니다.</p></section>}
      <section className="layer-observations">
        <div className="section-kicker"><span>GRAPH OBSERVATIONS</span><strong>{relevant.length}</strong></div>
        {relevant.length > 0 ? relevant.map((item) => <ObservationCard key={item.id} observation={item} selected={item.id === focused?.id} />) : <p className="rewrite-empty">No observations match the current graph context.</p>}
      </section>
    </div>
  );
}

export function GraphBottomPanel({ model, formula }: { model: CrossLayerModel; formula: ReactNode }) {
  const store = useGraphStore();
  const activeTab = store.bottomPanelTabs.graph;
  const artifacts = ['Optimized IR', 'Lowering IR', 'CUDA', 'PTX', 'SASS', 'Nsight trace'];
  return (
    <BottomPanelFrame workspace="graph" tabs={[
      { id: 'trace', label: 'Transformation Trace' },
      { id: 'graph-details', label: 'Graph Details' },
      { id: 'performance', label: 'Performance' },
      { id: 'artifacts', label: 'Artifacts' },
    ]}>
      {activeTab === 'trace' && (
        <div className="trace-flow">{model.trace.map((step, index) => (
          <div className="trace-segment" key={step.id}>
            {index > 0 && <span className="trace-arrow">→</span>}
            <button type="button" className={store.selectedLayerObservationId === step.observationId ? 'trace-card is-selected' : 'trace-card'} onClick={() => { store.setActiveWorkspace(step.workspace); store.selectLayerObservation(step.observationId, step.workspace); }}>
              <span className="trace-card__layer">{WORKSPACE_LABELS[step.workspace]} <i className={`source-dot source-dot--${step.source}`} /></span>
              <StatusBadge status={step.status} /><strong>{step.title}</strong><small>{step.summary}</small>
            </button>
          </div>
        ))}</div>
      )}
      {activeTab === 'graph-details' && <div className="graph-detail-bottom"><div><span>MATHEMATICAL FORM</span>{formula}</div><div><span>GRAPH</span><strong>{store.document.graph.nodes.length} nodes · {store.document.graph.edges.length} edges</strong><small>REAL · graph document</small></div></div>}
      {activeTab === 'performance' && <div className="performance-grid">{[
        ['Kernel launches', '1', '2 → 1 projected'], ['Registers / thread', '64', 'estimate'], ['Shared memory', '32 KB', 'estimate'],
        ['Latency', '0.88 ms', '−12% projected'], ['Throughput', '—', 'not measured'], ['Graph breaks', '0', 'not captured'],
      ].map(([label, value, delta]) => <article key={label}><span>{label}</span><strong>{value}</strong><small>{delta}</small><em>PLACEHOLDER</em></article>)}</div>}
      {activeTab === 'artifacts' && <div className="artifact-grid">{artifacts.map((artifact) => <button type="button" disabled key={artifact}><span>◇</span><strong>{artifact}</strong><small>PLACEHOLDER · adapter not connected</small></button>)}</div>}
    </BottomPanelFrame>
  );
}
