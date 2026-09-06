import { BottomPanelFrame, ProbeContextSummary, SourcePill, WorkspaceInspector } from '../components/WorkspaceChrome';
import { useGraphStore } from '../store/graphStore';

const EVENTS = [
  { id: 'runtime-event-0', label: 'gemm_epilogue_scale', lane: 'Stream 0', left: 7, width: 58, detail: 'Projected fused kernel' },
  { id: 'runtime-event-1', label: 'memcpy_d2d', lane: 'Stream 1', left: 27, width: 29, detail: 'Illustrative transfer' },
] as const;

export function RuntimeWorkspace() {
  const store = useGraphStore();
  const activeTab = store.bottomPanelTabs.runtime;
  const selected = EVENTS.find(({ id }) => id === store.selectedRuntimeEventId) ?? EVENTS[0];
  return (
    <>
      <section className="workspace-main runtime-main">
        <header className="workspace-view-header"><div><span>RUNTIME / EXECUTION VIEW</span><strong>Projected launch timeline</strong></div><SourcePill source="placeholder" /></header>
        <div className="runtime-timeline">
          <ProbeContextSummary /><div className="timeline-axis"><span>Time</span><i /><small>→</small></div>
          {['Stream 0', 'Stream 1'].map((lane) => <div className="timeline-lane" key={lane}><strong>{lane}</strong><div>{EVENTS.filter((event) => event.lane === lane).map((event) => <button type="button" key={event.id} className={selected.id === event.id ? 'timeline-event is-selected' : 'timeline-event'} style={{ left: event.left + '%', width: event.width + '%' }} onClick={() => store.selectRuntimeEvent(event.id)}><span>{event.label}</span><small>PLACEHOLDER</small></button>)}</div></div>)}
          <div className="launch-sequence"><span>KERNEL LAUNCH SEQUENCE</span>{EVENTS.map((event, index) => <button type="button" key={event.id} onClick={() => store.selectRuntimeEvent(event.id)}><i>{index + 1}</i><strong>{event.label}</strong><small>{event.detail}</small></button>)}</div>
        </div>
      </section>
      <WorkspaceInspector title="RUNTIME INSPECTOR" subtitle="launch · stream · dependencies">
        <div className="workspace-detail-stack"><div className="detail-hero"><span>▶</span><div><small>RUNTIME EVENT</small><h2>{selected.label}</h2></div><SourcePill source="placeholder" /></div>
          <dl className="workspace-detail-list"><div><dt>Stream</dt><dd>{selected.lane}</dd></div><div><dt>Start / End</dt><dd>Available after runtime tracing</dd></div><div><dt>Duration</dt><dd>Not measured</dd></div><div><dt>Grid / Block</dt><dd>Projected from Kernel workspace</dd></div><div><dt>Dependencies</dt><dd>{store.probeContext.transformationAttemptId ?? 'No transformation context'}</dd></div><div><dt>CUDA Graph state</dt><dd>Not captured</dd></div></dl>
        </div>
      </WorkspaceInspector>
      <BottomPanelFrame workspace="runtime" tabs={[{ id: 'timeline', label: 'Timeline' }, { id: 'launches', label: 'Launches' }, { id: 'buffers', label: 'Buffers' }, { id: 'cuda-graph', label: 'CUDA Graph' }]}>
        {activeTab === 'timeline' && <div className="runtime-bottom"><ProbeContextSummary /><div><span>EVENTS</span><strong>{EVENTS.length} illustrative</strong><small>PLACEHOLDER</small></div><div><span>STREAMS</span><strong>2</strong><small>PLACEHOLDER</small></div></div>}
        {activeTab === 'launches' && <div className="bottom-table">{EVENTS.map((event, index) => <div key={event.id}><strong>{index + 1}. {event.label}</strong><span>{event.lane}</span><SourcePill source="placeholder" /></div>)}</div>}
        {activeTab === 'buffers' && <div className="bottom-table"><div><strong>buffer_A / buffer_B</strong><span>Lifetime unavailable</span><SourcePill source="placeholder" /></div><div><strong>buffer_output</strong><span>Allocation unavailable</span><SourcePill source="placeholder" /></div></div>}
        {activeTab === 'cuda-graph' && <div className="workspace-empty inline-empty"><strong>CUDA Graph trace not connected</strong><p>This Capture refers to runtime launch capture, not Model → Graph capture.</p><SourcePill source="placeholder" /></div>}
      </BottomPanelFrame>
    </>
  );
}
