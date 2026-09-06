import { BottomPanelFrame, ProbeContextSummary, SourcePill, WorkspaceInspector } from '../components/WorkspaceChrome';
import { useGraphStore } from '../store/graphStore';

const METRICS = [
  { id: 'latency', label: 'Latency', value: '~16.1', unit: 'µs', fill: 68, interpretation: 'Illustrative optimized-kernel latency.' },
  { id: 'dram', label: 'DRAM Traffic', value: '~256', unit: 'KB', fill: 52, interpretation: 'Projected traffic after intermediate removal.' },
  { id: 'registers', label: 'Registers', value: '~36', unit: '/ thread', fill: 45, interpretation: 'Estimated allocation before compilation.' },
  { id: 'occupancy', label: 'Occupancy', value: '~75', unit: '%', fill: 75, interpretation: 'Projected active-warp occupancy.' },
  { id: 'sm-utilization', label: 'SM Utilization', value: '~82', unit: '%', fill: 82, interpretation: 'Illustrative compute utilization.' },
  { id: 'bandwidth', label: 'Memory Bandwidth', value: '—', unit: 'GB/s', fill: 0, interpretation: 'Requires hardware counters.' },
] as const;

export function HardwareWorkspace() {
  const store = useGraphStore();
  const activeTab = store.bottomPanelTabs.hardware;
  const selected = METRICS.find(({ id }) => id === store.selectedHardwareMetricId) ?? METRICS[0];
  return (
    <>
      <section className="workspace-main hardware-main">
        <header className="workspace-view-header"><div><span>HARDWARE / PERFORMANCE PROFILE</span><strong>Illustrative GPU metric dashboard</strong></div><SourcePill source="placeholder" /></header>
        <div className="hardware-dashboard">
          <ProbeContextSummary /><div className="metric-grid">{METRICS.map((metric) => <button type="button" key={metric.id} className={selected.id === metric.id ? 'metric-card is-selected' : 'metric-card'} onClick={() => store.selectHardwareMetric(metric.id)}>
            <div><span>{metric.label}</span><SourcePill source="placeholder" /></div><strong>{metric.value}<small>{metric.unit}</small></strong><div className="metric-bar"><i style={{ width: metric.fill + '%' }} /></div><p>Illustrative, not measured</p>
          </button>)}</div>
        </div>
      </section>
      <WorkspaceInspector title="HARDWARE INSPECTOR" subtitle="metric · source · interpretation">
        <div className="workspace-detail-stack"><div className="detail-hero"><span>◫</span><div><small>SELECTED METRIC</small><h2>{selected.label}</h2></div><SourcePill source="placeholder" /></div>
          <dl className="workspace-detail-list"><div><dt>Value</dt><dd>{selected.value} {selected.unit}</dd></div><div><dt>Source</dt><dd>Illustrative placeholder; profiler not connected</dd></div><div><dt>Related kernel</dt><dd>{store.probeContext.kernelId ?? 'kernel-0 projected'}</dd></div><div><dt>Related graph nodes</dt><dd>{store.probeContext.graphNodeIds.join(', ') || 'No nodes selected'}</dd></div><div><dt>Transformation</dt><dd>{store.probeContext.transformationAttemptId ?? 'None selected'}</dd></div><div><dt>Interpretation</dt><dd>{selected.interpretation}</dd></div></dl>
        </div>
      </WorkspaceInspector>
      <BottomPanelFrame workspace="hardware" tabs={[{ id: 'metrics', label: 'Metrics' }, { id: 'counters', label: 'Counters' }, { id: 'comparison', label: 'Comparison' }, { id: 'evidence', label: 'Evidence' }]}>
        {activeTab === 'metrics' && <div className="hardware-bottom"><ProbeContextSummary /><div><span>SELECTED</span><strong>{selected.label}</strong><small>{selected.value} {selected.unit} · PLACEHOLDER</small></div></div>}
        {activeTab === 'counters' && <div className="bottom-table"><div><strong>SM active cycles</strong><span>Not collected</span><SourcePill source="placeholder" /></div><div><strong>DRAM bytes</strong><span>Not collected</span><SourcePill source="placeholder" /></div></div>}
        {activeTab === 'comparison' && <div className="comparison-metrics"><article><span>BEFORE</span><strong>~18.3 µs</strong><small>PLACEHOLDER</small></article><span>→</span><article><span>AFTER</span><strong>~16.1 µs</strong><small>PLACEHOLDER</small></article><article><span>DELTA</span><strong>~−12%</strong><small>ILLUSTRATIVE</small></article></div>}
        {activeTab === 'evidence' && <div className="workspace-empty inline-empty"><strong>No profiler evidence attached</strong><p>Nsight or hardware-counter adapters can populate this contract later.</p><SourcePill source="placeholder" /></div>}
      </BottomPanelFrame>
    </>
  );
}
