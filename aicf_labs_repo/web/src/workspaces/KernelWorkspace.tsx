import { BottomPanelFrame, ProbeContextSummary, SourcePill, WorkspaceInspector } from '../components/WorkspaceChrome';
import { useGraphStore } from '../store/graphStore';

export function KernelWorkspace() {
  const store = useGraphStore();
  const activeTab = store.bottomPanelTabs.kernel;
  const relatedNodes = store.probeContext.graphNodeIds.length > 0 ? store.probeContext.graphNodeIds : store.document.graph.nodes.filter(({ operatorId }) => operatorId === 'matmul').map(({ id }) => id);
  const selectedKernelId = store.selectedKernelId ?? 'kernel-0';
  return (
    <>
      <section className="workspace-main kernel-main">
        <header className="workspace-view-header"><div><span>KERNEL / IMPLEMENTATION VIEW</span><strong>Projected GPU kernel structure</strong></div><SourcePill source="placeholder" /></header>
        <div className="kernel-structure">
          <ProbeContextSummary />
          <div className="kernel-region"><span>SELECTED GRAPH REGION</span><div>{relatedNodes.length > 0 ? relatedNodes.map((id) => <button type="button" key={id} onClick={() => { store.selectNode(id); store.setActiveWorkspace('graph'); }}>{id}</button>) : <small>No graph region selected</small>}</div></div>
          <span className="structure-arrow">↓ LOWERING</span>
          <button type="button" className="kernel-card is-selected" onClick={() => store.selectKernel(selectedKernelId)}>
            <div><span>KERNEL 0</span><SourcePill source="placeholder" /></div><h2>gemm_epilogue_scale</h2>
            <dl><div><dt>Mapping</dt><dd>MatMul → tiled GEMM</dd></div><div><dt>Block</dt><dd>128 × 1 × 1</dd></div><div><dt>Tile</dt><dd>BM 128 · BN 128 · BK 32</dd></div><div><dt>Shared memory</dt><dd>48 KB projected</dd></div><div><dt>Epilogue</dt><dd>Scale</dd></div></dl>
          </button>
        </div>
      </section>
      <WorkspaceInspector title="KERNEL INSPECTOR" subtitle="mapping · tile · memory · artifacts">
        <div className="workspace-detail-stack">
          <div className="detail-hero"><span>K</span><div><small>SELECTED KERNEL</small><h2>{selectedKernelId}</h2></div><SourcePill source="placeholder" /></div>
          <dl className="workspace-detail-list">
            <div><dt>Source graph nodes</dt><dd>{relatedNodes.join(', ') || 'None selected'}</dd></div><div><dt>Lowering decision</dt><dd>Tiled GEMM with fused scalar epilogue</dd></div>
            <div><dt>Grid / Block</dt><dd>Derived after concrete shapes / 128 × 1 × 1</dd></div><div><dt>Warp mapping</dt><dd>8 projected warps</dd></div>
            <div><dt>Memory layout</dt><dd>Row-major projected layout</dd></div><div><dt>Artifact availability</dt><dd>CUDA, PTX, SASS unavailable</dd></div>
          </dl>
        </div>
      </WorkspaceInspector>
      <BottomPanelFrame workspace="kernel" tabs={[{ id: 'kernel-summary', label: 'Kernel Summary' }, { id: 'lowering', label: 'Lowering' }, { id: 'artifacts', label: 'Artifacts' }, { id: 'performance', label: 'Performance' }]}>
        {activeTab === 'kernel-summary' && <div className="kernel-bottom-summary"><ProbeContextSummary /><div><span>KERNELS</span><strong>1 projected</strong><small>PLACEHOLDER</small></div><div><span>FUSION</span><strong>MatMul + Scale</strong><small>PLACEHOLDER</small></div></div>}
        {activeTab === 'lowering' && <pre className="artifact-preview">PLACEHOLDER LOWERED IR{String.fromCharCode(10)}for bm, bn:{String.fromCharCode(10)}  acc = dot(A, B){String.fromCharCode(10)}  output = acc * alpha</pre>}
        {activeTab === 'artifacts' && <div className="artifact-grid">{['Lowered IR', 'CUDA', 'PTX', 'SASS'].map((label) => <button type="button" disabled key={label}><span>◇</span><strong>{label}</strong><small>PLACEHOLDER</small></button>)}</div>}
        {activeTab === 'performance' && <div className="performance-grid">{[['Registers / thread', '—'], ['Shared memory', '48 KB*'], ['Occupancy', '—'], ['Kernel time', '—']].map(([label, value]) => <article key={label}><span>{label}</span><strong>{value}</strong><small>not measured</small><em>PLACEHOLDER</em></article>)}</div>}
      </BottomPanelFrame>
    </>
  );
}
