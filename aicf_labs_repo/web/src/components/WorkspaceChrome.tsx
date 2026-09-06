import type { ReactNode } from 'react';
import type { ObservationSource, WorkspaceBottomTab, WorkspaceId } from '../domain/layerObservation';
import { useGraphStore } from '../store/graphStore';

export function SourcePill({ source }: { source: ObservationSource }) {
  return <span className={`source-badge source-badge--${source}`}>{source === 'user-defined' ? 'user_defined' : source}</span>;
}

export function WorkspaceInspector({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <aside className="panel workspace-inspector">
      <div className="panel-heading"><span>03</span><div><strong>{title}</strong><small>{subtitle}</small></div></div>
      <div className="workspace-inspector__body">{children}</div>
    </aside>
  );
}

export function BottomPanelFrame({
  workspace,
  tabs,
  children,
}: {
  workspace: WorkspaceId;
  tabs: readonly { id: WorkspaceBottomTab; label: string }[];
  children: ReactNode;
}) {
  const store = useGraphStore();
  const activeTab = store.bottomPanelTabs[workspace];
  return (
    <section className="workspace-bottom" aria-label={`${workspace} workspace details`}>
      <div className="workspace-bottom__tabs" role="tablist" aria-label={`${workspace} details tabs`}>
        {tabs.map((tab) => (
          <button type="button" role="tab" key={tab.id} aria-selected={activeTab === tab.id} className={activeTab === tab.id ? 'is-active' : ''} onClick={() => store.setWorkspaceBottomTab(workspace, tab.id)}>{tab.label}</button>
        ))}
        <div className="bottom-size-controls" aria-label="Bottom panel size">
          {(['collapsed', 'compact', 'expanded'] as const).map((size) => (
            <button
              type="button"
              key={size}
              className={store.bottomPanelSize === size ? 'is-active' : ''}
              aria-label={`${size} bottom panel`}
              aria-pressed={store.bottomPanelSize === size}
              onClick={() => store.setBottomPanelSize(size)}
            >{size === 'collapsed' ? '–' : size === 'compact' ? '□' : '▱'}</button>
          ))}
        </div>
      </div>
      <div className="workspace-bottom__body">{children}</div>
    </section>
  );
}

export function ProbeContextSummary() {
  const probe = useGraphStore(({ probeContext }) => probeContext);
  return (
    <div className="probe-context-summary">
      <span>SHARED PROBE CONTEXT</span>
      <strong>{probe.transformationAttemptId ?? 'No transformation selected'}</strong>
      <small>{probe.graphNodeIds.length > 0 ? `${probe.graphNodeIds.length} related graph nodes` : 'Select a graph node or transformation to link workspaces.'}</small>
    </div>
  );
}
