import { useCallback, useRef } from 'react';
import { serializeGraphDocument } from './core/documentCodec';
import { ModelRootedExplorer } from './components/ModelRootedExplorer';
import { WORKSPACE_DEFINITIONS } from './domain/layerObservation';
import { MODEL_EXAMPLES } from './examples/modelExamples';
import { useGraphStore } from './store/graphStore';
import { GraphWorkspace } from './workspaces/GraphWorkspace';
import { HardwareWorkspace } from './workspaces/HardwareWorkspace';
import { KernelWorkspace } from './workspaces/KernelWorkspace';
import { ModelWorkspace } from './workspaces/ModelWorkspace';
import { RuntimeWorkspace } from './workspaces/RuntimeWorkspace';

function ActiveWorkspace() {
  const activeWorkspace = useGraphStore(({ activeWorkspace }) => activeWorkspace);
  switch (activeWorkspace) {
    case 'model': return <ModelWorkspace />;
    case 'graph': return <GraphWorkspace />;
    case 'kernel': return <KernelWorkspace />;
    case 'runtime': return <RuntimeWorkspace />;
    case 'hardware': return <HardwareWorkspace />;
  }
}

export function App() {
  const store = useGraphStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exportDocument = useCallback(() => {
    const blob = new Blob([serializeGraphDocument(store.document)], { type: 'application/json' });
    const href = URL.createObjectURL(blob);
    const anchor = window.document.createElement('a');
    anchor.href = href;
    anchor.download = (store.document.graph.id || 'graph') + '.json';
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
        <div className="brand-block"><span className="brand-mark">AICF</span><div><h1>Compiler Probe</h1><p>Model에서 silicon까지 하나의 context로 탐색합니다.</p></div></div>
        <div className={`topbar__status ${store.validation.valid ? '' : 'is-invalid'}`}><span className="status-dot" />{store.validation.valid ? '유효한 DAG' : `${store.validation.issues.length}개 문제`}</div>
        <div className="topbar__actions">
          <label className="visually-hidden" htmlFor="model-example-selector">예제 그래프</label>
          <select id="model-example-selector" aria-label="Model example" value="" onChange={(event) => event.target.value && store.loadExample(event.target.value)}>
            <option value="">Model example</option>
            {MODEL_EXAMPLES.map((example) => <option key={example.id} value={example.id}>{example.label}</option>)}
          </select>
          <button type="button" onClick={() => fileInputRef.current?.click()}>JSON 불러오기</button>
          <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={(event) => void importDocument(event.target.files?.[0])} />
          <button type="button" onClick={store.reset}>초기화</button>
          <button type="button" className="button-primary" onClick={exportDocument}>저장</button>
        </div>
      </header>

      <section className={`workspace workspace--${store.activeWorkspace} bottom-panel--${store.bottomPanelSize}`}>
        <nav className="workspace-tabs" role="tablist" aria-label="Compiler workspaces">
          {WORKSPACE_DEFINITIONS.map((workspace) => (
            <button type="button" role="tab" key={workspace.id} aria-selected={store.activeWorkspace === workspace.id} className={store.activeWorkspace === workspace.id ? 'workspace-tab is-active' : 'workspace-tab'} onClick={() => store.setActiveWorkspace(workspace.id)}>
              <span>{workspace.shortLabel}</span><strong>{workspace.label}</strong><small>{workspace.purpose}</small>
            </button>
          ))}
        </nav>
        <ModelRootedExplorer />
        <ActiveWorkspace />
      </section>
    </main>
  );
}
