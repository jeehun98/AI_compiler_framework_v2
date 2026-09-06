import { BottomPanelFrame, SourcePill, WorkspaceInspector } from '../components/WorkspaceChrome';
import { deriveModelLayerOrdering, modelExplorerItemsFromLayers, modelRootExplorerNodeId, summarizeModelRoot } from '../core/modelRootedExplorer';
import { useGraphStore } from '../store/graphStore';

export function ModelWorkspace() {
  const store = useGraphStore();
  const modelItems = modelExplorerItemsFromLayers(store.modelDocument.layers, store.document.graph);
  const modelSummary = summarizeModelRoot(store.document.graph, modelItems);
  const modelLayerOrdering = deriveModelLayerOrdering(store.document.graph, modelItems);
  const rootId = modelRootExplorerNodeId(store.modelDocument.id);
  const isModelRootSelected = store.selectedModelItemId === rootId;
  const selectedLayer = isModelRootSelected
    ? undefined
    : store.modelDocument.layers.find(({ id }) => id === store.selectedModelItemId) ?? store.modelDocument.layers[0];
  const activeTab = store.bottomPanelTabs.model;
  return (
    <>
      <section className="workspace-main model-main">
        <header className="workspace-view-header"><div><span>MODEL / EDITOR VIEW</span><strong>{store.modelDocument.name}</strong></div><SourcePill source="derived" /></header>
        <div className="model-structure-view">
          <div className={isModelRootSelected ? 'model-root is-selected' : 'model-root'}><span>MODEL</span><strong>{store.modelDocument.name}</strong><small>{modelSummary.layerCount} layers · {store.modelDocument.graphIds.length} graph{store.modelDocument.graphIds.length === 1 ? '' : 's'}</small></div>
          <div className="model-flow">
            {store.modelDocument.layers.length === 0 ? <div className="workspace-empty"><strong>No model structure</strong><p>Add a layer or load a model example.</p></div> : store.modelDocument.layers.map((layer, index) => (
              <div className="model-flow__segment" key={layer.id}>
                {index > 0 && <span>↓</span>}
                <button type="button" className={selectedLayer?.id === layer.id ? 'model-block is-selected' : 'model-block'} onClick={() => store.selectModelItem(layer.id)}>
                  <small>{layer.type}</small><strong>{layer.name}</strong><em>{layer.graphNodeIds.length} mapped graph node(s)</em><SourcePill source={layer.source} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>
      <WorkspaceInspector title="MODEL INSPECTOR" subtitle="definition · shape · parameters">
        {isModelRootSelected ? <div className="workspace-detail-stack">
          <div className="detail-hero"><span>M</span><div><small>Current model / document</small><h2>{store.modelDocument.name}</h2></div><SourcePill source="derived" /></div>
          <dl className="workspace-detail-list">
            <div><dt>Layers</dt><dd>{modelSummary.layerCount}</dd></div>
            <div><dt>Graph nodes</dt><dd>{modelSummary.graphNodeCount}</dd></div>
            <div><dt>Inputs</dt><dd>{modelSummary.inputCount}</dd></div>
            <div><dt>Outputs</dt><dd>{modelSummary.outputCount}</dd></div>
          </dl>
          <section className="model-layer-order" data-order-source={modelLayerOrdering.status}>
            <strong>Layer order</strong>
            {modelLayerOrdering.status === 'derived' ? <ol>{modelLayerOrdering.layers.map((layer) => (
              <li key={layer.modelItemId}><span>{String((layer.order ?? 0) + 1).padStart(2, '0')}</span><strong>{layer.label}</strong></li>
            ))}</ol> : <p><span>UNKNOWN</span> Dependencies do not define one safe layer sequence.</p>}
          </section>
        </div> : selectedLayer ? <div className="workspace-detail-stack">
          <div className="detail-hero"><span>L</span><div><small>{selectedLayer.type}</small><h2>{selectedLayer.name}</h2></div><SourcePill source={selectedLayer.source} /></div>
          <dl className="workspace-detail-list">
            <div><dt>Layer ID</dt><dd>{selectedLayer.id}</dd></div>
            <div><dt>Layer type</dt><dd>{selectedLayer.type}</dd></div>
            <div><dt>Source</dt><dd>{selectedLayer.source.toUpperCase().replace('-', '_')}</dd></div>
            <div><dt>DType / Device</dt><dd>Not represented in ModelDocument</dd></div>
            <div><dt>Related graph nodes</dt><dd>{selectedLayer.graphNodeIds.join(', ') || 'No graph mapping yet'}</dd></div>
          </dl>
          {selectedLayer.graphNodeIds[0] && <button type="button" className="context-link" onClick={() => { store.selectNode(selectedLayer.graphNodeIds[0]!); store.setActiveWorkspace('graph'); }}>Open related node in Graph</button>}
        </div> : <div className="workspace-empty"><strong>Select a model item</strong></div>}
      </WorkspaceInspector>
      <BottomPanelFrame workspace="model" tabs={[{ id: 'summary', label: 'Summary' }, { id: 'structure', label: 'Structure' }, { id: 'shapes', label: 'Shapes' }, { id: 'parameters', label: 'Parameters' }]}>
        <div className="model-bottom-content">
          {activeTab === 'summary' && <><article><span>Layers</span><strong>{modelSummary.layerCount}</strong><small>DERIVED model items</small></article><article><span>Graph nodes</span><strong>{modelSummary.graphNodeCount}</strong><small>REAL graph document</small></article><article><span>Inputs</span><strong>{modelSummary.inputCount}</strong><small>REAL graph operators</small></article><article><span>Outputs</span><strong>{modelSummary.outputCount}</strong><small>REAL graph document</small></article></>}
          {activeTab === 'structure' && <pre>{store.modelDocument.layers.map((layer) => '├─ ' + layer.name + ' (' + layer.type + ')').join('\n') || '(empty model)'}</pre>}
          {activeTab === 'shapes' && <div className="bottom-table">{store.document.graph.nodes.map((node) => <div key={node.id}><strong>{node.id}</strong><span>{'shape' in node.parameters ? '[' + node.parameters.shape.join(', ') + ']' : 'derived after inference'}</span><SourcePill source="derived" /></div>)}</div>}
          {activeTab === 'parameters' && <div className="bottom-table">{store.document.graph.nodes.filter(({ operatorId }) => operatorId === 'constant').map((node) => <div key={node.id}><strong>{node.id}</strong><span>{'value' in node.parameters ? String(node.parameters.value) : '—'}</span><SourcePill source="real" /></div>)}</div>}
        </div>
      </BottomPanelFrame>
    </>
  );
}
