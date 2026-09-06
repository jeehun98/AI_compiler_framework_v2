import { useEffect, useMemo, useState, type CSSProperties, type KeyboardEvent } from 'react';
import { OPERATORS } from '../catalog/operators';
import { buildModelRootedExplorer, flattenExplorerNodes, modelExplorerItemsFromLayers } from '../core/modelRootedExplorer';
import type { ExplorerNode } from '../domain/explorer';
import { MODEL_LAYER_TYPES, type ModelLayerType } from '../domain/model';
import type { OperatorId } from '../domain/operator';
import { useGraphStore } from '../store/graphStore';
import { SourcePill } from './WorkspaceChrome';

const NODE_ICONS: Record<ExplorerNode['kind'], string> = {
  'model-root': 'M',
  section: '▾',
  'model-module': '▦',
  'model-layer': '◇',
  operator: 'O',
  'graph-document': 'G',
  'model-output': '↗',
  representation: '⌁',
  'graph-node': 'G',
  'property-group': 'P',
  property: '·',
  'transformation-group': '↻',
  'rewrite-candidate': '∆',
  'transformation-attempt': 'A',
  evidence: '✓',
  kernel: 'K',
  'kernel-detail': '·',
  'runtime-event': '▶',
  'runtime-detail': '·',
  'hardware-metric': '▥',
};

function TreeNode({
  node,
  depth,
  expandedIds,
  selectedId,
  onToggle,
  onSelect,
  onAction,
  onDelete,
  canDeleteGraph,
}: {
  node: ExplorerNode;
  depth: number;
  expandedIds: ReadonlySet<string>;
  selectedId: string | null;
  onToggle: (id: string, expanded: boolean) => void;
  onSelect: (node: ExplorerNode) => void;
  onAction: (node: ExplorerNode) => void;
  onDelete: (node: ExplorerNode) => void;
  canDeleteGraph: boolean;
}) {
  const deletable = ['model-layer', 'operator'].includes(node.kind);
  const deleteDisabled = node.kind === 'graph-document' && !canDeleteGraph;
  const deleteTitle = deleteDisabled ? 'Keep at least one graph. Add another graph before deleting this one.'
    : node.kind === 'model-layer' ? 'Delete layer and exclusive operators; preserve shared operators'
    : node.kind === 'model-output' ? 'Remove output; keep the operator'
    : node.kind === 'graph-document' ? 'Delete graph and its operators'
    : 'Delete operator and connected edges';
  const expandable = Boolean(node.children?.length);
  const expanded = expandable && expandedIds.has(node.id);
  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (!expandable) return;
    if (event.key === 'ArrowRight' && !expanded) {
      event.preventDefault();
      onToggle(node.id, true);
    }
    if (event.key === 'ArrowLeft' && expanded) {
      event.preventDefault();
      onToggle(node.id, false);
    }
  };
  const sharedDescription = node.shared && node.sharedReferenceCount
    ? `Shared by ${node.sharedReferenceCount} model layers`
    : undefined;
  const orderDescription = node.orderSource === 'derived' && node.order !== undefined
    ? `Derived layer order ${node.order + 1}`
    : node.orderSource === 'unknown' ? 'Layer order is unknown because dependencies do not define one sequence' : undefined;
  const title = [node.label, node.description, orderDescription, sharedDescription].filter(Boolean).join('\n');
  return (
    <div className="cross-layer-tree__node">
      <div className={`cross-layer-tree__row ${node.action ? 'has-action' : ''} ${deletable ? 'has-delete' : ''} ${node.shared ? 'is-shared' : ''} ${selectedId === node.id ? 'is-selected' : ''}`} style={{ '--tree-depth': depth } as CSSProperties}>
        {expandable ? (
          <button type="button" className="cross-layer-tree__chevron" aria-label={`${node.label} ${expanded ? 'collapse' : 'expand'}`} aria-expanded={expanded} onClick={() => onToggle(node.id, !expanded)}>{expanded ? '⌄' : '›'}</button>
        ) : <span className="cross-layer-tree__chevron-placeholder" />}
        <button
          type="button"
          role="treeitem"
          aria-level={depth + 1}
          aria-expanded={expandable ? expanded : undefined}
          data-explorer-kind={node.kind}
          data-layer-order={node.order}
          data-order-source={node.orderSource}
          className="cross-layer-tree__label"
          title={title}
          onClick={() => onSelect(node)}
          onKeyDown={onKeyDown}
        >
          <span className={`cross-layer-tree__icon kind--${node.kind}`}>{NODE_ICONS[node.kind]}</span>
          {node.orderSource && <span className={`cross-layer-tree__order order--${node.orderSource}`}>{node.order === undefined ? '??' : String(node.order + 1).padStart(2, '0')}</span>}
          <span className="cross-layer-tree__text">{node.label}</span>
          <span className="cross-layer-tree__metadata">
            {node.badge && <span className="cross-layer-tree__badge">{node.badge}</span>}
            {node.source && <SourcePill source={node.source} />}
            {sharedDescription && <span className="cross-layer-tree__shared-badge" title={sharedDescription}>SHARED · {node.sharedReferenceCount}</span>}
          </span>
          {node.description && <span className="visually-hidden">{node.description}</span>}
        </button>
        {node.action && <button type="button" className="cross-layer-tree__add" aria-label={`+ ${node.action === 'add-layer' ? 'Add Layer' : node.action === 'add-operator' ? 'Add Operator' : node.action === 'add-output' ? 'Add Output' : 'Add Graph'}`} title={`Add ${node.label.toLowerCase()}`} onClick={() => onAction(node)}>+</button>}
        {deletable && <button type="button" className="cross-layer-tree__delete" aria-label={`Delete ${node.kind === 'operator' ? 'operator ' : ''}${node.label}`} title={deleteTitle} disabled={deleteDisabled} onClick={() => onDelete(node)}><svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><path d="M2 4h12M6 4V2h4v2M4 4l1 10h6l1-10M6.5 6v5M9.5 6v5" /></svg></button>}
      </div>
      {expanded && <div className="cross-layer-tree__children" role="group">{node.children?.map((child) => (
        <TreeNode key={child.id} node={child} depth={depth + 1} expandedIds={expandedIds} selectedId={selectedId} onToggle={onToggle} onSelect={onSelect} onAction={onAction} onDelete={onDelete} canDeleteGraph={canDeleteGraph} />
      ))}</div>}
    </div>
  );
}

export function ModelRootedExplorer() {
  const store = useGraphStore();
  const [composer, setComposer] = useState<'layer' | string | null>(null);
  const [layerType, setLayerType] = useState<ModelLayerType>('Linear');
  const [bias, setBias] = useState(true);
  const [operatorType, setOperatorType] = useState<OperatorId>('add');
  const modelItems = useMemo(() => modelExplorerItemsFromLayers(store.modelDocument.layers, store.document.graph), [store.modelDocument.layers, store.document.graph]);
  const tree = useMemo(() => buildModelRootedExplorer({
    graph: store.document.graph, modelId: store.modelDocument.id, modelName: store.modelDocument.name,
    modelItems, graphDocuments: store.graphDocuments,
    rewriteCandidates: store.rewriteCandidates, transformationAttempts: store.transformationAttempts,
  }), [store.document.graph, store.graphDocuments, store.modelDocument, modelItems, store.rewriteCandidates, store.transformationAttempts]);
  const rootId = tree[0]?.id;
  const defaultExpandedIds = () => new Set(tree.flatMap((root) => [root.id, ...(root.children?.map(({ id }) => id) ?? [])]));
  const [expandedIds, setExpandedIds] = useState<Set<string>>(defaultExpandedIds);
  useEffect(() => { setExpandedIds(defaultExpandedIds()); setComposer(null); }, [rootId]);
  const setExpanded = (id: string, expanded: boolean) => setExpandedIds((current) => {
    const next = new Set(current);
    if (expanded) next.add(id); else next.delete(id);
    return next;
  });
  const layerSection = tree[0]?.children?.find(({ id }) => id.endsWith(':section:layers'));
  const composerLayer = store.modelDocument.layers.find(({ id }) => id === composer);
  const handleAction = (node: ExplorerNode) => {
    setExpanded(node.id, true);
    if (node.action === 'add-layer') setComposer('layer');
    else if (node.action === 'add-operator' && node.target?.modelItemId) {
      const layer = store.modelDocument.layers.find(({ id }) => id === node.target?.modelItemId);
      if (layer?.graphId && layer.graphId !== store.selectedGraphId) store.selectGraph(layer.graphId);
      setComposer(node.target.modelItemId);
    }
  };
  const createLayer = () => {
    const id = store.addLayer(layerType, { bias });
    const path = rootId + ':model-item:' + id;
    if (layerSection) setExpanded(layerSection.id, true);
    setExpanded(path, true);
    setExpanded(path + ':operators', true);
    setComposer(null);
  };
  const select = (node: ExplorerNode) => {
    setComposer(null);
    if (node.target?.graphId && node.target.graphId !== store.selectedGraphId) store.selectGraph(node.target.graphId);
    if (node.kind === 'graph-document' && node.target?.graphId) store.selectGraph(node.target.graphId, node.id);
    else store.selectExplorerTarget(node.id, node.workspace, node.target);
  };
  return (
    <aside className="panel workspace-palette persistent-sidebar">
      <div className="panel-heading"><span>01</span><div><strong>MODEL-ROOTED EXPLORER</strong><small>model → layer → operators</small></div></div>
      <div className="sidebar-mode-tabs" role="tablist" aria-label="Sidebar modes">
        <button type="button" role="tab" aria-selected="true" className="is-active" onClick={() => store.setSidebarMode('explorer')}>Explorer</button>
      </div>
      <div className={`cross-layer-explorer-shell ${composer ? 'has-layer-composer' : ''}`}>
        <div className="cross-layer-explorer__actions"><button type="button" onClick={() => setExpandedIds(new Set(flattenExplorerNodes(tree).filter(({ children }) => children?.length).map(({ id }) => id)))}>Expand all</button><button type="button" onClick={() => setExpandedIds(new Set())}>Collapse all</button></div>
        {composer === 'layer' && <div className="explorer-layer-composer">
          <select aria-label="New layer type" value={layerType} onChange={(event) => setLayerType(event.target.value as ModelLayerType)}>{MODEL_LAYER_TYPES.map((type) => <option key={type}>{type}</option>)}</select>
          {layerType === 'Linear' && <label className="explorer-bias-option"><input type="checkbox" checked={bias} onChange={(event) => setBias(event.target.checked)} />Bias</label>}
          <button type="button" onClick={createLayer}>Create</button><button type="button" aria-label="Cancel layer creation" onClick={() => setComposer(null)}>×</button>
        </div>}
        {composerLayer && <div className="explorer-layer-composer" aria-label={`Add operator to ${composerLayer.name}`}>
          <select aria-label="New operator type" value={operatorType} onChange={(event) => setOperatorType(event.target.value as OperatorId)}>{OPERATORS.map((operator) => <option key={operator.id} value={operator.id}>{operator.name}</option>)}</select>
          <button type="button" onClick={() => { store.setActiveWorkspace('graph'); store.addOperator(operatorType, undefined, composerLayer.id); setComposer(null); }}>Create</button>
          <button type="button" aria-label="Cancel operator creation" onClick={() => setComposer(null)}>×</button>
        </div>}
        <div className="cross-layer-explorer" role="tree" aria-label="Model-rooted cross-layer explorer">
          {tree.map((node) => <TreeNode key={node.id} node={node} depth={0} expandedIds={expandedIds} selectedId={store.selectedExplorerNodeId} onToggle={setExpanded} onAction={handleAction} canDeleteGraph={false} onSelect={select} onDelete={(selected) => {
            if (selected.kind === 'model-layer' && selected.target?.modelItemId) { store.deleteLayer(selected.target.modelItemId); setComposer(null); }
            else if (selected.kind === 'operator' && selected.target?.graphNodeId) store.deleteNode(selected.target.graphNodeId);
          }} />)}
          {layerSection?.children?.length === 0 && <p className="cross-layer-explorer__empty">No layers yet. Use + Add Layer to create an operator composition.</p>}
        </div>
      </div>
    </aside>
  );
}
