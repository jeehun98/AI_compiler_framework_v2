import { create } from 'zustand';
import { getOperator } from '../catalog/operators';
import { parseGraphDocument, type DocumentParseResult } from '../core/documentCodec';
import { graphFingerprint, searchRewriteCandidates } from '../core/rewriteEngine';
import { createDerivedModelDocument, createEmptyGraphDocument, createUserDefinedLayer } from '../core/modelDocument';
import { getLayerTemplate } from '../core/layerTemplates';
import { canConnect, validateGraph, type ConnectCheck, type ConnectRequest } from '../core/validateGraph';
import type { Graph, GraphDocument, GraphPosition, InputPortId, NodeParameters } from '../domain/graph';
import type { ExplorerTarget, SidebarMode } from '../domain/explorer';
import type {
  BottomPanelSize,
  ProbeContext,
  WorkspaceBottomTab,
  WorkspaceBottomTabs,
  WorkspaceId,
} from '../domain/layerObservation';
import type { OperatorId } from '../domain/operator';
import type { ModelDocument, ModelLayerType } from '../domain/model';
import type { RewriteCandidate, TransformationAttempt } from '../domain/rewrite';
import type { ValidationResult } from '../domain/validation';
import { createEmptyDocument } from '../examples/graphExamples';
import { getModelExample } from '../examples/modelExamples';

function cloneDocument(document: GraphDocument): GraphDocument {
  return structuredClone(document);
}

function replaceActiveGraph(graphDocuments: readonly GraphDocument[], selectedGraphId: string, document: GraphDocument): GraphDocument[] {
  return graphDocuments.map((candidate) => candidate.graph.id === selectedGraphId ? cloneDocument(document) : candidate);
}

function defaultParameters(operatorId: OperatorId, id: string): NodeParameters {
  switch (operatorId) {
    case 'input': return { symbol: id === 'input-1' ? 'x' : id.replace(/[^A-Za-z0-9]/g, ''), shape: [] };
    case 'constant': return { value: 1 };
    case 'reduceSum': return { axis: 'all', keepDims: false };
    default: return {};
  }
}

function nextNodeId(graph: Graph, operatorId: OperatorId): string {
  const occupied = new Set(graph.nodes.map(({ id }) => id));
  let index = 1;
  while (occupied.has(`${operatorId}-${index}`)) index += 1;
  return `${operatorId}-${index}`;
}

function nextEdgeId(graph: Graph, request: ConnectRequest): string {
  const base = `${request.sourceNodeId}-${request.targetNodeId}-${request.targetPort}`;
  const occupied = new Set(graph.edges.map(({ id }) => id));
  if (!occupied.has(base)) return base;
  let index = 2;
  while (occupied.has(`${base}-${index}`)) index += 1;
  return `${base}-${index}`;
}

function inferSinkOutputs(graph: Graph): string[] {
  const sources = new Set(graph.edges.map(({ sourceNodeId }) => sourceNodeId));
  return graph.nodes.map(({ id }) => id).filter((id) => !sources.has(id));
}

function invalidateSemanticRegions(graph: Graph): void {
  delete graph.semanticRegions;
}

function removeDocumentNodes(document: GraphDocument, ids: ReadonlySet<string>): void {
  invalidateSemanticRegions(document.graph);
  document.graph.nodes = document.graph.nodes.filter(({ id }) => !ids.has(id));
  document.graph.edges = document.graph.edges.filter(({ sourceNodeId, targetNodeId }) => !ids.has(sourceNodeId) && !ids.has(targetNodeId));
  document.graph.outputs = document.graph.outputs.filter((id) => !ids.has(id));
  if (document.graph.outputs.length === 0 && document.graph.nodes.length > 0) document.graph.outputs = inferSinkOutputs(document.graph);
  for (const id of ids) delete document.layout.positions[id];
}

function derive(document: GraphDocument, visitedFingerprints: readonly string[]) {
  const validation = validateGraph(document.graph);
  const excluded = new Set(visitedFingerprints);
  const search = searchRewriteCandidates(document.graph);
  const rewriteCandidates = search.candidates
    .filter(({ graph }) => !excluded.has(graphFingerprint(graph)));
  return { validation, rewriteCandidates, transformationAttempts: search.attempts };
}

function attemptGraphNodeIdsForStore(attempt: TransformationAttempt, graph: Graph): string[] {
  const graphNodeIds = new Set(graph.nodes.map(({ id }) => id));
  return [...new Set(Object.values(attempt.bindings).filter((id) => graphNodeIds.has(id)))];
}

export interface GraphStoreState {
  document: GraphDocument;
  documentName: string;
  modelDocument: ModelDocument;
  graphDocuments: GraphDocument[];
  selectedModelId: string;
  selectedGraphId: string;
  selectedNodeId: string | null;
  selectedRewriteCandidateId: string | null;
  selectedTransformationAttemptId: string | null;
  selectedLayerObservationId: string | null;
  selectedExplorerNodeId: string | null;
  sidebarMode: SidebarMode;
  activeWorkspace: WorkspaceId;
  selectedModelItemId: string | null;
  selectedKernelId: string | null;
  selectedRuntimeEventId: string | null;
  selectedHardwareMetricId: string | null;
  probeContext: ProbeContext;
  bottomPanelSize: BottomPanelSize;
  bottomPanelTabs: WorkspaceBottomTabs;
  validation: ValidationResult;
  rewriteCandidates: RewriteCandidate[];
  transformationAttempts: TransformationAttempt[];
  visitedFingerprints: string[];
  connectionError: string | null;
  documentError: string | null;
  addOperator: (operatorId: OperatorId, position?: GraphPosition, layerId?: string) => string;
  addLayer: (type: ModelLayerType, options?: { bias?: boolean }) => string;
  deleteLayer: (layerId: string) => void;
  addGraph: (name?: string) => string;
  deleteGraph: (graphId: string) => boolean;
  selectGraph: (graphId: string, explorerNodeId?: string) => boolean;
  moveNode: (nodeId: string, position: GraphPosition) => void;
  deleteNode: (nodeId: string) => void;
  connectEdge: (request: ConnectRequest) => ConnectCheck;
  deleteEdge: (edgeId: string) => void;
  selectNode: (nodeId: string | null) => void;
  setSidebarMode: (mode: SidebarMode) => void;
  selectExplorerTarget: (nodeId: string, workspace?: WorkspaceId, target?: ExplorerTarget) => void;
  setActiveWorkspace: (workspace: WorkspaceId) => void;
  selectLayerObservation: (observationId: string | null, workspace?: WorkspaceId) => void;
  selectTransformationAttempt: (attemptId: string | null) => void;
  selectModelItem: (itemId: string | null) => void;
  selectKernel: (kernelId: string | null) => void;
  selectRuntimeEvent: (eventId: string | null) => void;
  selectHardwareMetric: (metricId: string | null) => void;
  setWorkspaceBottomTab: (workspace: WorkspaceId, tab: WorkspaceBottomTab) => void;
  setBottomPanelSize: (size: BottomPanelSize) => void;
  toggleOutput: (nodeId: string) => void;
  replaceDocument: (document: GraphDocument, documentName?: string) => void;
  loadExample: (exampleId: string) => boolean;
  selectRewriteCandidate: (candidateId: string | null) => void;
  applySelectedRewrite: () => boolean;
  importJson: (json: string) => DocumentParseResult;
  clearMessages: () => void;
  reset: () => void;
}

function initialState() {
  const document = createEmptyDocument();
  const modelDocument = createDerivedModelDocument(document);
  return {
    document,
    documentName: modelDocument.name,
    modelDocument,
    graphDocuments: [cloneDocument(document)],
    selectedModelId: modelDocument.id,
    selectedGraphId: document.graph.id,
    selectedNodeId: null,
    selectedRewriteCandidateId: null,
    selectedTransformationAttemptId: null,
    selectedLayerObservationId: null,
    selectedExplorerNodeId: null,
    sidebarMode: 'explorer' as const,
    activeWorkspace: 'graph' as const,
    selectedModelItemId: null,
    selectedKernelId: null,
    selectedRuntimeEventId: null,
    selectedHardwareMetricId: null,
    probeContext: { graphNodeIds: [] } as ProbeContext,
    bottomPanelSize: 'compact' as const,
    bottomPanelTabs: {
      model: 'summary', graph: 'trace', kernel: 'kernel-summary', runtime: 'timeline', hardware: 'metrics',
    } as WorkspaceBottomTabs,
    visitedFingerprints: [] as string[],
    connectionError: null,
    documentError: null,
    ...derive(document, []),
  };
}

export const useGraphStore = create<GraphStoreState>((set, get) => ({
  ...initialState(),
  addLayer(type, options) {
    const current = get();
    const layer = createUserDefinedLayer(current.modelDocument, type);
    const document = cloneDocument(current.document);
    const template = getLayerTemplate(type, options?.bias);
    layer.graphId = document.graph.id;
    for (const operatorId of template.operatorTemplates) {
      const id = nextNodeId(document.graph, operatorId);
      document.graph.nodes.push({ id, operatorId, parameters: defaultParameters(operatorId, id) });
      document.layout.positions[id] = { x: 60 + layer.graphNodeIds.length * 200, y: 80 + current.modelDocument.layers.length * 120 };
      layer.graphNodeIds.push(id);
    }
    for (const edge of template.edges) {
      const request = { sourceNodeId: layer.graphNodeIds[edge.source], targetNodeId: layer.graphNodeIds[edge.target], targetPort: edge.targetPort };
      document.graph.edges.push({ id: nextEdgeId(document.graph, request), ...request, sourcePort: 'out' });
    }
    if (layer.graphNodeIds.length) {
      invalidateSemanticRegions(document.graph);
      if (!document.graph.outputs.length) document.graph.outputs = [layer.graphNodeIds[layer.graphNodeIds.length - 1]];
    }
    const visited = [graphFingerprint(document.graph)];
    set({
      document,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, document),
      modelDocument: { ...current.modelDocument, layers: [...current.modelDocument.layers, layer] },
      selectedModelItemId: layer.id,
      selectedExplorerNodeId: null,
      activeWorkspace: 'model',
      sidebarMode: 'explorer',
      selectedNodeId: null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedKernelId: null, selectedRuntimeEventId: null, selectedHardwareMetricId: null,
      probeContext: { graphNodeIds: [...layer.graphNodeIds] },
      visitedFingerprints: visited,
      connectionError: null,
      ...derive(document, visited),
    });
    return layer.id;
  },
  deleteLayer(layerId) {
    const current = get();
    const layer = current.modelDocument.layers.find(({ id }) => id === layerId);
    if (!layer) return;
    const graphId = layer.graphId ?? current.selectedGraphId;
    const layers = current.modelDocument.layers.filter(({ id }) => id !== layerId);
    const sharedIds = new Set(layers.filter((item) => !item.graphId || item.graphId === graphId).flatMap(({ graphNodeIds }) => graphNodeIds));
    const exclusiveIds = new Set(layer.graphNodeIds.filter((id) => !sharedIds.has(id)));
    const graphDocuments = current.graphDocuments.map((document) => {
      if (document.graph.id !== graphId || !exclusiveIds.size) return document;
      const next = cloneDocument(document);
      removeDocumentNodes(next, exclusiveIds);
      return next;
    });
    const document = graphDocuments.find(({ graph }) => graph.id === current.selectedGraphId)!;
    const visited = [graphFingerprint(document.graph)];
    set({
      document, graphDocuments,
      modelDocument: { ...current.modelDocument, layers },
      selectedModelItemId: current.selectedModelItemId === layerId ? null : current.selectedModelItemId,
      selectedExplorerNodeId: null,
      selectedLayerObservationId: null,
      probeContext: { graphNodeIds: [] },
      selectedNodeId: current.selectedGraphId === graphId && current.selectedNodeId && exclusiveIds.has(current.selectedNodeId) ? null : current.selectedNodeId,
      selectedRewriteCandidateId: null, selectedTransformationAttemptId: null,
      selectedKernelId: null, selectedRuntimeEventId: null, selectedHardwareMetricId: null,
      visitedFingerprints: visited,
      ...derive(document, visited),
    });
  },
  deleteGraph(graphId) {
    const current = get();
    const removed = current.graphDocuments.find(({ graph }) => graph.id === graphId);
    // The canvas always needs one active document.
    if (!removed || current.graphDocuments.length <= 1) return false;
    const graphDocuments = current.graphDocuments.filter(({ graph }) => graph.id !== graphId);
    const removedNodeIds = new Set(removed.graph.nodes.map(({ id }) => id));
    const remainingNodeIds = new Set(graphDocuments.flatMap(({ graph }) => graph.nodes.map(({ id }) => id)));
    set({
      graphDocuments,
      modelDocument: {
        ...current.modelDocument,
        graphIds: current.modelDocument.graphIds.filter((id) => id !== graphId),
        layers: current.modelDocument.layers.map((layer) => ({
          ...layer,
          graphNodeIds: layer.graphId ? (layer.graphId === graphId ? [] : layer.graphNodeIds) : layer.graphNodeIds.filter((id) => !removedNodeIds.has(id) || remainingNodeIds.has(id)),
        })),
      },
    });
    if (current.selectedGraphId === graphId) get().selectGraph(graphDocuments[0].graph.id);
    return true;
  },
  addGraph(name) {
    const current = get();
    const document = createEmptyGraphDocument(current.modelDocument, name);
    const graphDocuments = [...current.graphDocuments, cloneDocument(document)];
    set({
      document,
      graphDocuments,
      modelDocument: { ...current.modelDocument, graphIds: [...current.modelDocument.graphIds, document.graph.id] },
      selectedGraphId: document.graph.id,
      selectedNodeId: null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: `graph-document:${document.graph.id}`,
      activeWorkspace: 'graph',
      sidebarMode: 'explorer',
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: [],
      connectionError: null,
      documentError: null,
      ...derive(document, []),
    });
    return document.graph.id;
  },
  selectGraph(graphId, explorerNodeId) {
    const current = get();
    const document = current.graphDocuments.find(({ graph }) => graph.id === graphId);
    if (!document) return false;
    const nextDocument = cloneDocument(document);
    set({
      document: nextDocument,
      selectedGraphId: graphId,
      selectedNodeId: null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: explorerNodeId ?? null,
      selectedKernelId: null,
      selectedRuntimeEventId: null,
      selectedHardwareMetricId: null,
      activeWorkspace: 'graph',
      sidebarMode: 'explorer',
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: [graphFingerprint(nextDocument.graph)],
      connectionError: null,
      documentError: null,
      ...derive(nextDocument, [graphFingerprint(nextDocument.graph)]),
    });
    return true;
  },
  addOperator(operatorId, position, layerId) {
    if (!getOperator(operatorId)) throw new Error(`Unknown operator: ${operatorId}`);
    const current = get();
    const layer = layerId ? current.modelDocument.layers.find(({ id }) => id === layerId) : undefined;
    if (layerId && (!layer || (layer.graphId && layer.graphId !== current.selectedGraphId))) throw new Error('Select the layer graph before adding an operator.');
    const id = nextNodeId(current.document.graph, operatorId);
    const nextDocument = cloneDocument(current.document);
    nextDocument.graph.nodes.push({ id, operatorId, parameters: defaultParameters(operatorId, id) });
    nextDocument.layout.positions[id] = position ?? {
      x: 60 + (nextDocument.graph.nodes.length % 3) * 190,
      y: 70 + (nextDocument.graph.nodes.length % 4) * 95,
    };
    if (nextDocument.graph.outputs.length === 0) nextDocument.graph.outputs = [id];
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument),
      ...(layer ? { modelDocument: { ...current.modelDocument, layers: current.modelDocument.layers.map((item) => item.id === layer.id ? { ...item, graphId: current.selectedGraphId, graphNodeIds: [...item.graphNodeIds, id] } : item) } } : {}),
      selectedNodeId: id,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: [id] },
      visitedFingerprints: visited,
      connectionError: null,
      documentError: null,
      ...derive(nextDocument, visited),
    });
    return id;
  },
  moveNode(nodeId, position) {
    const current = get();
    if (!current.document.graph.nodes.some(({ id }) => id === nodeId)) return;
    const nextDocument = cloneDocument(current.document);
    nextDocument.layout.positions[nodeId] = { ...position };
    set({ document: nextDocument, graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument) });
  },
  deleteNode(nodeId) {
    const current = get();
    if (!current.document.graph.nodes.some(({ id }) => id === nodeId)) return;
    const nextDocument = cloneDocument(current.document);
    removeDocumentNodes(nextDocument, new Set([nodeId]));
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument),
      modelDocument: {
        ...current.modelDocument,
        layers: current.modelDocument.layers.map((layer) => ({ ...layer, graphNodeIds: layer.graphId && layer.graphId !== current.selectedGraphId ? layer.graphNodeIds : layer.graphNodeIds.filter((id) => id !== nodeId) })),
      },
      selectedNodeId: current.selectedNodeId === nodeId ? null : current.selectedNodeId,
      selectedKernelId: null,
      selectedRuntimeEventId: null,
      selectedHardwareMetricId: null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: visited,
      connectionError: null,
      ...derive(nextDocument, visited),
    });
  },
  connectEdge(request) {
    const current = get();
    const check = canConnect(current.document.graph, request);
    if (!check.allowed) {
      set({ connectionError: check.reason ?? '연결할 수 없습니다.' });
      return check;
    }
    const nextDocument = cloneDocument(current.document);
    invalidateSemanticRegions(nextDocument.graph);
    nextDocument.graph.edges.push({
      id: nextEdgeId(nextDocument.graph, request),
      sourceNodeId: request.sourceNodeId,
      sourcePort: 'out',
      targetNodeId: request.targetNodeId,
      targetPort: request.targetPort,
    });
    if (nextDocument.graph.outputs.includes(request.sourceNodeId)) {
      nextDocument.graph.outputs = [...new Set(nextDocument.graph.outputs
        .map((nodeId) => nodeId === request.sourceNodeId ? request.targetNodeId : nodeId))];
    } else if (nextDocument.graph.outputs.length === 0) {
      nextDocument.graph.outputs = [request.targetNodeId];
    }
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument),
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: visited,
      connectionError: null,
      ...derive(nextDocument, visited),
    });
    return { allowed: true };
  },
  deleteEdge(edgeId) {
    const current = get();
    if (!current.document.graph.edges.some(({ id }) => id === edgeId)) return;
    const nextDocument = cloneDocument(current.document);
    invalidateSemanticRegions(nextDocument.graph);
    nextDocument.graph.edges = nextDocument.graph.edges.filter(({ id }) => id !== edgeId);
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument),
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: visited,
      connectionError: null,
      ...derive(nextDocument, visited),
    });
  },
  selectNode(nodeId) {
    const current = get();
    set({
      selectedNodeId: nodeId,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: {
        ...current.probeContext,
        graphNodeIds: nodeId ? [nodeId] : [],
      },
    });
  },
  setActiveWorkspace(workspace) {
    set({ activeWorkspace: workspace, ...(workspace === 'graph' ? {} : { sidebarMode: 'explorer' as const }) });
  },
  setSidebarMode(mode) {
    set({ sidebarMode: mode === 'operators' && get().activeWorkspace !== 'graph' ? 'explorer' : mode });
  },
  selectExplorerTarget(nodeId, workspace, target = {}) {
    const current = get();
    const patch: Partial<GraphStoreState> = {
      selectedExplorerNodeId: nodeId,
      sidebarMode: 'explorer',
      ...(workspace ? { activeWorkspace: workspace } : {}),
    };
    const probeContext = { ...current.probeContext };
    if (target.modelItemId) patch.selectedModelItemId = target.modelItemId;
    const targetGraphNodeIds = target.graphNodeIds ? [...target.graphNodeIds] : target.graphNodeId ? [target.graphNodeId] : undefined;
    if (targetGraphNodeIds) {
      probeContext.graphNodeIds = targetGraphNodeIds;
      patch.selectedNodeId = target.graphNodeId ?? targetGraphNodeIds[0] ?? null;
    }
    if (target.rewriteCandidateId) {
      const candidate = current.rewriteCandidates.find(({ id }) => id === target.rewriteCandidateId);
      if (candidate) {
        patch.activeWorkspace = 'graph';
        patch.selectedRewriteCandidateId = candidate.id;
        patch.selectedTransformationAttemptId = candidate.attemptId;
        patch.selectedLayerObservationId = `semantic:candidate:${candidate.id}`;
        patch.selectedNodeId = target.graphNodeId ?? candidate.affectedNodeIds[0] ?? null;
        probeContext.graphNodeIds = [...candidate.affectedNodeIds];
        probeContext.transformationAttemptId = candidate.attemptId;
      }
    } else if (target.transformationAttemptId) {
      const attempt = current.transformationAttempts.find(({ id }) => id === target.transformationAttemptId);
      if (attempt) {
        const relatedNodeIds = attemptGraphNodeIdsForStore(attempt, current.document.graph);
        patch.activeWorkspace = 'graph';
        patch.selectedTransformationAttemptId = attempt.id;
        patch.selectedLayerObservationId = `semantic:attempt:${attempt.id}`;
        patch.selectedNodeId = target.graphNodeId ?? relatedNodeIds[0] ?? patch.selectedNodeId ?? null;
        probeContext.graphNodeIds = targetGraphNodeIds ?? relatedNodeIds;
        probeContext.transformationAttemptId = attempt.id;
      }
    }
    if (target.kernelId) {
      patch.selectedKernelId = target.kernelId;
      probeContext.kernelId = target.kernelId;
    }
    if (target.runtimeEventId) {
      patch.selectedRuntimeEventId = target.runtimeEventId;
      probeContext.runtimeEventId = target.runtimeEventId;
    }
    if (target.hardwareMetricId) {
      patch.selectedHardwareMetricId = target.hardwareMetricId;
      probeContext.metricId = target.hardwareMetricId;
    }
    patch.probeContext = probeContext;
    set(patch);
  },
  selectLayerObservation(observationId, workspace) {
    set({
      selectedLayerObservationId: observationId,
      selectedExplorerNodeId: null,
      ...(workspace ? { activeWorkspace: workspace } : {}),
    });
  },
  selectTransformationAttempt(attemptId) {
    const attempt = get().transformationAttempts.find(({ id }) => id === attemptId);
    const relatedNodeId = attempt
      ? Object.values(attempt.bindings).find((id) => get().document.graph.nodes.some((node) => node.id === id))
      : undefined;
    set({
      selectedTransformationAttemptId: attempt?.id ?? null,
      selectedLayerObservationId: attempt ? `semantic:attempt:${attempt.id}` : null,
      selectedExplorerNodeId: null,
      activeWorkspace: 'graph',
      probeContext: {
        graphNodeIds: relatedNodeId ? [relatedNodeId] : [],
        ...(attempt ? { transformationAttemptId: attempt.id } : {}),
      },
      ...(relatedNodeId ? { selectedNodeId: relatedNodeId } : {}),
    });
  },
  selectModelItem(itemId) {
    set({ selectedModelItemId: itemId, selectedExplorerNodeId: null });
  },
  selectKernel(kernelId) {
    const current = get();
    const probeContext = { ...current.probeContext };
    if (kernelId) probeContext.kernelId = kernelId;
    else delete probeContext.kernelId;
    set({ selectedKernelId: kernelId, selectedExplorerNodeId: null, probeContext });
  },
  selectRuntimeEvent(eventId) {
    const current = get();
    const probeContext = { ...current.probeContext };
    if (eventId) probeContext.runtimeEventId = eventId;
    else delete probeContext.runtimeEventId;
    set({ selectedRuntimeEventId: eventId, selectedExplorerNodeId: null, probeContext });
  },
  selectHardwareMetric(metricId) {
    const current = get();
    const probeContext = { ...current.probeContext };
    if (metricId) probeContext.metricId = metricId;
    else delete probeContext.metricId;
    set({ selectedHardwareMetricId: metricId, selectedExplorerNodeId: null, probeContext });
  },
  setWorkspaceBottomTab(workspace, tab) {
    const current = get();
    set({ bottomPanelTabs: { ...current.bottomPanelTabs, [workspace]: tab } });
  },
  setBottomPanelSize(size) {
    set({ bottomPanelSize: size });
  },
  toggleOutput(nodeId) {
    const current = get();
    if (!current.document.graph.nodes.some(({ id }) => id === nodeId)) return;
    const nextDocument = cloneDocument(current.document);
    invalidateSemanticRegions(nextDocument.graph);
    nextDocument.graph.outputs = nextDocument.graph.outputs.includes(nodeId)
      ? nextDocument.graph.outputs.filter((id) => id !== nodeId)
      : [...nextDocument.graph.outputs, nodeId];
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument),
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: visited,
      ...derive(nextDocument, visited),
    });
  },
  replaceDocument(document, documentName = document.graph.name) {
    const nextDocument = cloneDocument(document);
    const modelDocument = createDerivedModelDocument(nextDocument, documentName);
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      documentName: modelDocument.name,
      modelDocument,
      graphDocuments: [cloneDocument(nextDocument)],
      selectedModelId: modelDocument.id,
      selectedGraphId: nextDocument.graph.id,
      selectedNodeId: null,
      selectedModelItemId: null,
      selectedKernelId: null,
      selectedRuntimeEventId: null,
      selectedHardwareMetricId: null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: visited,
      connectionError: null,
      documentError: null,
      ...derive(nextDocument, visited),
    });
  },
  loadExample(exampleId) {
    const example = getModelExample(exampleId);
    if (!example) return false;
    const graphDocuments = example.graphs.map(cloneDocument);
    const document = graphDocuments.find(({ graph }) => graph.id === example.selectedGraphId);
    if (!document) return false;
    const modelDocument = structuredClone(example.model);
    const visited = [graphFingerprint(document.graph)];
    set({
      document,
      documentName: modelDocument.name,
      modelDocument,
      graphDocuments,
      selectedModelId: modelDocument.id,
      selectedGraphId: document.graph.id,
      selectedNodeId: null,
      selectedModelItemId: null,
      selectedKernelId: null,
      selectedRuntimeEventId: null,
      selectedHardwareMetricId: null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      activeWorkspace: 'graph',
      sidebarMode: 'explorer',
      probeContext: { graphNodeIds: [] },
      visitedFingerprints: visited,
      connectionError: null,
      documentError: null,
      ...derive(document, visited),
    });
    return true;
  },
  selectRewriteCandidate(candidateId) {
    const candidate = get().rewriteCandidates.find(({ id }) => id === candidateId);
    set({
      selectedRewriteCandidateId: candidate?.id ?? null,
      selectedTransformationAttemptId: candidate?.attemptId ?? null,
      selectedLayerObservationId: candidate ? `semantic:candidate:${candidate.id}` : null,
      selectedExplorerNodeId: null,
      activeWorkspace: 'graph',
      probeContext: {
        graphNodeIds: candidate ? [...candidate.affectedNodeIds] : [],
        ...(candidate ? { transformationAttemptId: candidate.attemptId } : {}),
      },
      ...(candidate?.affectedNodeIds[0] ? { selectedNodeId: candidate.affectedNodeIds[0] } : {}),
    });
  },
  applySelectedRewrite() {
    const current = get();
    const candidate = current.rewriteCandidates.find(({ id }) => id === current.selectedRewriteCandidateId);
    if (!candidate) return false;
    const nextDocument = cloneDocument(current.document);
    nextDocument.graph = structuredClone(candidate.graph);
    const retainedIds = new Set(nextDocument.graph.nodes.map(({ id }) => id));
    nextDocument.layout.positions = Object.fromEntries(Object.entries(nextDocument.layout.positions)
      .filter(([nodeId]) => retainedIds.has(nodeId)));
    for (const node of nextDocument.graph.nodes) {
      nextDocument.layout.positions[node.id] ??= { x: 80, y: 80 };
    }
    const visited = [...new Set([
      ...current.visitedFingerprints,
      graphFingerprint(current.document.graph),
      graphFingerprint(nextDocument.graph),
    ])];
    set({
      document: nextDocument,
      graphDocuments: replaceActiveGraph(current.graphDocuments, current.selectedGraphId, nextDocument),
      modelDocument: {
        ...current.modelDocument,
        layers: current.modelDocument.layers.map((layer) => ({ ...layer, graphNodeIds: layer.graphId && layer.graphId !== current.selectedGraphId ? layer.graphNodeIds : layer.graphNodeIds.filter((id) => retainedIds.has(id)) })),
      },
      selectedNodeId: current.selectedNodeId && retainedIds.has(current.selectedNodeId) ? current.selectedNodeId : null,
      selectedRewriteCandidateId: null,
      selectedTransformationAttemptId: null,
      selectedLayerObservationId: null,
      selectedExplorerNodeId: null,
      probeContext: { graphNodeIds: current.selectedNodeId && retainedIds.has(current.selectedNodeId) ? [current.selectedNodeId] : [] },
      visitedFingerprints: visited,
      connectionError: null,
      documentError: null,
      ...derive(nextDocument, visited),
    });
    return true;
  },
  importJson(json) {
    const parsed = parseGraphDocument(json);
    if (!parsed.ok) {
      set({ documentError: parsed.error });
      return parsed;
    }
    get().replaceDocument(parsed.value, parsed.value.graph.name);
    return parsed;
  },
  clearMessages() {
    set({ connectionError: null, documentError: null });
  },
  reset() {
    set(initialState());
  },
}));

export function toConnectRequest(source: string, target: string, targetPort: string): ConnectRequest | undefined {
  if (!source || !target || !targetPort) return undefined;
  return { sourceNodeId: source, targetNodeId: target, targetPort: targetPort as InputPortId };
}
