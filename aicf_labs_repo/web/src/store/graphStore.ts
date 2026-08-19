import { create } from 'zustand';
import { getOperator } from '../catalog/operators';
import { parseGraphDocument, type DocumentParseResult } from '../core/documentCodec';
import { findRewriteCandidates, graphFingerprint } from '../core/rewriteEngine';
import { canConnect, validateGraph, type ConnectCheck, type ConnectRequest } from '../core/validateGraph';
import type { Graph, GraphDocument, GraphPosition, InputPortId, NodeParameters } from '../domain/graph';
import type { OperatorId } from '../domain/operator';
import type { RewriteCandidate } from '../domain/rewrite';
import type { ValidationResult } from '../domain/validation';
import { createEmptyDocument, getExample } from '../examples/graphExamples';

function cloneDocument(document: GraphDocument): GraphDocument {
  return structuredClone(document);
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

function derive(document: GraphDocument, visitedFingerprints: readonly string[]) {
  const validation = validateGraph(document.graph);
  const excluded = new Set(visitedFingerprints);
  const rewriteCandidates = findRewriteCandidates(document.graph)
    .filter(({ graph }) => !excluded.has(graphFingerprint(graph)));
  return { validation, rewriteCandidates };
}

export interface GraphStoreState {
  document: GraphDocument;
  documentName: string;
  selectedNodeId: string | null;
  selectedRewriteCandidateId: string | null;
  validation: ValidationResult;
  rewriteCandidates: RewriteCandidate[];
  visitedFingerprints: string[];
  connectionError: string | null;
  documentError: string | null;
  addOperator: (operatorId: OperatorId, position?: GraphPosition) => string;
  moveNode: (nodeId: string, position: GraphPosition) => void;
  deleteNode: (nodeId: string) => void;
  connectEdge: (request: ConnectRequest) => ConnectCheck;
  deleteEdge: (edgeId: string) => void;
  selectNode: (nodeId: string | null) => void;
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
  return {
    document,
    documentName: document.graph.name,
    selectedNodeId: null,
    selectedRewriteCandidateId: null,
    visitedFingerprints: [] as string[],
    connectionError: null,
    documentError: null,
    ...derive(document, []),
  };
}

export const useGraphStore = create<GraphStoreState>((set, get) => ({
  ...initialState(),
  addOperator(operatorId, position) {
    if (!getOperator(operatorId)) throw new Error(`Unknown operator: ${operatorId}`);
    const current = get();
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
      selectedNodeId: id,
      selectedRewriteCandidateId: null,
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
    set({ document: nextDocument });
  },
  deleteNode(nodeId) {
    const current = get();
    if (!current.document.graph.nodes.some(({ id }) => id === nodeId)) return;
    const nextDocument = cloneDocument(current.document);
    nextDocument.graph.nodes = nextDocument.graph.nodes.filter(({ id }) => id !== nodeId);
    nextDocument.graph.edges = nextDocument.graph.edges.filter(({ sourceNodeId, targetNodeId }) => sourceNodeId !== nodeId && targetNodeId !== nodeId);
    nextDocument.graph.outputs = nextDocument.graph.outputs.filter((id) => id !== nodeId);
    if (nextDocument.graph.outputs.length === 0 && nextDocument.graph.nodes.length > 0) {
      nextDocument.graph.outputs = inferSinkOutputs(nextDocument.graph);
    }
    delete nextDocument.layout.positions[nodeId];
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      selectedNodeId: current.selectedNodeId === nodeId ? null : current.selectedNodeId,
      selectedRewriteCandidateId: null,
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
      selectedRewriteCandidateId: null,
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
    nextDocument.graph.edges = nextDocument.graph.edges.filter(({ id }) => id !== edgeId);
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      selectedRewriteCandidateId: null,
      visitedFingerprints: visited,
      connectionError: null,
      ...derive(nextDocument, visited),
    });
  },
  selectNode(nodeId) {
    set({ selectedNodeId: nodeId, selectedRewriteCandidateId: null });
  },
  toggleOutput(nodeId) {
    const current = get();
    if (!current.document.graph.nodes.some(({ id }) => id === nodeId)) return;
    const nextDocument = cloneDocument(current.document);
    nextDocument.graph.outputs = nextDocument.graph.outputs.includes(nodeId)
      ? nextDocument.graph.outputs.filter((id) => id !== nodeId)
      : [...nextDocument.graph.outputs, nodeId];
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      selectedRewriteCandidateId: null,
      visitedFingerprints: visited,
      ...derive(nextDocument, visited),
    });
  },
  replaceDocument(document, documentName = document.graph.name) {
    const nextDocument = cloneDocument(document);
    const visited = [graphFingerprint(nextDocument.graph)];
    set({
      document: nextDocument,
      documentName,
      selectedNodeId: null,
      selectedRewriteCandidateId: null,
      visitedFingerprints: visited,
      connectionError: null,
      documentError: null,
      ...derive(nextDocument, visited),
    });
  },
  loadExample(exampleId) {
    const example = getExample(exampleId);
    if (!example) return false;
    get().replaceDocument(example.document, example.label);
    return true;
  },
  selectRewriteCandidate(candidateId) {
    set({
      selectedRewriteCandidateId: candidateId && get().rewriteCandidates.some(({ id }) => id === candidateId)
        ? candidateId
        : null,
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
      selectedNodeId: current.selectedNodeId && retainedIds.has(current.selectedNodeId) ? current.selectedNodeId : null,
      selectedRewriteCandidateId: null,
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
