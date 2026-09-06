import type { ObservationSource, WorkspaceId } from './layerObservation';

export type SidebarMode = 'explorer' | 'operators';
export type ModelLayerOrderSource = 'derived' | 'unknown';

export type ExplorerNodeKind =
  | 'model-root'
  | 'section'
  | 'model-module'
  | 'model-layer'
  | 'operator'
  | 'graph-document'
  | 'model-output'
  | 'representation'
  | 'graph-node'
  | 'property-group'
  | 'property'
  | 'transformation-group'
  | 'rewrite-candidate'
  | 'transformation-attempt'
  | 'evidence'
  | 'kernel'
  | 'kernel-detail'
  | 'runtime-event'
  | 'runtime-detail'
  | 'hardware-metric';

export interface ExplorerTarget {
  modelItemId?: string;
  graphNodeId?: string;
  graphNodeIds?: readonly string[];
  rewriteCandidateId?: string;
  transformationAttemptId?: string;
  kernelId?: string;
  runtimeEventId?: string;
  hardwareMetricId?: string;
  graphId?: string;
}

export interface ModelRootSummary {
  layerCount: number;
  graphNodeCount: number;
  inputCount: number;
  outputCount: number;
}

/**
 * Read-only projection node for the cross-layer explorer. Domain objects remain
 * owned by GraphDocument, the semantic engine, and future backend adapters.
 */
export interface ExplorerNode {
  id: string;
  kind: ExplorerNodeKind;
  label: string;
  description?: string;
  workspace?: WorkspaceId;
  source?: ObservationSource;
  badge?: string;
  shared?: boolean;
  sharedReferenceCount?: number;
  order?: number;
  orderSource?: ModelLayerOrderSource;
  modelSummary?: ModelRootSummary;
  action?: 'add-layer' | 'add-operator' | 'add-graph' | 'add-output';
  target?: ExplorerTarget;
  children?: readonly ExplorerNode[];
}

/** Future frontend/model adapters can supply real nested modules through this shape. */
export interface ModelExplorerItem {
  id: string;
  kind: 'module' | 'layer' | 'input' | 'output' | 'parameter';
  label: string;
  source: Extract<ObservationSource, 'real' | 'derived' | 'user-defined'>;
  graphNodeIds: readonly string[];
  children?: readonly ModelExplorerItem[];
}
