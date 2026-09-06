export const WORKSPACE_IDS = ['model', 'graph', 'kernel', 'runtime', 'hardware'] as const;

export type WorkspaceId = typeof WORKSPACE_IDS[number];

export interface WorkspaceDefinition {
  id: WorkspaceId;
  label: string;
  shortLabel: string;
  purpose: string;
}

export const WORKSPACE_DEFINITIONS: readonly WorkspaceDefinition[] = [
  { id: 'model', label: 'Model', shortLabel: '01', purpose: 'Model structure and definition' },
  { id: 'graph', label: 'Graph', shortLabel: '02', purpose: 'Operator graph and semantic transformations' },
  { id: 'kernel', label: 'Kernel', shortLabel: '03', purpose: 'Lowering and GPU kernel implementation' },
  { id: 'runtime', label: 'Runtime', shortLabel: '04', purpose: 'Launch order, streams, buffers, and events' },
  { id: 'hardware', label: 'Hardware', shortLabel: '05', purpose: 'GPU counters and measured performance' },
] as const;

export const WORKSPACE_LABELS = Object.fromEntries(
  WORKSPACE_DEFINITIONS.map(({ id, label }) => [id, label]),
) as Record<WorkspaceId, string>;

export type ObservationSource = 'real' | 'derived' | 'user-defined' | 'estimated' | 'placeholder';

export type LayerObservationStatus =
  | 'valid'
  | 'applied'
  | 'rejected'
  | 'changed'
  | 'embedded'
  | 'improved'
  | 'warning'
  | 'unknown';

export interface LayerEvidence {
  label: string;
  value: string;
}

export interface LayerMetric {
  label: string;
  value: string;
  delta?: string;
}

interface LayerObservationBase {
  id: string;
  workspace: WorkspaceId;
  category: string;
  title: string;
  summary?: string;
  status?: LayerObservationStatus;
  evidence?: readonly LayerEvidence[];
  metrics?: readonly LayerMetric[];
  relatedNodeIds?: readonly string[];
  relatedTransformationAttemptIds?: readonly string[];
}

export type LayerObservation =
  | (LayerObservationBase & {
      source: 'real';
      provenance: 'graph-document' | 'rewrite-engine';
    })
  | (LayerObservationBase & {
      source: 'derived';
      derivedFrom: 'graph-structure' | 'semantic-context';
    })
  | (LayerObservationBase & {
      source: 'user-defined';
      definedBy: 'model-editor';
    })
  | (LayerObservationBase & {
      source: 'estimated';
      estimateBasis: string;
    })
  | (LayerObservationBase & {
      source: 'placeholder';
      placeholderReason: string;
    });

export interface ProbeContext {
  graphNodeIds: string[];
  transformationAttemptId?: string;
  kernelId?: string;
  runtimeEventId?: string;
  metricId?: string;
}

export type BottomPanelSize = 'collapsed' | 'compact' | 'expanded';

export type WorkspaceBottomTab =
  | 'summary' | 'structure' | 'shapes' | 'parameters'
  | 'trace' | 'graph-details' | 'performance' | 'artifacts'
  | 'kernel-summary' | 'lowering'
  | 'timeline' | 'launches' | 'buffers' | 'cuda-graph'
  | 'metrics' | 'counters' | 'comparison' | 'evidence';

export type WorkspaceBottomTabs = Record<WorkspaceId, WorkspaceBottomTab>;

export interface TransformationTraceStep {
  id: string;
  observationId: string;
  workspace: WorkspaceId;
  title: string;
  summary: string;
  status: LayerObservationStatus;
  source: ObservationSource;
}
