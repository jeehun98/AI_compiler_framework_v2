import type { GraphDocument } from './graph';
import type { ObservationSource } from './layerObservation';

export const MODEL_LAYER_TYPES = ['Linear', 'ReLU', 'Scale', 'Reduction', 'Custom'] as const;

export type ModelLayerType = typeof MODEL_LAYER_TYPES[number];
export type ModelLayerSource = Extract<ObservationSource, 'derived' | 'user-defined'>;

export interface ModelLayer {
  id: string;
  name: string;
  type: ModelLayerType;
  graphNodeIds: string[];
  graphId?: string;
  source: ModelLayerSource;
}

export interface ModelDocument {
  id: string;
  name: string;
  layers: ModelLayer[];
  graphIds: string[];
}

export interface ModelExample {
  id: string;
  label: string;
  model: ModelDocument;
  graphs: GraphDocument[];
  selectedGraphId: string;
}
