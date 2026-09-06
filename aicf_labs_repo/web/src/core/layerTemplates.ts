import type { InputPortId } from '../domain/graph';
import type { ModelLayerType } from '../domain/model';
import type { OperatorId } from '../domain/operator';

export interface LayerTemplate {
  operatorTemplates: readonly OperatorId[];
  edges: readonly { source: number; target: number; targetPort: InputPortId }[];
}

export function getLayerTemplate(type: ModelLayerType, bias = true): LayerTemplate {
  switch (type) {
    case 'Linear': return bias
      ? { operatorTemplates: ['matmul', 'add'], edges: [{ source: 0, target: 1, targetPort: 'in-0' }] }
      : { operatorTemplates: ['matmul'], edges: [] };
    case 'ReLU': return { operatorTemplates: ['relu'], edges: [] };
    case 'Scale': return { operatorTemplates: ['mul'], edges: [] };
    case 'Reduction': return { operatorTemplates: ['reduceSum'], edges: [] };
    case 'Custom': return { operatorTemplates: [], edges: [] };
  }
}
