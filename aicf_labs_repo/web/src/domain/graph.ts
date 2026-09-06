import type { OperatorId } from './operator';
import type { DependencyKind, MaterializationRequirement } from './semantic';

export type Dimension = number | string;
export type TensorShape = Dimension[];
export type InputPortId = `in-${number}`;

export interface InputParameters {
  symbol: string;
  shape: TensorShape;
}

export interface ConstantParameters {
  value: number;
}

export interface ReduceSumParameters {
  axis: number | 'all';
  keepDims: boolean;
}

export interface EmptyParameters {
  readonly [key: string]: never;
}

export type NodeParameters =
  | InputParameters
  | ConstantParameters
  | ReduceSumParameters
  | EmptyParameters;

export interface GraphNode {
  id: string;
  operatorId: OperatorId;
  parameters: NodeParameters;
}

export interface GraphEdge {
  id: string;
  sourceNodeId: string;
  sourcePort: 'out';
  targetNodeId: string;
  targetPort: InputPortId;
}

export interface ReductionInputFusionRegion {
  id: string;
  kind: 'reduction-input-fusion';
  inputTransformNodeId: string;
  reducerNodeId: string;
  reducerInputPort: InputPortId;
  producerInputNodeIds: string[];
  dependencyKind: DependencyKind;
  materializationRequirement: MaterializationRequirement;
}

export type SemanticRegion = ReductionInputFusionRegion;

export interface Graph {
  id: string;
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  outputs: string[];
  semanticRegions?: SemanticRegion[];
}

export interface GraphPosition {
  x: number;
  y: number;
}

export interface GraphViewport {
  x: number;
  y: number;
  zoom: number;
}

export interface GraphDocument {
  schemaVersion: 1;
  graph: Graph;
  layout: {
    positions: Record<string, GraphPosition>;
    viewport?: GraphViewport;
  };
}

export function emptyParameters(): EmptyParameters {
  return {};
}
