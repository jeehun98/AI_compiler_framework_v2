export const TransformationFactKind = {
  DIRECT_DATAFLOW: 'DIRECT_DATAFLOW',
} as const;

export type TransformationFactKind = typeof TransformationFactKind[keyof typeof TransformationFactKind];

export interface TransformationFact {
  scope: 'graph-instance';
  kind: TransformationFactKind;
  reason: string;
}

export const DependencyFootprintKind = {
  CORRESPONDING_ELEMENT: 'CORRESPONDING_ELEMENT',
  FULL_AXIS: 'FULL_AXIS',
  UNKNOWN: 'UNKNOWN',
} as const;

export type DependencyFootprintKind = typeof DependencyFootprintKind[keyof typeof DependencyFootprintKind];

export const CorrespondingElementMapping = {
  EXACT: 'EXACT',
  EXACT_OR_SCALAR_BROADCAST: 'EXACT_OR_SCALAR_BROADCAST',
} as const;

export type CorrespondingElementMapping = typeof CorrespondingElementMapping[keyof typeof CorrespondingElementMapping];

export type DependencyFootprint =
  | {
    kind: typeof DependencyFootprintKind.CORRESPONDING_ELEMENT;
    inputMapping: CorrespondingElementMapping;
    reason: string;
  }
  | {
    kind: typeof DependencyFootprintKind.FULL_AXIS;
    axis: number | 'all' | 'FROM_OPERATOR_ATTRIBUTE';
    reason: string;
  }
  | {
    kind: typeof DependencyFootprintKind.UNKNOWN;
    reason: string;
  };

export const ReuseKind = {
  NONE: 'NONE',
  WITHIN_OUTPUT: 'WITHIN_OUTPUT',
  ACROSS_OUTPUTS: 'ACROSS_OUTPUTS',
  UNKNOWN: 'UNKNOWN',
} as const;

export type ReuseKind = typeof ReuseKind[keyof typeof ReuseKind];

export interface ReuseRelation {
  kind: ReuseKind;
  reason: string;
}

export interface PartialStateSemantics {
  supportsIncrementalUpdate: boolean;
  supportsMerge: boolean;
  stateKind: 'ACCUMULATOR';
  stateScope: 'REDUCTION_OUTPUT';
}

export interface OperatorSemanticFacts {
  scope: 'operator';
  dependencyFootprint: DependencyFootprint;
  reuse: ReuseRelation;
  partialState?: PartialStateSemantics;
}

export const OperatorSemanticFactKind = {
  CORRESPONDING_ELEMENT_DEPENDENCY: 'CORRESPONDING_ELEMENT_DEPENDENCY',
  FULL_AXIS_DEPENDENCY: 'FULL_AXIS_DEPENDENCY',
  INCREMENTAL_PARTIAL_STATE: 'INCREMENTAL_PARTIAL_STATE',
  MERGEABLE_PARTIAL_STATE: 'MERGEABLE_PARTIAL_STATE',
} as const;

export type OperatorSemanticFactKind = typeof OperatorSemanticFactKind[keyof typeof OperatorSemanticFactKind];

export function operatorSemanticFactKinds(
  facts: OperatorSemanticFacts,
): readonly OperatorSemanticFactKind[] {
  const kinds: OperatorSemanticFactKind[] = [];
  if (facts.dependencyFootprint.kind === DependencyFootprintKind.CORRESPONDING_ELEMENT) {
    kinds.push(OperatorSemanticFactKind.CORRESPONDING_ELEMENT_DEPENDENCY);
  }
  if (facts.dependencyFootprint.kind === DependencyFootprintKind.FULL_AXIS) {
    kinds.push(OperatorSemanticFactKind.FULL_AXIS_DEPENDENCY);
  }
  if (facts.partialState?.supportsIncrementalUpdate) {
    kinds.push(OperatorSemanticFactKind.INCREMENTAL_PARTIAL_STATE);
  }
  if (facts.partialState?.supportsMerge) {
    kinds.push(OperatorSemanticFactKind.MERGEABLE_PARTIAL_STATE);
  }
  return kinds;
}

export const DependencyKind = {
  ELEMENTWISE_CORRESPONDING_INPUTS: 'ELEMENTWISE_CORRESPONDING_INPUTS',
  UNKNOWN: 'UNKNOWN',
} as const;

export type DependencyKind = typeof DependencyKind[keyof typeof DependencyKind];

export const MaterializationRequirement = {
  REQUIRED: 'REQUIRED',
  NOT_REQUIRED: 'NOT_REQUIRED',
  UNKNOWN: 'UNKNOWN',
} as const;

export type MaterializationRequirement = typeof MaterializationRequirement[keyof typeof MaterializationRequirement];

export interface ProducerEmbeddingEvidence {
  kind: 'producer-embedding';
  producerNodeId: string;
  producerOperatorId: string;
  producerProperties: readonly string[];
  consumerNodeId: string;
  consumerOperatorId: string;
  consumerProperties: readonly string[];
  producerSemantics: {
    declaredDependencyFootprint: DependencyFootprint;
    resolvedDependencyFootprint: DependencyFootprint;
    declaredReuse: ReuseRelation;
    resolvedReuse: ReuseRelation;
  };
  consumerSemantics: {
    declaredDependencyFootprint: DependencyFootprint;
    resolvedDependencyFootprint: DependencyFootprint;
    reuse: ReuseRelation;
    partialState?: PartialStateSemantics;
  };
  graphFacts: {
    directDataflow: boolean;
    exclusiveUse: boolean;
    producerIsGraphOutput: boolean;
  };
  dependency: {
    kind: DependencyKind;
    reason: string;
  };
  materialization: {
    intermediateNodeId: string;
    requirement: MaterializationRequirement;
    reason: string;
  };
}

export type TransformationEvidence = ProducerEmbeddingEvidence;
