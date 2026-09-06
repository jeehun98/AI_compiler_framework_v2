import {
  OperatorSemanticFactKind,
  TransformationFactKind,
  operatorSemanticFactKinds,
  type OperatorSemanticFacts,
  type TransformationFact,
} from './semantic';

export const PropertyKind = {
  LINEAR: 'LINEAR',
  POSITIVE_HOMOGENEOUS: 'POSITIVE_HOMOGENEOUS',
  ASSOCIATIVE: 'ASSOCIATIVE',
  COMMUTATIVE: 'COMMUTATIVE',
  PURE: 'PURE',
  ELEMENTWISE: 'ELEMENTWISE',
  REDUCTION: 'REDUCTION',
  PERMUTATION: 'PERMUTATION',
  SHAPE_PRESERVING: 'SHAPE_PRESERVING',
} as const;

export type PropertyKind = typeof PropertyKind[keyof typeof PropertyKind];

export type PropertyScope =
  | { kind: 'operator' }
  | { kind: 'input'; inputPort: `in-${number}` }
  | { kind: 'output'; outputPort: 'out' };

export type PropertyCondition =
  | { kind: 'input-fixed'; inputPort: `in-${number}` }
  | { kind: 'scale-comparison'; operator: '>='; value: 0 };

export interface PropertyClaim {
  kind: PropertyKind;
  scope: PropertyScope;
  conditions: readonly PropertyCondition[];
  justification?: string;
}

export const TransformationCapability = {
  SCALE_PROPAGATION: 'SCALE_PROPAGATION',
  POSITIVE_SCALE_PROPAGATION: 'POSITIVE_SCALE_PROPAGATION',
  REASSOCIATION: 'REASSOCIATION',
  REDUCTION_INPUT_FUSION: 'REDUCTION_INPUT_FUSION',
} as const;

export type TransformationCapability = typeof TransformationCapability[keyof typeof TransformationCapability];

export interface CapabilityDerivation {
  capability: TransformationCapability;
  /** All listed property kinds must be present in the already scope-resolved claims. */
  requiredPropertyKinds: readonly PropertyKind[];
  requiredGraphFactKinds?: readonly TransformationFactKind[];
  requiredOperatorSemanticFacts?: readonly OperatorSemanticFactRequirement[];
}

export interface OperatorSemanticFactRequirement {
  role: string;
  requiredKinds: readonly OperatorSemanticFactKind[];
}

export interface ScopedOperatorSemanticFacts {
  role: string;
  facts: OperatorSemanticFacts;
}

export interface CapabilityFactInputs {
  graphFacts?: readonly TransformationFact[];
  operatorSemanticFacts?: readonly ScopedOperatorSemanticFacts[];
}

/**
 * Capability derivation is intentionally many-to-many and may also require
 * matched graph facts. Properties describe operators; facts establish that
 * the required relation actually exists in the current graph.
 */
export const CAPABILITY_DERIVATIONS: readonly CapabilityDerivation[] = [
  {
    capability: TransformationCapability.SCALE_PROPAGATION,
    requiredPropertyKinds: [PropertyKind.LINEAR],
  },
  {
    capability: TransformationCapability.POSITIVE_SCALE_PROPAGATION,
    requiredPropertyKinds: [PropertyKind.POSITIVE_HOMOGENEOUS],
  },
  {
    capability: TransformationCapability.REASSOCIATION,
    requiredPropertyKinds: [PropertyKind.ASSOCIATIVE],
  },
  {
    capability: TransformationCapability.REDUCTION_INPUT_FUSION,
    requiredPropertyKinds: [PropertyKind.ELEMENTWISE, PropertyKind.PURE, PropertyKind.REDUCTION],
    requiredGraphFactKinds: [TransformationFactKind.DIRECT_DATAFLOW],
    requiredOperatorSemanticFacts: [
      {
        role: 'producer',
        requiredKinds: [OperatorSemanticFactKind.CORRESPONDING_ELEMENT_DEPENDENCY],
      },
      {
        role: 'consumer',
        requiredKinds: [
          OperatorSemanticFactKind.FULL_AXIS_DEPENDENCY,
          OperatorSemanticFactKind.INCREMENTAL_PARTIAL_STATE,
          OperatorSemanticFactKind.MERGEABLE_PARTIAL_STATE,
        ],
      },
    ],
  },
];

export function deriveTransformationCapabilities(
  claims: readonly PropertyClaim[],
  facts: CapabilityFactInputs = {},
): readonly TransformationCapability[] {
  const presentKinds = new Set(claims.map(({ kind }) => kind));
  const presentGraphFactKinds = new Set((facts.graphFacts ?? []).map(({ kind }) => kind));
  const operatorFactsByRole = new Map(
    (facts.operatorSemanticFacts ?? []).map(({ role, facts: semanticFacts }) => [
      role,
      new Set(operatorSemanticFactKinds(semanticFacts)),
    ]),
  );
  return CAPABILITY_DERIVATIONS
    .filter(({
      requiredPropertyKinds,
      requiredGraphFactKinds = [],
      requiredOperatorSemanticFacts = [],
    }) =>
      requiredPropertyKinds.every((kind) => presentKinds.has(kind))
      && requiredGraphFactKinds.every((kind) => presentGraphFactKinds.has(kind))
      && requiredOperatorSemanticFacts.every(({ role, requiredKinds }) => {
        const present = operatorFactsByRole.get(role);
        return present !== undefined && requiredKinds.every((kind) => present.has(kind));
      }))
    .map(({ capability }) => capability);
}

export function capabilitiesForProperty(claim: PropertyClaim): readonly TransformationCapability[] {
  return deriveTransformationCapabilities([claim]);
}

export function formatPropertyScope(scope: PropertyScope): string {
  switch (scope.kind) {
    case 'operator': return 'operator';
    case 'input': return `input:${scope.inputPort.slice(3)}`;
    case 'output': return `output:${scope.outputPort}`;
  }
}

export function formatPropertyClaim(claim: Pick<PropertyClaim, 'kind' | 'scope'>): string {
  return `${claim.kind}(${formatPropertyScope(claim.scope)})`;
}
