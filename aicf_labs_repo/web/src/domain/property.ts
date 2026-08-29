export const PropertyKind = {
  LINEAR: 'LINEAR',
  POSITIVE_HOMOGENEOUS: 'POSITIVE_HOMOGENEOUS',
  ASSOCIATIVE: 'ASSOCIATIVE',
  COMMUTATIVE: 'COMMUTATIVE',
  PURE: 'PURE',
  ELEMENTWISE: 'ELEMENTWISE',
  REDUCTION: 'REDUCTION',
  PERMUTATION: 'PERMUTATION',
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
} as const;

export type TransformationCapability = typeof TransformationCapability[keyof typeof TransformationCapability];

export interface CapabilityDerivation {
  capability: TransformationCapability;
  /** All listed property kinds must be present in the already scope-resolved claims. */
  requiredPropertyKinds: readonly PropertyKind[];
}

/**
 * Capability derivation is intentionally many-to-many. A property may enable
 * several transformations, while a future capability may require several
 * resolved properties rather than being forced into a 1:1 mapping.
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
];

export function deriveTransformationCapabilities(
  claims: readonly PropertyClaim[],
): readonly TransformationCapability[] {
  const presentKinds = new Set(claims.map(({ kind }) => kind));
  return CAPABILITY_DERIVATIONS
    .filter(({ requiredPropertyKinds }) => requiredPropertyKinds.every((kind) => presentKinds.has(kind)))
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
