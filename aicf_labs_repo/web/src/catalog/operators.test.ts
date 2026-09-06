import { describe, expect, it } from 'vitest';
import { matchesMask, OperatorMask, summarizeMasks } from '../domain/operator';
import {
  capabilitiesForProperty,
  deriveTransformationCapabilities,
  PropertyKind,
  TransformationCapability,
} from '../domain/property';
import {
  CorrespondingElementMapping,
  DependencyFootprintKind,
  ReuseKind,
  TransformationFactKind,
} from '../domain/semantic';
import { OPERATORS, OPERATOR_MAP } from './operators';

describe('operator catalog', () => {
  it('contains exactly the eight MVP operators', () => {
    expect(OPERATORS.map(({ id }) => id)).toEqual([
      'input', 'constant', 'add', 'mul', 'matmul', 'relu', 'transpose', 'reduceSum',
    ]);
  });

  it('keeps fixed arity metadata addressable by id', () => {
    expect(OPERATOR_MAP.input.arity).toBe(0);
    expect(OPERATOR_MAP.add.arity).toBe(2);
    expect(OPERATOR_MAP.transpose.arity).toBe(1);
  });

  it('indexes shared features without replacing operator identity', () => {
    const required = OperatorMask.ELEMENTWISE | OperatorMask.PURE;
    const matched = OPERATORS.filter(({ mask }) => matchesMask(mask, required));

    expect(matched.map(({ id }) => id)).toEqual(['add', 'mul', 'relu']);
    expect(matched.every(({ category }) => category === 'elementwise')).toBe(true);
  });

  it('summarizes a region with only common and present bit sets', () => {
    const summary = summarizeMasks([OPERATOR_MAP.add.mask, OPERATOR_MAP.relu.mask]);

    expect(summary.common).toBe(OperatorMask.ELEMENTWISE | OperatorMask.PURE);
    expect(matchesMask(summary.present, OperatorMask.BROADCAST)).toBe(true);
  });

  it('declares scoped linearity for both MatMul inputs', () => {
    const claims = OPERATOR_MAP.matmul.propertyClaims.filter(({ kind }) => kind === PropertyKind.LINEAR);
    expect(claims.map(({ scope }) => scope)).toEqual([
      { kind: 'input', inputPort: 'in-0' },
      { kind: 'input', inputPort: 'in-1' },
    ]);
    expect(claims.map(({ conditions }) => conditions)).toEqual([
      [{ kind: 'input-fixed', inputPort: 'in-1' }],
      [{ kind: 'input-fixed', inputPort: 'in-0' }],
    ]);
    expect(claims.every((claim) => capabilitiesForProperty(claim).includes(TransformationCapability.SCALE_PROPAGATION))).toBe(true);
  });

  it('derives the requested scale capabilities from ReduceSum and ReLU claims', () => {
    const reduceLinearity = OPERATOR_MAP.reduceSum.propertyClaims.find(({ kind }) => kind === PropertyKind.LINEAR);
    const reluHomogeneity = OPERATOR_MAP.relu.propertyClaims.find(({ kind }) => kind === PropertyKind.POSITIVE_HOMOGENEOUS);
    expect(reduceLinearity?.scope).toEqual({ kind: 'input', inputPort: 'in-0' });
    expect(reluHomogeneity?.scope).toEqual({ kind: 'input', inputPort: 'in-0' });
    expect(reluHomogeneity?.conditions).toEqual([{ kind: 'scale-comparison', operator: '>=', value: 0 }]);
    expect(reduceLinearity && capabilitiesForProperty(reduceLinearity)).toContain(TransformationCapability.SCALE_PROPAGATION);
    expect(reluHomogeneity && capabilitiesForProperty(reluHomogeneity)).toContain(TransformationCapability.POSITIVE_SCALE_PROPAGATION);
  });

  it('does not invent scale properties for an unsupported operator', () => {
    expect(OPERATOR_MAP.transpose.propertyClaims.some(({ kind }) =>
      kind === PropertyKind.LINEAR || kind === PropertyKind.POSITIVE_HOMOGENEOUS)).toBe(false);
  });

  it('derives reassociation from associative metadata as a set-level capability', () => {
    const addAssociativity = OPERATOR_MAP.add.propertyClaims.find(
      ({ kind }) => kind === PropertyKind.ASSOCIATIVE,
    );
    expect(addAssociativity?.scope).toEqual({ kind: 'operator' });
    expect(OPERATOR_MAP.matmul.propertyClaims.some(
      ({ kind }) => kind === PropertyKind.ASSOCIATIVE,
    )).toBe(false);
    expect(deriveTransformationCapabilities(OPERATOR_MAP.add.propertyClaims)).toContain(
      TransformationCapability.REASSOCIATION,
    );
    expect(deriveTransformationCapabilities(OPERATOR_MAP.mul.propertyClaims)).toContain(
      TransformationCapability.REASSOCIATION,
    );
    expect(deriveTransformationCapabilities(OPERATOR_MAP.matmul.propertyClaims)).not.toContain(
      TransformationCapability.REASSOCIATION,
    );
  });

  it('derives reduction input fusion only from the full property set plus direct dataflow', () => {
    const claims = [...OPERATOR_MAP.relu.propertyClaims, ...OPERATOR_MAP.reduceSum.propertyClaims];
    const graphFacts = [{
      scope: 'graph-instance' as const,
      kind: TransformationFactKind.DIRECT_DATAFLOW,
      reason: 'test producer output directly supplies the reduction input',
    }];
    const operatorSemanticFacts = [
      { role: 'producer', facts: OPERATOR_MAP.relu.semanticFacts },
      { role: 'consumer', facts: OPERATOR_MAP.reduceSum.semanticFacts },
    ];
    expect(OPERATOR_MAP.relu.propertyClaims.some(
      ({ kind }) => kind === PropertyKind.SHAPE_PRESERVING,
    )).toBe(true);
    expect(deriveTransformationCapabilities(claims)).not.toContain(
      TransformationCapability.REDUCTION_INPUT_FUSION,
    );
    expect(deriveTransformationCapabilities(claims, { graphFacts })).not.toContain(
      TransformationCapability.REDUCTION_INPUT_FUSION,
    );
    expect(deriveTransformationCapabilities(claims, { operatorSemanticFacts })).not.toContain(
      TransformationCapability.REDUCTION_INPUT_FUSION,
    );
    expect(deriveTransformationCapabilities(claims, {
      graphFacts,
      operatorSemanticFacts: [
        { role: 'producer', facts: OPERATOR_MAP.reduceSum.semanticFacts },
        { role: 'consumer', facts: OPERATOR_MAP.relu.semanticFacts },
      ],
    })).not.toContain(TransformationCapability.REDUCTION_INPUT_FUSION);
    expect(deriveTransformationCapabilities(claims, {
      graphFacts,
      operatorSemanticFacts,
    })).toContain(TransformationCapability.REDUCTION_INPUT_FUSION);
  });

  it('separates operator-level dependency, reuse, and partial-state semantics', () => {
    expect(OPERATOR_MAP.relu.semanticFacts).toMatchObject({
      scope: 'operator',
      dependencyFootprint: {
        kind: DependencyFootprintKind.CORRESPONDING_ELEMENT,
        inputMapping: CorrespondingElementMapping.EXACT,
      },
      reuse: { kind: ReuseKind.NONE },
    });
    expect(OPERATOR_MAP.add.semanticFacts).toMatchObject({
      dependencyFootprint: {
        kind: DependencyFootprintKind.CORRESPONDING_ELEMENT,
        inputMapping: CorrespondingElementMapping.EXACT_OR_SCALAR_BROADCAST,
      },
      reuse: { kind: ReuseKind.UNKNOWN },
    });
    expect(OPERATOR_MAP.reduceSum.semanticFacts).toMatchObject({
      dependencyFootprint: {
        kind: DependencyFootprintKind.FULL_AXIS,
        axis: 'FROM_OPERATOR_ATTRIBUTE',
      },
      reuse: { kind: ReuseKind.WITHIN_OUTPUT },
      partialState: {
        supportsIncrementalUpdate: true,
        supportsMerge: true,
        stateKind: 'ACCUMULATOR',
        stateScope: 'REDUCTION_OUTPUT',
      },
    });
    expect(OPERATOR_MAP.transpose.semanticFacts.dependencyFootprint.kind).toBe(DependencyFootprintKind.UNKNOWN);
    expect(OPERATOR_MAP.matmul.semanticFacts.dependencyFootprint.kind).toBe(DependencyFootprintKind.UNKNOWN);
    expect(OPERATOR_MAP.add.semanticFacts.partialState).toBeUndefined();
  });
});
