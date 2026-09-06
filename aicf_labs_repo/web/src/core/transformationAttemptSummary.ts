import { LegalityStatus, type TransformationAttempt } from '../domain/rewrite';
import { DependencyFootprintKind } from '../domain/semantic';

function unknownSemanticReason(attempt: TransformationAttempt): string | undefined {
  if (attempt.evidence?.kind !== 'producer-embedding') return undefined;
  const producerFootprint = attempt.evidence.producerSemantics.resolvedDependencyFootprint;
  if (producerFootprint.kind === DependencyFootprintKind.UNKNOWN) return producerFootprint.reason;
  const consumerFootprint = attempt.evidence.consumerSemantics.resolvedDependencyFootprint;
  if (consumerFootprint.kind === DependencyFootprintKind.UNKNOWN) return consumerFootprint.reason;
  return undefined;
}

export function selectTransformationAttemptReason(attempt: TransformationAttempt): string {
  if (attempt.status === LegalityStatus.REJECTED) {
    if (attempt.mathematicalLegality.status === LegalityStatus.REJECTED) {
      return attempt.mathematicalLegality.reason;
    }
    if (attempt.graphLegality.status === LegalityStatus.REJECTED) return attempt.graphLegality.reason;
  }

  if (attempt.status === LegalityStatus.UNKNOWN) {
    if (attempt.mathematicalLegality.status === LegalityStatus.UNKNOWN) {
      return unknownSemanticReason(attempt) ?? attempt.mathematicalLegality.reason;
    }
    if (attempt.graphLegality.status === LegalityStatus.UNKNOWN) return attempt.graphLegality.reason;
  }

  if (attempt.graphLegality.status === LegalityStatus.APPLICABLE) return attempt.graphLegality.reason;
  if (attempt.mathematicalLegality.status === LegalityStatus.APPLICABLE) {
    return attempt.mathematicalLegality.reason;
  }
  return attempt.reason ?? 'No transformation reason is available.';
}
