import { LegalityStatus, type TransformationAttempt } from '../domain/rewrite';

export function serializeTransformationAttempts(attempts: readonly TransformationAttempt[]): string {
  return JSON.stringify(attempts, null, 2);
}

export function formatTransformationAttemptsText(attempts: readonly TransformationAttempt[]): string {
  if (attempts.length === 0) return '(no transformation attempts)';
  const sourceGraphId = attempts[0].sourceGraphId;
  const lines = [sourceGraphId];
  attempts.forEach((attempt, index) => {
    const branch = index === attempts.length - 1 ? '└─' : '├─';
    const outcome = attempt.status === LegalityStatus.APPLICABLE
      ? attempt.targetGraphId ?? LegalityStatus.UNKNOWN
      : attempt.status;
    const reason = attempt.status === LegalityStatus.APPLICABLE || !attempt.reason
      ? ''
      : ` — ${attempt.reason}`;
    lines.push(`${branch} ${attempt.ruleName} [${attempt.matchId}] → ${outcome}${reason}`);
  });
  return lines.join('\n');
}
