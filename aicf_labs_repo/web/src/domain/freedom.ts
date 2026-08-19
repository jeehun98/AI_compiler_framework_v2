export type FreedomStatus = 'available' | 'conditional' | 'fixed' | 'unknown';

export interface FreedomAxis {
  status: FreedomStatus;
  summary: string;
  constraints: readonly string[];
}

export interface FreedomProfile {
  algebraic: FreedomAxis;
  numerical: FreedomAxis;
  structural: FreedomAxis;
  implementation: FreedomAxis;
}
