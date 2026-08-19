import type { GraphDocument } from '../domain/graph';
import type { FreedomAxis } from '../domain/freedom';

export type BackendKind = 'python' | 'cuda' | string;

export interface BackendCapabilities {
  backend: BackendKind;
  supportedOperatorIds: string[];
  supportedDtypes: string[];
  supportsExecution: boolean;
}

export interface BackendDiagnostic {
  severity: 'error' | 'warning' | 'info';
  message: string;
  nodeIds: string[];
}

export interface BackendAnalysisResult {
  diagnostics: BackendDiagnostic[];
  implementationFreedom: Record<string, FreedomAxis>;
}

/**
 * Future Python/CUDA integrations implement this boundary. The browser MVP keeps
 * semantic validation and rewriting local and does not provide an adapter yet.
 */
export interface GraphBackendAdapter {
  getCapabilities(signal?: AbortSignal): Promise<BackendCapabilities>;
  analyze(document: GraphDocument, signal?: AbortSignal): Promise<BackendAnalysisResult>;
}
