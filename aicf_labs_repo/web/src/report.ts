// Transport shape only. All operator properties and decisions come from Python.
interface Node {
  id: string;
  type: string;
  inputs: string[];
  outputs: string[];
  attributes: Record<string, unknown>;
  shape: number[];
  semantics: { arity: number; expression: string; properties: {
    kind: string; argument: number | null; condition: string | null;
  }[] };
}

export interface Report {
  schema_version: number;
  model: { type: string; weight_shape?: number[]; bias_shape?: number[] }[];
  original: { nodes: Node[]; output: string };
  transformed: { nodes: Node[]; output: string };
  decisions: { rule: string; status: string; nodes: string[]; reason: string; domain: string }[];
  execution: { backend: string; original_output: unknown; transformed_output: unknown; equal: boolean; scope: string };
  cuda: { node_id: string; implementation_id: string; launch: {
    backend: string; implementation_id: string; status: string; output: unknown; observations: unknown[];
  } }[];
}
