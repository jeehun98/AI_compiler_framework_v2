import { beforeEach, describe, expect, it } from 'vitest';
import { EXAMPLE_DOCUMENTS } from '../examples/graphExamples';
import { graphFingerprint } from '../core/rewriteEngine';
import { serializeGraphDocument } from '../core/documentCodec';
import { useGraphStore } from './graphStore';

describe('graph store', () => {
  beforeEach(() => useGraphStore.getState().reset());

  it('adds operators and selects graph nodes', () => {
    const inputId = useGraphStore.getState().addOperator('input', { x: 40, y: 50 });
    useGraphStore.getState().selectNode(inputId);

    const state = useGraphStore.getState();
    expect(state.document.graph.nodes).toContainEqual(expect.objectContaining({ id: inputId, operatorId: 'input' }));
    expect(state.document.layout.positions[inputId]).toEqual({ x: 40, y: 50 });
    expect(state.selectedNodeId).toBe(inputId);
  });

  it('connects free ports and rejects duplicate ports, self-loops, and cycles', () => {
    const x = useGraphStore.getState().addOperator('input');
    const first = useGraphStore.getState().addOperator('relu');
    const second = useGraphStore.getState().addOperator('relu');

    expect(useGraphStore.getState().connectEdge({ sourceNodeId: x, targetNodeId: first, targetPort: 'in-0' }).allowed).toBe(true);
    expect(useGraphStore.getState().connectEdge({ sourceNodeId: second, targetNodeId: first, targetPort: 'in-0' }).allowed).toBe(false);
    expect(useGraphStore.getState().connectEdge({ sourceNodeId: first, targetNodeId: first, targetPort: 'in-0' }).allowed).toBe(false);
    expect(useGraphStore.getState().connectEdge({ sourceNodeId: x, targetNodeId: second, targetPort: 'in-9' }).allowed).toBe(false);
    expect(useGraphStore.getState().connectEdge({ sourceNodeId: first, targetNodeId: second, targetPort: 'in-0' }).allowed).toBe(true);
    useGraphStore.getState().deleteEdge(useGraphStore.getState().document.graph.edges.find(({ targetNodeId }) => targetNodeId === first)?.id ?? '');
    expect(useGraphStore.getState().connectEdge({ sourceNodeId: second, targetNodeId: first, targetPort: 'in-0' }).allowed).toBe(false);
    expect(useGraphStore.getState().connectionError).toContain('순환');
  });

  it('loads an example and applies a rewrite only after the action is invoked', () => {
    useGraphStore.getState().loadExample('x-times-one');
    const before = graphFingerprint(useGraphStore.getState().document.graph);
    const rewrite = useGraphStore.getState().rewriteCandidates.find(({ ruleId }) => ruleId === 'mul-one');
    expect(rewrite).toBeDefined();
    expect(graphFingerprint(useGraphStore.getState().document.graph)).toBe(before);

    useGraphStore.getState().selectRewriteCandidate(rewrite?.id ?? null);
    expect(useGraphStore.getState().applySelectedRewrite()).toBe(true);
    expect(graphFingerprint(useGraphStore.getState().document.graph)).not.toBe(before);
    expect(useGraphStore.getState().document.graph.outputs).toEqual(['x']);
  });

  it('preserves the current graph when JSON import fails', () => {
    useGraphStore.getState().loadExample('constant-expression');
    const before = serializeGraphDocument(useGraphStore.getState().document);
    const result = useGraphStore.getState().importJson('{"schemaVersion":1,"graph":null}');

    expect(result.ok).toBe(false);
    expect(serializeGraphDocument(useGraphStore.getState().document)).toBe(before);
    expect(useGraphStore.getState().documentError).not.toBeNull();
  });

  it('replaces the complete document when loading each example', () => {
    for (const example of EXAMPLE_DOCUMENTS) {
      useGraphStore.getState().loadExample(example.id);
      expect(useGraphStore.getState().document.graph.name).toBe(example.document.graph.name);
      expect(useGraphStore.getState().validation.valid).toBe(true);
    }
  });
});
