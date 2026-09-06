import { beforeEach, describe, expect, it } from 'vitest';
import { EXAMPLE_DOCUMENTS } from '../examples/graphExamples';
import { graphFingerprint } from '../core/rewriteEngine';
import { serializeGraphDocument } from '../core/documentCodec';
import { useGraphStore } from './graphStore';

describe('graph store', () => {
  beforeEach(() => useGraphStore.getState().reset());

  it('creates a Linear template and its internal data-flow edge atomically', () => {
    const id = useGraphStore.getState().addLayer('Linear');
    const state = useGraphStore.getState();
    const layer = state.modelDocument.layers.find((item) => item.id === id)!;
    expect(state.document.graph.nodes.map(({ operatorId }) => operatorId)).toEqual(['matmul', 'add']);
    expect(layer.graphNodeIds).toEqual(state.document.graph.nodes.map(({ id }) => id));
    expect(state.document.graph.edges).toEqual([expect.objectContaining({ sourceNodeId: layer.graphNodeIds[0], targetNodeId: layer.graphNodeIds[1], targetPort: 'in-0' })]);
    expect(state.graphDocuments[0]).toEqual(state.document);
  });

  it.each([
    ['ReLU', ['relu']], ['Scale', ['mul']], ['Reduction', ['reduceSum']], ['Custom', []],
  ] as const)('creates the %s composition', (type, operatorIds) => {
    useGraphStore.getState().addLayer(type);
    const state = useGraphStore.getState();
    expect(state.document.graph.nodes.map(({ operatorId }) => operatorId)).toEqual(operatorIds);
    expect(state.modelDocument.layers[0].graphNodeIds).toEqual(state.document.graph.nodes.map(({ id }) => id));
  });

  it('supports Linear without bias', () => {
    useGraphStore.getState().addLayer('Linear', { bias: false });
    expect(useGraphStore.getState().document.graph.nodes.map(({ operatorId }) => operatorId)).toEqual(['matmul']);
  });

  it('deletes exclusive layer operators and their edges, positions, and output references', () => {
    const layerId = useGraphStore.getState().addLayer('Linear');
    const ids = useGraphStore.getState().modelDocument.layers[0].graphNodeIds;
    useGraphStore.getState().selectNode(ids[0]);
    useGraphStore.getState().deleteLayer(layerId);
    const state = useGraphStore.getState();
    expect(state.modelDocument.layers).toEqual([]);
    expect(state.document.graph).toMatchObject({ nodes: [], edges: [], outputs: [] });
    expect(state.document.layout.positions).toEqual({});
    expect(state.selectedNodeId).toBeNull();
    expect(state.graphDocuments[0]).toEqual(state.document);
  });

  it('preserves an operator referenced by another layer during layer deletion', () => {
    const first = useGraphStore.getState().addLayer('Linear');
    const second = useGraphStore.getState().addLayer('Custom');
    const sharedId = useGraphStore.getState().modelDocument.layers[0].graphNodeIds[0];
    const model = useGraphStore.getState().modelDocument;
    useGraphStore.setState({ modelDocument: { ...model, layers: model.layers.map((layer) => layer.id === second ? { ...layer, graphNodeIds: [sharedId] } : layer) } });
    useGraphStore.getState().deleteLayer(first);
    expect(useGraphStore.getState().document.graph.nodes.map(({ id }) => id)).toEqual([sharedId]);
    expect(useGraphStore.getState().modelDocument.layers[0].graphNodeIds).toEqual([sharedId]);
  });

  it('cleans the owning graph even when a different graph is active', () => {
    const layerId = useGraphStore.getState().addLayer('ReLU');
    const originalGraphId = useGraphStore.getState().selectedGraphId;
    const otherGraphId = useGraphStore.getState().addGraph();
    const otherLayerId = useGraphStore.getState().addLayer('ReLU');
    useGraphStore.getState().deleteLayer(layerId);
    const state = useGraphStore.getState();
    expect(state.selectedGraphId).toBe(otherGraphId);
    expect(state.document.graph.nodes).toHaveLength(1);
    expect(state.graphDocuments.find(({ graph }) => graph.id === originalGraphId)?.graph.nodes).toEqual([]);
    expect(state.modelDocument.layers[0]).toMatchObject({ id: otherLayerId, graphNodeIds: ['relu-1'] });
  });

  it('adds operators and selects graph nodes', () => {
    const inputId = useGraphStore.getState().addOperator('input', { x: 40, y: 50 });
    useGraphStore.getState().selectNode(inputId);

    const state = useGraphStore.getState();
    expect(state.document.graph.nodes).toContainEqual(expect.objectContaining({ id: inputId, operatorId: 'input' }));
    expect(state.document.layout.positions[inputId]).toEqual({ x: 40, y: 50 });
    expect(state.selectedNodeId).toBe(inputId);
  });

  it('adds unique user-defined layers to the selected model', () => {
    useGraphStore.getState().loadExample('tiny-mlp');
    const firstId = useGraphStore.getState().addLayer('Linear');
    const secondId = useGraphStore.getState().addLayer('Linear');

    expect(firstId).not.toBe(secondId);
    expect(useGraphStore.getState().modelDocument.layers.slice(-2)).toEqual([
      expect.objectContaining({ id: firstId, name: 'Linear_2', type: 'Linear', source: 'user-defined', graphNodeIds: [expect.any(String), expect.any(String)] }),
      expect.objectContaining({ id: secondId, name: 'Linear_3', type: 'Linear', source: 'user-defined', graphNodeIds: [expect.any(String), expect.any(String)] }),
    ]);
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'model', selectedModelItemId: secondId });
  });

  it('adds an empty graph and preserves the previous graph for selection', () => {
    useGraphStore.getState().loadExample('tiny-mlp');
    const graphId = useGraphStore.getState().addGraph();

    expect(useGraphStore.getState().document.graph).toMatchObject({ id: graphId, name: 'ScratchGraph_1', nodes: [], edges: [] });
    expect(useGraphStore.getState().modelDocument.graphIds).toEqual(['tiny-mlp-forward', graphId]);
    expect(useGraphStore.getState().selectGraph('tiny-mlp-forward')).toBe(true);
    expect(useGraphStore.getState().document.graph.nodes).toHaveLength(10);
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

  it('removes graph references while preserving the remaining graph and layers', () => {
    useGraphStore.getState().loadExample('tiny-mlp');
    const scratchId = useGraphStore.getState().addGraph();
    const inputId = useGraphStore.getState().addOperator('input');
    expect(useGraphStore.getState().deleteGraph('tiny-mlp-forward')).toBe(true);
    const state = useGraphStore.getState();
    expect(state.selectedGraphId).toBe(scratchId);
    expect(state.document.graph.nodes.map(({ id }) => id)).toEqual([inputId]);
    expect(state.modelDocument.graphIds).toEqual([scratchId]);
    expect(state.modelDocument.layers).toHaveLength(3);
    expect(state.modelDocument.layers.every(({ graphNodeIds }) => graphNodeIds.length === 0)).toBe(true);
    expect(state.deleteGraph(scratchId)).toBe(false);
    expect(state.deleteGraph('missing')).toBe(false);
  });

  it('clears probe and representation selections when deleting the active graph', () => {
    useGraphStore.getState().loadExample('scale-matmul-left');
    const originalId = useGraphStore.getState().selectedGraphId;
    const scratchId = useGraphStore.getState().addGraph();
    useGraphStore.getState().selectGraph(originalId);
    useGraphStore.getState().selectKernel('kernel-0');
    expect(useGraphStore.getState().deleteGraph(originalId)).toBe(true);
    expect(useGraphStore.getState()).toMatchObject({
      selectedGraphId: scratchId, selectedNodeId: null, selectedKernelId: null,
      selectedExplorerNodeId: null, selectedRewriteCandidateId: null,
      probeContext: { graphNodeIds: [] },
    });
    expect(useGraphStore.getState().rewriteCandidates).toEqual([]);
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

  it('clears stale node and rewrite-candidate selection when changing examples', () => {
    useGraphStore.getState().loadExample('x-times-one');
    const rewrite = useGraphStore.getState().rewriteCandidates[0];
    expect(rewrite).toBeDefined();
    useGraphStore.getState().selectNode('mul');
    useGraphStore.getState().selectRewriteCandidate(rewrite?.id ?? null);
    expect(useGraphStore.getState()).toMatchObject({
      selectedNodeId: 'mul',
      selectedRewriteCandidateId: rewrite?.id,
      selectedTransformationAttemptId: rewrite?.attemptId,
      selectedLayerObservationId: `semantic:candidate:${rewrite?.id}`,
    });

    expect(useGraphStore.getState().loadExample('fusion-unknown-matmul')).toBe(true);
    const state = useGraphStore.getState();
    expect(state.document.graph.id).toBe('example-fusion-unknown-matmul');
    expect(state.selectedNodeId).toBeNull();
    expect(state.selectedRewriteCandidateId).toBeNull();
    expect(state.selectedTransformationAttemptId).toBeNull();
    expect(state.selectedLayerObservationId).toBeNull();
    expect(state.visitedFingerprints).toEqual([graphFingerprint(state.document.graph)]);
  });

  it('shares workspace, observation, and probe context across panels', () => {
    expect(useGraphStore.getState().activeWorkspace).toBe('graph');
    useGraphStore.getState().setActiveWorkspace('hardware');
    useGraphStore.getState().selectLayerObservation('trace:hardware', 'hardware');

    expect(useGraphStore.getState()).toMatchObject({
      activeWorkspace: 'hardware',
      selectedLayerObservationId: 'trace:hardware',
    });
  });

  it('keeps a selected transformation in probe context when changing workspaces', () => {
    useGraphStore.getState().loadExample('scale-matmul-left');
    const candidate = useGraphStore.getState().rewriteCandidates.find(({ ruleName }) => ruleName === 'ScaleThroughLinear')!;
    useGraphStore.getState().selectRewriteCandidate(candidate.id);
    useGraphStore.getState().setActiveWorkspace('kernel');
    useGraphStore.getState().setActiveWorkspace('runtime');

    expect(useGraphStore.getState().probeContext).toMatchObject({
      transformationAttemptId: candidate.attemptId,
      graphNodeIds: expect.arrayContaining(['scale', 'matmul']),
    });
    expect(useGraphStore.getState().selectedRewriteCandidateId).toBe(candidate.id);
  });

  it('replaces transformation attempts with the newly loaded graph results', () => {
    useGraphStore.getState().loadExample('scale-relu-negative');
    expect(useGraphStore.getState().transformationAttempts).toContainEqual(expect.objectContaining({
      ruleId: 'scale-through-positive-homogeneous',
      status: 'REJECTED',
    }));

    useGraphStore.getState().loadExample('fusion-unknown-matmul');
    const attempts = useGraphStore.getState().transformationAttempts;
    expect(attempts.some(({ ruleId }) => ruleId === 'scale-through-positive-homogeneous')).toBe(false);
    expect(attempts).toContainEqual(expect.objectContaining({
      ruleId: 'embed-elementwise-producer-into-reduction',
      status: 'UNKNOWN',
    }));
  });
});
