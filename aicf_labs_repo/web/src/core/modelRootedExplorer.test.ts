import { describe, expect, it } from 'vitest';
import type { ExplorerNode, ModelExplorerItem } from '../domain/explorer';
import type { Graph } from '../domain/graph';
import { getExample } from '../examples/graphExamples';
import { searchRewriteCandidates } from './rewriteEngine';
import { buildModelRootedExplorer, flattenExplorerNodes } from './modelRootedExplorer';

function scaleExample() {
  const example = getExample('scale-matmul-left');
  if (!example) throw new Error('Missing scale-matmul-left example');
  const search = searchRewriteCandidates(example.document.graph);
  return { example, search };
}

function maxDepth(nodes: readonly ExplorerNode[], depth = 1): number {
  return Math.max(...nodes.map((node) => node.children?.length ? maxDepth(node.children, depth + 1) : depth));
}

function section(tree: readonly ExplorerNode[], label: string): ExplorerNode {
  const found = tree[0]?.children?.find((node) => node.kind === 'section' && node.label === label);
  expect(found, `missing ${label} section`).toBeDefined();
  if (!found) throw new Error(`Missing ${label} section`);
  return found;
}

describe('model-rooted explorer projection', () => {
  it('places every derived model item under exactly one derived model root', () => {
    const { example, search } = scaleExample();
    const tree = buildModelRootedExplorer({ graph: example.document.graph, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });
    const flattened = flattenExplorerNodes(tree);

    expect(tree).toHaveLength(1);
    expect(tree[0]).toMatchObject({
      kind: 'model-root',
      label: 'Scale · MatMul Left',
      source: 'derived',
      modelSummary: { layerCount: 2, graphNodeCount: 5, inputCount: 2, outputCount: 1 },
      target: { modelItemId: 'model-root:example-scale-matmul-left' },
    });
    expect(tree[0]?.children?.map(({ label }) => label)).toEqual(['Layers', 'Model Graph']);
    expect(section(tree, 'Layers').children?.map(({ label }) => label)).toEqual(['ScaleLayer_0', 'Linear_0']);
    expect(flattened).toContainEqual(expect.objectContaining({ kind: 'graph-node', label: 'Mul · scale', source: 'real' }));
    expect(flattened).toContainEqual(expect.objectContaining({ kind: 'kernel', source: 'placeholder' }));
  });

  it('projects one model layer to multiple graph nodes', () => {
    const { example, search } = scaleExample();
    const modelItems: ModelExplorerItem[] = [{ id: 'linear-block', kind: 'layer', label: 'LinearBlock_0', source: 'real', graphNodeIds: ['scale', 'matmul'] }];
    const tree = buildModelRootedExplorer({ graph: example.document.graph, modelItems, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });
    const graphRepresentation = flattenExplorerNodes(tree).find((node) => node.kind === 'representation' && node.workspace === 'graph');

    expect(graphRepresentation?.children?.filter(({ kind }) => kind === 'graph-node').map(({ target }) => target?.graphNodeId))
      .toEqual(['scale', 'matmul']);
  });

  it('derives a stable layer order for a simple dependency chain', () => {
    const { example, search } = scaleExample();
    const tree = buildModelRootedExplorer({ graph: example.document.graph, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });
    const orderedLayers = section(tree, 'Layers').children?.filter(({ orderSource }) => orderSource === 'derived') ?? [];

    expect(orderedLayers.map(({ target, order }) => [target?.modelItemId, order])).toEqual([
      ['scale', 0],
      ['matmul', 1],
    ]);
  });

  it('uses graph dependencies instead of graph node array order', () => {
    const { example } = scaleExample();
    const graph = { ...example.document.graph, nodes: [...example.document.graph.nodes].reverse() };
    const search = searchRewriteCandidates(graph);
    const tree = buildModelRootedExplorer({ graph, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });
    const orderedLayers = section(tree, 'Layers').children?.filter(({ orderSource }) => orderSource === 'derived') ?? [];

    expect(orderedLayers.map(({ target, order }) => [target?.modelItemId, order])).toEqual([
      ['scale', 0],
      ['matmul', 1],
    ]);
  });

  it('does not invent a total order for ambiguous branches', () => {
    const graph: Graph = {
      id: 'ambiguous-branch',
      name: 'Ambiguous branch',
      nodes: [
        { id: 'x', operatorId: 'input', parameters: { symbol: 'x', shape: [] } },
        { id: 'left', operatorId: 'relu', parameters: {} },
        { id: 'right', operatorId: 'relu', parameters: {} },
      ],
      edges: [
        { id: 'x-left', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'left', targetPort: 'in-0' },
        { id: 'x-right', sourceNodeId: 'x', sourcePort: 'out', targetNodeId: 'right', targetPort: 'in-0' },
      ],
      outputs: ['left', 'right'],
    };
    const tree = buildModelRootedExplorer({ graph, rewriteCandidates: [], transformationAttempts: [] });
    const ambiguousLayers = section(tree, 'Layers').children?.filter(({ orderSource }) => orderSource === 'unknown') ?? [];

    expect(ambiguousLayers).toHaveLength(2);
    expect(ambiguousLayers.every(({ order }) => order === undefined)).toBe(true);
  });

  it('represents an N:1 fused kernel with shared target identity under multiple layers', () => {
    const { example, search } = scaleExample();
    const tree = buildModelRootedExplorer({ graph: example.document.graph, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });
    const sharedKernelNodes = flattenExplorerNodes(tree).filter((node) => node.kind === 'kernel' && node.target?.graphNodeIds?.includes('scale') && node.target.graphNodeIds.includes('matmul'));

    expect(sharedKernelNodes).toHaveLength(2);
    expect(new Set(sharedKernelNodes.map(({ target }) => target?.kernelId)).size).toBe(1);
    expect(sharedKernelNodes).toEqual(expect.arrayContaining([
      expect.objectContaining({ shared: true, sharedReferenceCount: 2 }),
      expect.objectContaining({ shared: true, sharedReferenceCount: 2 }),
    ]));
  });

  it('does not mark a single-reference kernel as shared', () => {
    const { example, search } = scaleExample();
    const modelItems: ModelExplorerItem[] = [{ id: 'linear-block', kind: 'layer', label: 'LinearBlock_0', source: 'real', graphNodeIds: ['scale', 'matmul'] }];
    const tree = buildModelRootedExplorer({ graph: example.document.graph, modelItems, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });
    const kernel = flattenExplorerNodes(tree).find((node) => node.kind === 'kernel');

    expect(kernel?.target?.kernelId).toBeDefined();
    expect(kernel).not.toHaveProperty('shared');
    expect(kernel).not.toHaveProperty('sharedReferenceCount');
  });

  it('supports recursive model modules and semantic evidence beyond five levels', () => {
    const { example, search } = scaleExample();
    const modelItems: ModelExplorerItem[] = [{
      id: 'transformer', kind: 'module', label: 'Transformer', source: 'real', graphNodeIds: [], children: [{
        id: 'mlp', kind: 'module', label: 'MLP_0', source: 'real', graphNodeIds: [], children: [
          { id: 'linear', kind: 'layer', label: 'Linear_0', source: 'real', graphNodeIds: ['scale', 'matmul'] },
        ],
      }],
    }];
    const tree = buildModelRootedExplorer({ graph: example.document.graph, modelItems, rewriteCandidates: search.candidates, transformationAttempts: search.attempts });

    expect(maxDepth(tree)).toBeGreaterThanOrEqual(8);
    expect(flattenExplorerNodes(tree)).toContainEqual(expect.objectContaining({ kind: 'transformation-attempt', source: 'real' }));
  });
});
