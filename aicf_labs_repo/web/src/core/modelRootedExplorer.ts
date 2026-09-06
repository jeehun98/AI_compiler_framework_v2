import { getOperator } from '../catalog/operators';
import type { ExplorerNode, ModelExplorerItem, ModelLayerOrderSource, ModelRootSummary } from '../domain/explorer';
import type { Graph, GraphDocument, GraphNode } from '../domain/graph';
import type { ModelLayer } from '../domain/model';
import type { RewriteCandidate, TransformationAttempt } from '../domain/rewrite';
import { deriveModelLayersFromGraph } from './modelDocument';
import { selectTransformationAttemptReason } from './transformationAttemptSummary';

export interface BuildModelRootedExplorerInput {
  graph: Graph;
  modelId?: string;
  modelName?: string;
  modelItems?: readonly ModelExplorerItem[];
  graphDocuments?: readonly GraphDocument[];
  rewriteCandidates: readonly RewriteCandidate[];
  transformationAttempts: readonly TransformationAttempt[];
}

function safeId(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-|-$/g, '').toLowerCase() || 'item';
}

function countLabel(count: number, singular: string): string {
  return `${count} ${singular}${count === 1 ? '' : 's'}`;
}

export function deriveModelItemsFromGraph(graph: Graph): ModelExplorerItem[] {
  return modelExplorerItemsFromLayers(deriveModelLayersFromGraph(graph), graph);
}

export function modelExplorerItemsFromLayers(layers: readonly ModelLayer[], graph: Graph): ModelExplorerItem[] {
  return layers.map((layer) => ({
    id: layer.id,
    kind: layer.graphNodeIds.some((id) => graph.outputs.includes(id)) ? 'output' : 'layer',
    label: layer.name,
    source: layer.source,
    graphNodeIds: !layer.graphId || layer.graphId === graph.id ? layer.graphNodeIds : [],
  }));
}

export function modelRootExplorerNodeId(modelId: string): string {
  return `model-root:${safeId(modelId)}`;
}

export function summarizeModelRoot(graph: Graph, modelItems: readonly ModelExplorerItem[]): ModelRootSummary {
  const graphNodeIds = new Set(graph.nodes.map(({ id }) => id));
  const layerCount = (items: readonly ModelExplorerItem[]): number => items.reduce(
    (count, item) => count + (item.kind === 'layer' || item.kind === 'output' ? 1 : 0) + layerCount(item.children ?? []),
    0,
  );
  return {
    layerCount: layerCount(modelItems),
    graphNodeCount: graph.nodes.length,
    inputCount: graph.nodes.filter(({ operatorId }) => operatorId === 'input').length,
    outputCount: new Set(graph.outputs.filter((id) => graphNodeIds.has(id))).size,
  };
}

export interface ModelLayerOrderEntry {
  modelItemId: string;
  label: string;
  order?: number;
  source: ModelLayerOrderSource;
}

export interface ModelLayerOrdering {
  status: ModelLayerOrderSource;
  layers: readonly ModelLayerOrderEntry[];
}

function flattenedOrderedLayerItems(items: readonly ModelExplorerItem[]): ModelExplorerItem[] {
  return items.flatMap((item) => [
    ...(item.kind === 'layer' || item.kind === 'output' ? [item] : []),
    ...flattenedOrderedLayerItems(item.children ?? []),
  ]);
}

export function deriveModelLayerOrdering(graph: Graph, modelItems: readonly ModelExplorerItem[]): ModelLayerOrdering {
  const layers = flattenedOrderedLayerItems(modelItems);
  const unknown = (): ModelLayerOrdering => ({
    status: 'unknown',
    layers: layers.map(({ id, label }) => ({ modelItemId: id, label, source: 'unknown' })),
  });
  if (layers.length === 0) return { status: 'derived', layers: [] };

  const layerIds = new Set(layers.map(({ id }) => id));
  const ownersByGraphNode = new Map<string, Set<string>>();
  for (const layer of layers) {
    for (const graphNodeId of uniqueGraphNodeIds(layer.graphNodeIds, graph)) {
      const owners = ownersByGraphNode.get(graphNodeId) ?? new Set<string>();
      owners.add(layer.id);
      ownersByGraphNode.set(graphNodeId, owners);
    }
  }

  const successors = new Map(layers.map(({ id }) => [id, new Set<string>()]));
  const inDegree = new Map(layers.map(({ id }) => [id, 0]));
  for (const edge of graph.edges) {
    const sourceOwners = ownersByGraphNode.get(edge.sourceNodeId) ?? [];
    const targetOwners = ownersByGraphNode.get(edge.targetNodeId) ?? [];
    for (const sourceId of sourceOwners) {
      for (const targetId of targetOwners) {
        if (sourceId === targetId || !layerIds.has(sourceId) || !layerIds.has(targetId)) continue;
        const targets = successors.get(sourceId)!;
        if (targets.has(targetId)) continue;
        targets.add(targetId);
        inDegree.set(targetId, (inDegree.get(targetId) ?? 0) + 1);
      }
    }
  }

  const remaining = new Set(layerIds);
  const orderedIds: string[] = [];
  while (remaining.size > 0) {
    const ready = [...remaining].filter((id) => inDegree.get(id) === 0);
    if (ready.length !== 1) return unknown();
    const currentId = ready[0]!;
    orderedIds.push(currentId);
    remaining.delete(currentId);
    for (const targetId of successors.get(currentId) ?? []) {
      inDegree.set(targetId, (inDegree.get(targetId) ?? 0) - 1);
    }
  }

  const layerById = new Map(layers.map((layer) => [layer.id, layer]));
  return {
    status: 'derived',
    layers: orderedIds.map((id, order) => ({ modelItemId: id, label: layerById.get(id)!.label, order, source: 'derived' })),
  };
}

function arrangeTopLevelModelItems(items: readonly ModelExplorerItem[], ordering: ModelLayerOrdering): readonly ModelExplorerItem[] {
  if (ordering.status === 'unknown') return items;
  const orderById = new Map(ordering.layers.map(({ modelItemId, order }) => [modelItemId, order ?? Number.MAX_SAFE_INTEGER]));
  const orderedLayers = items
    .filter(({ kind }) => kind === 'layer' || kind === 'output')
    .sort((left, right) => (orderById.get(left.id) ?? Number.MAX_SAFE_INTEGER) - (orderById.get(right.id) ?? Number.MAX_SAFE_INTEGER));
  let layerIndex = 0;
  return items.map((item) => item.kind === 'layer' || item.kind === 'output' ? orderedLayers[layerIndex++]! : item);
}

function uniqueGraphNodeIds(values: readonly string[], graph: Graph): string[] {
  const validIds = new Set(graph.nodes.map(({ id }) => id));
  return [...new Set(values.filter((id) => validIds.has(id)))];
}

function implementationGraphNodeIds(values: readonly string[], graph: Graph): string[] {
  const uniqueIds = uniqueGraphNodeIds(values, graph);
  const implementationIds = uniqueIds.filter((id) => {
    const operatorId = graph.nodes.find((node) => node.id === id)?.operatorId;
    return operatorId !== 'input' && operatorId !== 'constant';
  });
  return implementationIds.length > 0 ? implementationIds : uniqueIds;
}

function candidateImplementationNodeIds(candidate: RewriteCandidate, graph: Graph): string[] {
  return implementationGraphNodeIds(candidate.affectedNodeIds, graph);
}

function attemptGraphNodeIds(attempt: TransformationAttempt, graph: Graph): string[] {
  return implementationGraphNodeIds(Object.values(attempt.bindings), graph);
}

function attemptExplorerNode(attempt: TransformationAttempt, graph: Graph, path: string): ExplorerNode {
  const graphNodeIds = attemptGraphNodeIds(attempt, graph);
  return {
    id: `${path}:attempt:${safeId(attempt.id)}`,
    kind: 'transformation-attempt',
    label: `Attempt · ${attempt.ruleName}`,
    description: `${attempt.ruleId} · ${attempt.matchId} · ${selectTransformationAttemptReason(attempt)}`,
    workspace: 'graph',
    source: 'real',
    badge: attempt.status,
    target: { transformationAttemptId: attempt.id, graphNodeIds },
    children: [
      {
        id: `${path}:attempt:${safeId(attempt.id)}:mathematical`,
        kind: 'evidence',
        label: `Mathematical · ${attempt.mathematicalLegality.status}`,
        description: attempt.mathematicalLegality.reason,
        workspace: 'graph',
        source: 'real',
        badge: attempt.mathematicalLegality.status,
        target: { transformationAttemptId: attempt.id, graphNodeIds },
      },
      {
        id: `${path}:attempt:${safeId(attempt.id)}:graph-legality`,
        kind: 'evidence',
        label: `Graph legality · ${attempt.graphLegality.status}`,
        description: attempt.graphLegality.reason,
        workspace: 'graph',
        source: 'real',
        badge: attempt.graphLegality.status,
        target: { transformationAttemptId: attempt.id, graphNodeIds },
      },
    ],
  };
}

function graphNodeExplorerNode(
  graphNode: GraphNode,
  graph: Graph,
  candidates: readonly RewriteCandidate[],
  attempts: readonly TransformationAttempt[],
  modelItemId: string,
  path: string,
): ExplorerNode {
  const operator = getOperator(graphNode.operatorId);
  const relatedCandidates = candidates.filter((candidate) => candidateImplementationNodeIds(candidate, graph).includes(graphNode.id));
  const relatedAttempts = attempts.filter((attempt) => attemptGraphNodeIds(attempt, graph).includes(graphNode.id));
  const candidateAttemptIds = new Set(relatedCandidates.map(({ attemptId }) => attemptId));
  const transformationChildren: ExplorerNode[] = relatedCandidates.map((candidate) => {
    const attempt = attempts.find(({ id }) => id === candidate.attemptId);
    return {
      id: `${path}:candidate:${safeId(candidate.id)}`,
      kind: 'rewrite-candidate',
      label: candidate.ruleName,
      description: `${candidate.ruleId} · ${candidate.summary} · ${candidate.description}`,
      workspace: 'graph',
      source: 'real',
      badge: candidate.exactness,
      target: {
        modelItemId,
        graphNodeId: candidateImplementationNodeIds(candidate, graph)[0] ?? graphNode.id,
        graphNodeIds: candidate.affectedNodeIds,
        rewriteCandidateId: candidate.id,
        transformationAttemptId: candidate.attemptId,
      },
      children: attempt ? [attemptExplorerNode(attempt, graph, `${path}:candidate:${safeId(candidate.id)}`)] : [],
    };
  });
  transformationChildren.push(...relatedAttempts
    .filter(({ id }) => !candidateAttemptIds.has(id))
    .map((attempt) => attemptExplorerNode(attempt, graph, path)));

  const propertyChildren: ExplorerNode[] = operator?.propertyClaims.map((claim, index) => ({
    id: `${path}:property:${safeId(String(claim.kind))}:${index}`,
    kind: 'property',
    label: String(claim.kind),
    description: claim.justification ?? `${operator.name} declares ${String(claim.kind)} semantics.`,
    workspace: 'graph',
    source: 'real',
    target: { modelItemId, graphNodeId: graphNode.id, graphNodeIds: [graphNode.id] },
  })) ?? [];

  return {
    id: `${path}:graph-node:${safeId(graphNode.id)}`,
    kind: 'graph-node',
    label: `${operator?.name ?? graphNode.operatorId} · ${graphNode.id}`,
    description: operator?.meaning,
    workspace: 'graph',
    source: 'real',
    target: { modelItemId, graphNodeId: graphNode.id, graphNodeIds: [graphNode.id] },
    children: [
      {
        id: `${path}:graph-node:${safeId(graphNode.id)}:properties`,
        kind: 'property-group',
        label: 'Properties',
        workspace: 'graph',
        source: 'real',
        target: { modelItemId, graphNodeId: graphNode.id, graphNodeIds: [graphNode.id] },
        children: propertyChildren,
      },
      {
        id: `${path}:graph-node:${safeId(graphNode.id)}:transformations`,
        kind: 'transformation-group',
        label: 'Transformations',
        workspace: 'graph',
        source: 'real',
        target: { modelItemId, graphNodeId: graphNode.id, graphNodeIds: [graphNode.id] },
        children: transformationChildren,
      },
      {
        id: `${path}:graph-node:${safeId(graphNode.id)}:evidence`,
        kind: 'evidence',
        label: 'Evidence',
        description: relatedAttempts.length > 0 ? `${relatedAttempts.length} semantic attempt record(s)` : 'No transformation evidence for this graph node.',
        workspace: 'graph',
        source: 'real',
        target: { modelItemId, graphNodeId: graphNode.id, graphNodeIds: [graphNode.id] },
      },
    ],
  };
}


function buildModelItemNode(
  item: ModelExplorerItem,
  input: BuildModelRootedExplorerInput,
  parentPath: string,
  layerOrderById: ReadonlyMap<string, ModelLayerOrderEntry>,
): ExplorerNode {
  const path = `${parentPath}:model-item:${safeId(item.id)}`;
  const graphNodeIds = uniqueGraphNodeIds(item.graphNodeIds, input.graph);
  const graphNodes = graphNodeIds.map((id) => input.graph.nodes.find((node) => node.id === id)).filter((node): node is GraphNode => Boolean(node));
  const relatedCandidate = input.rewriteCandidates
    .filter((candidate) => candidateImplementationNodeIds(candidate, input.graph).some((id) => graphNodeIds.includes(id)))
    .sort((left, right) => candidateImplementationNodeIds(right, input.graph).length - candidateImplementationNodeIds(left, input.graph).length)[0];
  const relatedGraphNodeIds = relatedCandidate ? candidateImplementationNodeIds(relatedCandidate, input.graph) : graphNodeIds;
  const kernelId = relatedCandidate ? `kernel:${relatedCandidate.attemptId}` : `kernel:${item.id}`;
  const kernelLabel = graphNodes.some(({ operatorId }) => operatorId === 'matmul') || relatedCandidate?.ruleName.includes('Linear')
    ? 'gemm_epilogue_scale'
    : `${graphNodes[0]?.operatorId ?? 'projected'}_kernel_0`;
  const runtimeEventId = 'runtime-event-0';
  const commonTarget = { modelItemId: item.id, graphNodeIds: relatedGraphNodeIds } as const;

  const operators: ExplorerNode = {
    id: `${path}:operators`, kind: 'section', label: 'Operators', badge: String(graphNodes.length),
    description: `Operator composition of ${item.label}`, action: 'add-operator',
    target: { modelItemId: item.id, graphNodeIds },
    children: operatorSectionNodes(input.graph).filter((node) => graphNodeIds.includes(node.target!.graphNodeId!)).map((node) => ({
      ...node, id: `${path}:${node.id}`, target: { ...node.target, modelItemId: item.id },
    })),
  };
  const graphRepresentation: ExplorerNode = {
    id: `${path}:representation:graph`, kind: 'representation', label: 'Graph', workspace: 'graph', source: graphNodes.length > 0 ? 'real' : 'placeholder',
    target: { modelItemId: item.id, graphNodeIds, graphNodeId: graphNodeIds[0] },
    children: graphNodes.map((node) => graphNodeExplorerNode(node, input.graph, input.rewriteCandidates, input.transformationAttempts, item.id, path)),
  };
  const kernelRepresentation: ExplorerNode = {
    id: `${path}:representation:kernel`, kind: 'representation', label: 'Kernel', workspace: 'kernel', source: 'placeholder',
    target: { ...commonTarget, kernelId },
    children: [{
      id: `${path}:kernel:${safeId(kernelId)}`, kind: 'kernel', label: kernelLabel, description: relatedCandidate ? `Projected implementation for ${relatedCandidate.affectedNodeIds.join(', ')}.` : 'Backend kernel mapping is not connected.',
      workspace: 'kernel', source: 'placeholder', target: { ...commonTarget, kernelId },
      children: ['Lowering', 'Tiling', 'Memory', 'Artifacts'].map((label) => ({
        id: `${path}:kernel:${safeId(kernelId)}:${safeId(label)}`, kind: 'kernel-detail' as const, label,
        workspace: 'kernel' as const, source: 'placeholder' as const, target: { ...commonTarget, kernelId },
      })),
    }],
  };
  const runtimeRepresentation: ExplorerNode = {
    id: `${path}:representation:runtime`, kind: 'representation', label: 'Runtime', workspace: 'runtime', source: 'placeholder',
    target: { ...commonTarget, kernelId, runtimeEventId },
    children: [{
      id: `${path}:runtime:${safeId(runtimeEventId)}`, kind: 'runtime-event', label: `Launch · ${kernelLabel}`, workspace: 'runtime', source: 'placeholder',
      target: { ...commonTarget, kernelId, runtimeEventId },
      children: ['Stream 0', 'Buffers', 'Timing'].map((label) => ({
        id: `${path}:runtime:${safeId(runtimeEventId)}:${safeId(label)}`, kind: 'runtime-detail' as const, label,
        workspace: 'runtime' as const, source: 'placeholder' as const, target: { ...commonTarget, kernelId, runtimeEventId },
      })),
    }],
  };
  const hardwareRepresentation: ExplorerNode = {
    id: `${path}:representation:hardware`, kind: 'representation', label: 'Hardware', workspace: 'hardware', source: 'placeholder',
    target: { ...commonTarget, kernelId, runtimeEventId },
    children: [
      ['registers', 'Registers'], ['dram', 'DRAM Traffic'], ['occupancy', 'Occupancy'], ['latency', 'Latency'],
    ].map(([hardwareMetricId, label]) => ({
      id: `${path}:hardware:${hardwareMetricId}`, kind: 'hardware-metric' as const, label, workspace: 'hardware' as const, source: 'placeholder' as const,
      target: { ...commonTarget, kernelId, runtimeEventId, hardwareMetricId },
    })),
  };
  const nestedModelItems = item.children?.map((child) => buildModelItemNode(child, input, path, layerOrderById)) ?? [];
  const layerOrder = layerOrderById.get(item.id);

  return {
    id: path,
    kind: item.kind === 'module' ? 'model-module' : 'model-layer',
    label: item.label,
    description: `${graphNodeIds.length} related graph node(s)`,
    workspace: 'model',
    source: item.source,
    ...(layerOrder ? { order: layerOrder.order, orderSource: layerOrder.source } : {}),
    target: { modelItemId: item.id, graphNodeIds },
    children: [...nestedModelItems, operators, graphRepresentation, kernelRepresentation, runtimeRepresentation, hardwareRepresentation],
  };
}

function annotateSharedKernelReferences(nodes: readonly ExplorerNode[]): ExplorerNode[] {
  const referenceCounts = new Map<string, number>();
  for (const node of flattenExplorerNodes(nodes)) {
    const kernelId = node.kind === 'kernel' ? node.target?.kernelId : undefined;
    if (kernelId) referenceCounts.set(kernelId, (referenceCounts.get(kernelId) ?? 0) + 1);
  }

  const annotate = (branch: readonly ExplorerNode[]): ExplorerNode[] => branch.map((node) => {
    const children = node.children ? annotate(node.children) : undefined;
    const kernelId = node.kind === 'kernel' ? node.target?.kernelId : undefined;
    const sharedReferenceCount = kernelId ? referenceCounts.get(kernelId) : undefined;
    return sharedReferenceCount && sharedReferenceCount > 1
      ? { ...node, shared: true, sharedReferenceCount, children }
      : { ...node, children };
  });

  return annotate(nodes);
}

function operatorSectionNodes(graph: Graph): ExplorerNode[] {
  const counters = new Map<string, number>();
  return graph.nodes.map((node) => {
    const operatorName = getOperator(node.operatorId)?.name ?? node.operatorId;
    const index = counters.get(node.operatorId) ?? 0;
    counters.set(node.operatorId, index + 1);
    return {
      id: `operator:${safeId(graph.id)}:${safeId(node.id)}`,
      kind: 'operator' as const,
      label: `${operatorName}_${index}`,
      description: `GraphNode ${node.id} · ${node.operatorId}`,
      workspace: 'graph' as const,
      source: 'real' as const,
      target: { graphId: graph.id, graphNodeId: node.id, graphNodeIds: [node.id] },
    };
  });
}

function graphSectionNodes(graph: Graph, graphDocuments?: readonly GraphDocument[]): ExplorerNode[] {
  const documents = graphDocuments?.length ? graphDocuments : undefined;
  const graphs = documents?.map(({ graph: documentGraph }) => documentGraph) ?? [graph];
  return graphs.map((item) => ({
    id: `graph-document:${safeId(item.id)}`,
    kind: 'graph-document',
    label: item.name,
    description: `${item.nodes.length} nodes · ${item.edges.length} edges`,
    workspace: 'graph',
    source: 'real',
    target: { graphId: item.id },
    children: [{ id: `graph-document:${safeId(item.id)}:outputs`, kind: 'section', label: 'Outputs', badge: String(item.outputs.length), children: outputSectionNodes(item) }],
  }));
}

function outputSectionNodes(graph: Graph): ExplorerNode[] {
  return graph.outputs.flatMap((graphNodeId) => {
    const node = graph.nodes.find(({ id }) => id === graphNodeId);
    if (!node) return [];
    return [{
      id: `model-output:${safeId(graph.id)}:${safeId(graphNodeId)}`,
      kind: 'model-output' as const,
      label: `Output · ${graphNodeId}`,
      description: getOperator(node.operatorId)?.name ?? node.operatorId,
      workspace: 'graph' as const,
      source: 'real' as const,
      target: { graphId: graph.id, graphNodeId, graphNodeIds: [graphNodeId] },
    }];
  });
}

export function buildModelRootedExplorer(input: BuildModelRootedExplorerInput): ExplorerNode[] {
  const modelItems = input.modelItems ?? deriveModelItemsFromGraph(input.graph);
  const layerOrdering = deriveModelLayerOrdering(input.graph, modelItems);
  const layerOrderById = new Map(layerOrdering.layers.map((entry) => [entry.modelItemId, entry]));
  const arrangedModelItems = arrangeTopLevelModelItems(modelItems, layerOrdering);
  const rootId = modelRootExplorerNodeId(input.modelId ?? input.graph.id);
  const modelSummary = summarizeModelRoot(input.graph, modelItems);
  const layerNodes = arrangedModelItems.map((item) => buildModelItemNode(item, input, rootId, layerOrderById));
  const sectionNodes: ExplorerNode[] = [
    {
      id: `${rootId}:section:layers`, kind: 'section', label: 'Layers', description: 'Model-level abstractions', badge: String(layerNodes.length), action: 'add-layer', children: layerNodes,
    },
    {
      id: `${rootId}:section:graphs`, kind: 'section', label: 'Model Graph', description: 'Complete model graph', badge: String(input.graphDocuments?.length ?? 1), children: graphSectionNodes(input.graph, input.graphDocuments),
    },
  ];
  const tree: ExplorerNode[] = [{
    id: rootId,
    kind: 'model-root',
    label: input.modelName ?? input.graph.name ?? 'ExampleModel',
    description: [
      countLabel(modelSummary.layerCount, 'layer'),
      countLabel(modelSummary.graphNodeCount, 'graph node'),
      countLabel(modelSummary.inputCount, 'input'),
      countLabel(modelSummary.outputCount, 'output'),
    ].join(' · '),
    workspace: 'model',
    source: 'derived',
    modelSummary,
    target: { modelItemId: rootId },
    children: sectionNodes,
  }];
  return annotateSharedKernelReferences(tree);
}

export function flattenExplorerNodes(nodes: readonly ExplorerNode[]): ExplorerNode[] {
  return nodes.flatMap((node) => [node, ...flattenExplorerNodes(node.children ?? [])]);
}
