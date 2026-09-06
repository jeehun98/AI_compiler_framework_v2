// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useGraphStore } from './store/graphStore';
import { EXAMPLE_DOCUMENTS } from './examples/graphExamples';
import { MODEL_EXAMPLES } from './examples/modelExamples';
import { App } from './App';

class TestResizeObserver implements ResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

globalThis.ResizeObserver = TestResizeObserver;

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('App integration', () => {
  beforeEach(() => useGraphStore.getState().reset());

  function attemptRow(ruleId: string, matchNeedle = ''): HTMLElement {
    fireEvent.click(screen.getByRole('button', { name: 'Expand all' }));
    const found = Array.from(document.querySelectorAll<HTMLElement>('[data-explorer-kind="transformation-attempt"]')).find((row) =>
      row.textContent?.includes(ruleId) && row.textContent.includes(matchNeedle));
    expect(found, `missing attempt row for ${ruleId} ${matchNeedle}`).toBeDefined();
    if (!found) throw new Error(`Missing attempt row for ${ruleId}`);
    return found;
  }

  function explorerRow(kind: string, text: string): HTMLElement {
    fireEvent.click(screen.getByRole('button', { name: 'Expand all' }));
    const found = Array.from(document.querySelectorAll<HTMLElement>(`[data-explorer-kind="${kind}"]`))
      .find((row) => row.textContent?.includes(text));
    expect(found, `missing ${kind} explorer row containing ${text}`).toBeDefined();
    if (!found) throw new Error(`Missing ${kind} explorer row`);
    return found;
  }

  function workspaceTab(name: string): HTMLElement {
    return within(screen.getByRole('tablist', { name: 'Compiler workspaces' }))
      .getByRole('tab', { name: new RegExp(name) });
  }

  it('adds an operator to a Custom layer and reflects its graph selection', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '+ Add Layer' }));
    fireEvent.change(screen.getByLabelText('New layer type'), { target: { value: 'Custom' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    fireEvent.click(screen.getByRole('button', { name: '+ Add Operator' }));
    fireEvent.change(screen.getByLabelText('New operator type'), { target: { value: 'input' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(screen.getByRole('heading', { name: 'Input' })).toBeInTheDocument();
    expect(useGraphStore.getState().modelDocument.layers[0].graphNodeIds).toEqual([useGraphStore.getState().selectedNodeId]);
  });
  it('loads an example and shows its live rewrite candidate', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'x-times-one' } });

    expect(explorerRow('rewrite-candidate', 'x × 1 → x')).toBeInTheDocument();
    expect(useGraphStore.getState().document.graph.outputs).toEqual(['mul']);
  });

  it('lists and loads every catalog example through the existing dropdown', () => {
    render(<App />);
    const dropdown = screen.getByLabelText('예제 그래프') as HTMLSelectElement;
    const optionValues = Array.from(dropdown.options).map(({ value }) => value).filter(Boolean);
    expect(optionValues).toEqual(MODEL_EXAMPLES.map(({ id }) => id));

    for (const example of EXAMPLE_DOCUMENTS) {
      fireEvent.change(dropdown, { target: { value: example.id } });
      expect(useGraphStore.getState().document.graph.id).toBe(example.document.graph.id);
    }
  });

  it('shows an applicable attempt for the MatMul scale example', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });

    const row = attemptRow('scale-through-linear');
    expect(row).toHaveTextContent('APPLICABLE');
    expect(row).toHaveTextContent('ScaleThroughLinear');
  });

  it('shows a mathematical rejection and its alpha reason', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-relu-negative' } });

    const row = attemptRow('scale-through-positive-homogeneous');
    expect(row).toHaveTextContent('REJECTED');
    expect(row).toHaveTextContent('Positive homogeneity requires alpha >= 0.');
  });

  it('shows the graph rejection reason for a shared fusion producer', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'fusion-shared-producer' } });

    const row = attemptRow('embed-elementwise-producer-into-reduction', 'reducer:in-0:producer');
    expect(row).toHaveTextContent('REJECTED');
    expect(row).toHaveTextContent('The producer result has multiple consumers.');
  });

  it('shows unknown contraction-footprint reasoning for MatMul fusion', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'fusion-unknown-matmul' } });

    const row = attemptRow('embed-elementwise-producer-into-reduction', 'reducer:in-0:producer');
    expect(row).toHaveTextContent('UNKNOWN');
    expect(row).toHaveTextContent('row/column dependency regions');
  });

  it('replaces the attempt list immediately when switching examples', () => {
    render(<App />);
    const dropdown = screen.getByLabelText('예제 그래프');
    fireEvent.change(dropdown, { target: { value: 'scale-relu-negative' } });
    expect(attemptRow('scale-through-positive-homogeneous')).toHaveTextContent('REJECTED');

    fireEvent.change(dropdown, { target: { value: 'fusion-unknown-matmul' } });
    expect(document.body).not.toHaveTextContent('Positive homogeneity requires alpha >= 0.');
    expect(attemptRow('embed-elementwise-producer-into-reduction', 'reducer:in-0:producer'))
      .toHaveTextContent('UNKNOWN');
  });

  it('shows a distinct empty state when the graph has no attempts', () => {
    render(<App />);
    expect(screen.getByText('No layers yet. Use + Add Layer to create an operator composition.')).toBeInTheDocument();
  });

  it('starts in Graph workspace with the persistent model-rooted Explorer', () => {
    render(<App />);
    const graph = workspaceTab('Graph');
    expect(graph).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('MODEL-ROOTED EXPLORER')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Explorer' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByLabelText('계산 그래프 캔버스')).toBeInTheDocument();
    expect(useGraphStore.getState().activeWorkspace).toBe('graph');
  });

  it('renders derived model items under the model root with real and placeholder descendants', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });

    expect(explorerRow('model-layer', 'ScaleLayer_0')).toHaveTextContent('derived');
    expect(explorerRow('graph-node', 'Mul · scale')).toHaveTextContent('real');
    expect(explorerRow('kernel', 'gemm_epilogue_scale')).toHaveTextContent(/placeholder.*SHARED · 2/i);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'x-times-one' } });
    expect(explorerRow('kernel', 'mul_kernel_0')).not.toHaveTextContent('SHARED');
  });

  it('selects the single model root and shows the whole-model summary', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });

    const modelRoots = document.querySelectorAll<HTMLElement>('[data-explorer-kind="model-root"]');
    expect(modelRoots).toHaveLength(1);
    expect(modelRoots[0]).toHaveTextContent(/TinyScaleLinear.*derived/i);
    const orderedTreeLayers = Array.from(document.querySelectorAll<HTMLElement>('[data-explorer-kind="model-layer"][data-order-source="derived"]'));
    expect(orderedTreeLayers.map((row) => row.querySelector('.cross-layer-tree__text')?.textContent)).toEqual(['ScaleLayer_0', 'Linear_0']);
    expect(orderedTreeLayers.map((row) => row.querySelector('.cross-layer-tree__order')?.textContent)).toEqual(['01', '02']);
    fireEvent.click(modelRoots[0]!);

    expect(useGraphStore.getState().activeWorkspace).toBe('model');
    expect(useGraphStore.getState().selectedModelItemId).toBe('model-root:model-scale-matmul-left');
    expect(screen.getByText('2 layers · 1 graph')).toBeInTheDocument();
    expect(screen.getByText('Current model / document')).toBeInTheDocument();
    expect(document.querySelector('.model-root.is-selected')).not.toBeNull();
    expect(Array.from(document.querySelectorAll('.model-layer-order li strong')).map(({ textContent }) => textContent)).toEqual(
      orderedTreeLayers.map((row) => row.querySelector('.cross-layer-tree__text')?.textContent),
    );

    fireEvent.click(explorerRow('model-layer', 'ScaleLayer_0'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'model', selectedModelItemId: 'scale' });
  });

  it('adds a typed user-defined layer from the Layers section', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });

    fireEvent.click(screen.getByRole('button', { name: '+ Add Layer' }));
    fireEvent.change(screen.getByLabelText('New layer type'), { target: { value: 'ReLU' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(useGraphStore.getState().modelDocument.layers).toContainEqual(expect.objectContaining({
      name: 'ReLU_1',
      type: 'ReLU',
      source: 'user-defined',
      graphNodeIds: expect.arrayContaining([expect.any(String)]),
    }));
    expect(useGraphStore.getState().activeWorkspace).toBe('model');
    expect(screen.getAllByText('ReLU_1').length).toBeGreaterThan(0);
    expect(screen.getAllByText('USER_DEFINED').length).toBeGreaterThan(0);
  });

  it('adds an operator inside Explorer and selects it on the graph', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });

    const branch = explorerRow('model-layer', 'Linear_0').closest('.cross-layer-tree__node') as HTMLElement;
    fireEvent.click(within(branch).getByRole('button', { name: '+ Add Operator' }));
    expect(useGraphStore.getState().sidebarMode).toBe('explorer');
    fireEvent.change(screen.getByLabelText('New operator type'), { target: { value: 'transpose' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    const addedNodeId = useGraphStore.getState().selectedNodeId;
    expect(useGraphStore.getState().document.graph.nodes).toContainEqual(expect.objectContaining({ id: addedNodeId, operatorId: 'transpose' }));

    fireEvent.click(screen.getByRole('tab', { name: 'Explorer' }));
    fireEvent.click(explorerRow('operator', 'Transpose_0'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'graph', selectedNodeId: addedNodeId });
    expect(Array.from(document.querySelectorAll('.react-flow__node.selected')).some((node) => node.textContent?.includes('Transpose'))).toBe(true);
  });

  it('deletes layers and operators independently from Explorer', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });
    fireEvent.click(explorerRow('model-layer', 'Linear_0'));
    fireEvent.click(screen.getByRole('button', { name: 'Delete Linear_0' }));
    expect(useGraphStore.getState().modelDocument.layers.some(({ id }) => id === 'matmul-0')).toBe(false);
    expect(useGraphStore.getState().document.graph.nodes.some(({ id }) => id === 'matmul-0')).toBe(false);
    expect(useGraphStore.getState().selectedModelItemId).toBeNull();

    fireEvent.click(explorerRow('operator', 'ReLU_0'));
    fireEvent.click(screen.getByRole('button', { name: 'Delete operator ReLU_0' }));
    const state = useGraphStore.getState();
    expect(state.selectedNodeId).toBeNull();
    expect(state.document.graph.nodes.some(({ id }) => id === 'relu-0')).toBe(false);
    expect(state.document.graph.edges.some(({ sourceNodeId, targetNodeId }) => sourceNodeId === 'relu-0' || targetNodeId === 'relu-0')).toBe(false);
    expect(state.modelDocument.layers.find(({ id }) => id === 'relu-0')?.graphNodeIds).toEqual([]);
  });

  it('creates a Linear composition with optional bias in Explorer', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '+ Add Layer' }));
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    const state = useGraphStore.getState();
    expect(state.document.graph.nodes.map(({ operatorId }) => operatorId)).toEqual(['matmul', 'add']);
    expect(state.modelDocument.layers[0].graphNodeIds).toEqual(state.document.graph.nodes.map(({ id }) => id));
    expect(explorerRow('operator', 'MatMul_0')).toBeInTheDocument();
    expect(explorerRow('operator', 'Add_0')).toBeInTheDocument();
  });
  it('places outputs under the model graph as metadata', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });
    const graphBranch = explorerRow('graph-document', 'ForwardGraph').closest('.cross-layer-tree__node')!;
    expect(graphBranch.querySelector('[data-explorer-kind="model-output"]')).toHaveTextContent('add-1');
    expect(screen.queryByRole('button', { name: '+ Add Output' })).not.toBeInTheDocument();
  });
  it('selects the full model graph from Model Graph', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });
    fireEvent.click(explorerRow('model-layer', 'Linear_0'));
    fireEvent.click(explorerRow('graph-document', 'ForwardGraph'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'graph', selectedGraphId: 'tiny-mlp-forward' });
    expect(useGraphStore.getState().document.graph.nodes).toHaveLength(10);
    expect(screen.queryByRole('button', { name: '+ Add Graph' })).not.toBeInTheDocument();
  });
  it('distinguishes Model layer selection from Graph operator selection', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });

    fireEvent.click(explorerRow('model-layer', 'Linear_0'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'model', selectedModelItemId: 'matmul-0' });
    expect(screen.getByText('Related graph nodes').parentElement).toHaveTextContent('matmul-0');

    fireEvent.click(explorerRow('operator', 'MatMul_0'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'graph', selectedNodeId: 'matmul-0' });
    expect(Array.from(document.querySelectorAll('.react-flow__node.selected')).some((node) => node.textContent?.includes('MatMul'))).toBe(true);
  });

  it('selects the same shared kernel identity from both model layers', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });
    fireEvent.click(screen.getByRole('button', { name: 'Expand all' }));

    const sharedKernelRows = Array.from(document.querySelectorAll<HTMLElement>('[data-explorer-kind="kernel"]'))
      .filter((row) => row.textContent?.includes('gemm_epilogue_scale'));
    expect(sharedKernelRows).toHaveLength(2);
    expect(sharedKernelRows[0]).toHaveTextContent(/placeholder.*SHARED · 2/i);
    expect(sharedKernelRows[0]).toHaveAttribute('title', expect.stringContaining('Shared by 2 model layers'));
    expect(sharedKernelRows[1]).toHaveAttribute('title', expect.stringContaining('Shared by 2 model layers'));

    fireEvent.click(sharedKernelRows[0]!);
    const firstKernelId = useGraphStore.getState().selectedKernelId;
    expect(firstKernelId).toContain('kernel:');
    expect(useGraphStore.getState().probeContext.kernelId).toBe(firstKernelId);

    fireEvent.click(sharedKernelRows[1]!);
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'kernel', selectedKernelId: firstKernelId });
    expect(useGraphStore.getState().probeContext.kernelId).toBe(firstKernelId);
    expect(document.querySelector('.kernel-card.is-selected')).not.toBeNull();
  });

  it('synchronizes Graph, Kernel, Runtime, and Hardware selections from one tree', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });

    fireEvent.click(explorerRow('graph-node', 'Mul · scale'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'graph', selectedNodeId: 'scale' });
    expect(Array.from(document.querySelectorAll('.react-flow__node.selected')).some((node) => node.textContent?.includes('Mul'))).toBe(true);

    fireEvent.click(explorerRow('kernel', 'gemm_epilogue_scale'));
    expect(useGraphStore.getState().activeWorkspace).toBe('kernel');
    expect(useGraphStore.getState().selectedKernelId).toContain('kernel:');
    expect(document.querySelector('.kernel-card.is-selected')).not.toBeNull();

    fireEvent.click(explorerRow('runtime-event', 'Launch · gemm_epilogue_scale'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'runtime', selectedRuntimeEventId: 'runtime-event-0' });
    expect(document.querySelector('.timeline-event.is-selected')).not.toBeNull();

    fireEvent.click(explorerRow('hardware-metric', 'DRAM Traffic'));
    expect(useGraphStore.getState()).toMatchObject({ activeWorkspace: 'hardware', selectedHardwareMetricId: 'dram' });
    expect(document.querySelector('.metric-card.is-selected')?.textContent).toContain('DRAM Traffic');
  });

  it('keeps only Layers and Model Graph at the root', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'tiny-mlp' } });
    const sections = Array.from(document.querySelectorAll('[data-explorer-kind="section"][aria-level="2"]'));
    expect(sections.map((row) => row.querySelector('.cross-layer-tree__text')?.textContent)).toEqual(['Layers', 'Model Graph']);
    expect(screen.queryByRole('tab', { name: 'Operators' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '+ Add Operator' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '+ Add Layer' })).toBeInTheDocument();
  });
  it('links a semantic candidate to node, attempt, and observation context', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });
    fireEvent.click(explorerRow('rewrite-candidate', 'ScaleThroughLinear'));

    const state = useGraphStore.getState();
    expect(state.selectedNodeId).toBe('scale');
    expect(state.selectedTransformationAttemptId).toContain('scale-through-linear');
    expect(state.selectedLayerObservationId).toContain('semantic:candidate:');
    expect(state.probeContext.graphNodeIds).toEqual(expect.arrayContaining(['scale', 'matmul']));
    expect(screen.getByText('GRAPH INSPECTOR')).toBeInTheDocument();
  });

  it('keeps the model-rooted Explorer while rendering the Kernel View', () => {
    render(<App />);
    fireEvent.click(workspaceTab('Kernel'));

    expect(screen.getByText('MODEL-ROOTED EXPLORER')).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Operators' })).not.toBeInTheDocument();
    expect(screen.getByText('Projected GPU kernel structure')).toBeInTheDocument();
    expect(screen.queryByLabelText('계산 그래프 캔버스')).not.toBeInTheDocument();
  });

  it('keeps the Explorer while rendering the Runtime launch timeline', () => {
    render(<App />);
    fireEvent.click(workspaceTab('Runtime'));

    expect(screen.getByText('MODEL-ROOTED EXPLORER')).toBeInTheDocument();
    expect(screen.getByText('Projected launch timeline')).toBeInTheDocument();
    expect(screen.getByText('KERNEL LAUNCH SEQUENCE')).toBeInTheDocument();
  });

  it('keeps the Explorer while rendering the Hardware metric dashboard', () => {
    render(<App />);
    fireEvent.click(workspaceTab('Hardware'));

    expect(screen.getByText('MODEL-ROOTED EXPLORER')).toBeInTheDocument();
    expect(screen.getByText('Illustrative GPU metric dashboard')).toBeInTheDocument();
    expect(document.querySelectorAll('.metric-card')).toHaveLength(6);
  });

  it('uses trace cards to navigate into a workspace-specific placeholder view', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });
    fireEvent.click(screen.getByRole('button', { name: /Hardware.*−12% latency/ }));

    expect(useGraphStore.getState().activeWorkspace).toBe('hardware');
    expect(useGraphStore.getState().selectedLayerObservationId).toBe('trace:hardware');
    expect(screen.getByText('HARDWARE INSPECTOR')).toBeInTheDocument();
    expect(screen.getAllByText(/placeholder/i).length).toBeGreaterThan(0);
  });

  it('preserves transformation context and candidate selection across workspaces', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'scale-matmul-left' } });
    const candidate = explorerRow('rewrite-candidate', 'ScaleThroughLinear');
    fireEvent.click(candidate);
    const selectedCandidateId = useGraphStore.getState().selectedRewriteCandidateId;
    const selectedAttemptId = useGraphStore.getState().probeContext.transformationAttemptId;

    fireEvent.click(workspaceTab('Kernel'));
    expect(screen.getAllByText(selectedAttemptId!).length).toBeGreaterThan(0);
    fireEvent.click(workspaceTab('Graph'));

    expect(useGraphStore.getState().selectedRewriteCandidateId).toBe(selectedCandidateId);
    expect(useGraphStore.getState().selectedTransformationAttemptId).toBe(selectedAttemptId);
    expect(document.querySelector('.cross-layer-tree__row.is-selected [data-explorer-kind="rewrite-candidate"]')).not.toBeNull();
  });

  it('switches the lower workbench between performance and artifacts', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('tab', { name: 'Performance' }));
    expect(screen.getByText('Registers / thread')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Artifacts' }));
    expect(screen.getByRole('button', { name: /PTX/ })).toBeDisabled();
  });

  it('supports collapsed, compact, and expanded bottom panel states', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'collapsed bottom panel' }));
    expect(useGraphStore.getState().bottomPanelSize).toBe('collapsed');
    fireEvent.click(screen.getByRole('button', { name: 'expanded bottom panel' }));
    expect(useGraphStore.getState().bottomPanelSize).toBe('expanded');
    fireEvent.click(screen.getByRole('button', { name: 'compact bottom panel' }));
    expect(useGraphStore.getState().bottomPanelSize).toBe('compact');
  });

  it('keeps the Blob URL alive until after the download click task', () => {
    vi.useFakeTimers();
    const createObjectURL = vi.fn(() => 'blob:aicf-export');
    const revokeObjectURL = vi.fn();
    Object.defineProperties(URL, {
      createObjectURL: { configurable: true, value: createObjectURL },
      revokeObjectURL: { configurable: true, value: revokeObjectURL },
    });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: '저장' }));

    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(revokeObjectURL).not.toHaveBeenCalled();
    expect(document.querySelector('a[download]')).not.toBeNull();
    vi.runAllTimers();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:aicf-export');
    expect(document.querySelector('a[download]')).toBeNull();
  });
});
