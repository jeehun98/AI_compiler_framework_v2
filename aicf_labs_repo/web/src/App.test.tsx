// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useGraphStore } from './store/graphStore';
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

  it('adds a palette operator and reflects node selection in the inspector', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: /Input/ }));
    const inputId = useGraphStore.getState().document.graph.nodes[0]?.id;
    useGraphStore.getState().selectNode(inputId ?? null);

    expect(screen.getByRole('heading', { name: 'Input' })).toBeInTheDocument();
    expect(screen.getByText('계산 그래프에 외부 텐서를 도입합니다.')).toBeInTheDocument();
  });

  it('loads an example and shows its live rewrite candidate', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('예제 그래프'), { target: { value: 'x-times-one' } });

    expect(screen.getByText('x × 1 → x')).toBeInTheDocument();
    expect(useGraphStore.getState().document.graph.outputs).toEqual(['mul']);
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
