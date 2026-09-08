// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import App from './App';
import report from '../public/report.json';

afterEach(cleanup);
describe('Python backend report viewer', () => {
  it('renders the real model, transformation and reference result', () => {
    render(<App report={report} />);
    expect(screen.getByRole('region', { name: 'Model' })).toHaveTextContent('Linear');
    expect(screen.getByText(/linear_relu · ACCEPT/)).toBeInTheDocument();
    expect(screen.getByText('원본과 변환 결과 일치')).toBeInTheDocument();
    expect(report.execution.original_output).toEqual([[3], [3]]);
    fireEvent.click(screen.getByRole('button', { name: 'linear_relu.0' }));
    const inspector = screen.getByRole('region', { name: 'Selected operator' });
    expect(inspector).toHaveTextContent('PURE');
    expect(inspector).toHaveTextContent('cuda.linear_relu.placeholder');
    expect(inspector).toHaveTextContent('not-implemented');
  });
  it('shows original edges and source-of-truth scoped semantics', () => {
    render(<App report={report} />);
    fireEvent.click(screen.getByRole('button', { name: '원본' }));
    fireEvent.click(screen.getByRole('button', { name: 'matmul.0' }));
    expect(screen.getByRole('region', { name: 'Selected operator' })).toHaveTextContent('argument 1 fixed');
    expect(screen.getByText('input.0 → matmul.0')).toBeInTheDocument();
    expect(screen.queryByText('Selected CUDA implementation')).not.toBeInTheDocument();
  });
});
