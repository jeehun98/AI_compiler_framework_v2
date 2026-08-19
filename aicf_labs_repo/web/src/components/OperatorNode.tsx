import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import type { OperatorId } from '../domain/operator';
import { getOperator } from '../catalog/operators';

export type GraphFlowNode = Node<
  { operatorId: OperatorId; subtitle?: string; isOutput?: boolean; hasError?: boolean },
  'operator'
>;

export function OperatorNode({ data, selected }: NodeProps<GraphFlowNode>) {
  const operator = getOperator(data.operatorId);
  if (!operator) return <div className="operator-node has-error">Unknown operator</div>;

  return (
    <div
      className={`operator-node ${selected ? 'is-selected' : ''} ${data.hasError ? 'has-error' : ''}`}
      style={{ '--node-accent': operator.accent } as React.CSSProperties}
    >
      {Array.from({ length: operator.arity }, (_, index) => (
        <Handle
          key={`in-${index}`}
          id={`in-${index}`}
          type="target"
          position={Position.Left}
          style={{ top: `${((index + 1) / (operator.arity + 1)) * 100}%` }}
        />
      ))}
      <span className="operator-node__symbol">{operator.symbol}</span>
      <span className="operator-node__copy">
        <strong>{operator.name}</strong>
        <small>{data.subtitle ?? operator.category}</small>
      </span>
      {data.isOutput && <span className="operator-node__output">OUT</span>}
      <Handle id="out" type="source" position={Position.Right} />
    </div>
  );
}
