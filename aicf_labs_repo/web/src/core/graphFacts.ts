import type { Dimension, Graph, GraphEdge, GraphNode, InputPortId, TensorShape } from '../domain/graph';

export function getIncomingEdge(graph: Graph, nodeId: string, inputPort: InputPortId): GraphEdge | undefined {
  return graph.edges.find((edge) => edge.targetNodeId === nodeId && edge.targetPort === inputPort);
}

export function getProducer(graph: Graph, nodeId: string, inputPort: InputPortId): GraphNode | undefined {
  const producerId = getIncomingEdge(graph, nodeId, inputPort)?.sourceNodeId;
  return producerId ? graph.nodes.find(({ id }) => id === producerId) : undefined;
}

export function getConsumers(graph: Graph, nodeId: string): GraphNode[] {
  const consumerIds = new Set(
    graph.edges
      .filter(({ sourceNodeId }) => sourceNodeId === nodeId)
      .map(({ targetNodeId }) => targetNodeId),
  );
  return graph.nodes.filter(({ id }) => consumerIds.has(id));
}

export function getConsumerCount(graph: Graph, nodeId: string): number {
  return getConsumers(graph, nodeId).length;
}

export function isGraphOutput(graph: Graph, nodeId: string): boolean {
  return graph.outputs.includes(nodeId);
}

function dimensionsEqual(left: Dimension, right: Dimension): boolean {
  return left === right;
}

function broadcastShapes(left: TensorShape, right: TensorShape): TensorShape | undefined {
  const rank = Math.max(left.length, right.length);
  const output: Dimension[] = [];
  for (let offset = 1; offset <= rank; offset += 1) {
    const leftDimension = left.at(-offset) ?? 1;
    const rightDimension = right.at(-offset) ?? 1;
    if (leftDimension === 1) output.unshift(rightDimension);
    else if (rightDimension === 1 || dimensionsEqual(leftDimension, rightDimension)) output.unshift(leftDimension);
    else return undefined;
  }
  return output;
}

/**
 * Infer only the small shape subset represented by the browser IR. The graph has
 * no dtype, alias, or mutation model; those facts deliberately remain outside
 * this ABSTRACT_REAL reasoning pass.
 */
export function inferNodeShape(graph: Graph, nodeId: string): TensorShape | undefined {
  const memo = new Map<string, TensorShape | undefined>();
  const visiting = new Set<string>();

  const infer = (currentId: string): TensorShape | undefined => {
    if (memo.has(currentId)) return memo.get(currentId);
    if (visiting.has(currentId)) return undefined;
    visiting.add(currentId);
    const node = graph.nodes.find(({ id }) => id === currentId);
    let shape: TensorShape | undefined;

    if (!node) {
      shape = undefined;
    } else if (node.operatorId === 'input') {
      shape = 'shape' in node.parameters && Array.isArray(node.parameters.shape)
        ? [...node.parameters.shape]
        : undefined;
    } else if (node.operatorId === 'constant') {
      shape = 'value' in node.parameters && Number.isFinite(node.parameters.value) ? [] : undefined;
    } else {
      const inputs = (portCount: number) => Array.from(
        { length: portCount },
        (_, index) => getIncomingEdge(graph, node.id, `in-${index}`)?.sourceNodeId,
      );
      switch (node.operatorId) {
        case 'add':
        case 'mul': {
          const [leftId, rightId] = inputs(2);
          const left = leftId ? infer(leftId) : undefined;
          const right = rightId ? infer(rightId) : undefined;
          shape = left && right ? broadcastShapes(left, right) : undefined;
          break;
        }
        case 'matmul': {
          const [leftId, rightId] = inputs(2);
          const left = leftId ? infer(leftId) : undefined;
          const right = rightId ? infer(rightId) : undefined;
          shape = left?.length === 2 && right?.length === 2 && dimensionsEqual(left[1], right[0])
            ? [left[0], right[1]]
            : undefined;
          break;
        }
        case 'relu': {
          const [inputId] = inputs(1);
          shape = inputId ? infer(inputId) : undefined;
          break;
        }
        case 'transpose': {
          const [inputId] = inputs(1);
          const input = inputId ? infer(inputId) : undefined;
          if (input && input.length >= 2) {
            shape = [...input.slice(0, -2), input.at(-1) as Dimension, input.at(-2) as Dimension];
          }
          break;
        }
        case 'reduceSum': {
          const [inputId] = inputs(1);
          const input = inputId ? infer(inputId) : undefined;
          if (input && 'axis' in node.parameters && 'keepDims' in node.parameters) {
            const axis = node.parameters.axis;
            const keepDims = node.parameters.keepDims;
            if (axis === 'all') {
              shape = keepDims ? input.map(() => 1) : [];
            } else if (axis >= 0 && axis < input.length) {
              shape = keepDims
                ? input.map((dimension, index) => index === axis ? 1 : dimension)
                : input.filter((_, index) => index !== axis);
            }
          }
          break;
        }
      }
    }

    visiting.delete(currentId);
    memo.set(currentId, shape);
    return shape;
  };

  return infer(nodeId);
}
