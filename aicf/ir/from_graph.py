from __future__ import annotations
from .module import IRModule
from .operation import Operation
from .value import IRValue
from .type import TensorType
from ..graph.graph import Graph


def graph_to_ir(graph: Graph) -> IRModule:
    module = IRModule()
    values = {}
    for v in graph.inputs:
        iv = IRValue(f"%{v.id}", TensorType(v.spec.shape, v.spec.dtype))
        values[v.id] = iv
        module.inputs.append(iv)

    for node in graph.nodes:
        operands = [values[v.id] for v in node.inputs]
        results = []
        for out in node.outputs:
            iv = IRValue(f"%{out.id}", TensorType(out.spec.shape, out.spec.dtype))
            values[out.id] = iv
            results.append(iv)
        module.ops.append(Operation(node.op, operands, results, dict(node.attrs)))

    module.outputs = [values[v.id] for v in graph.outputs]
    return module
