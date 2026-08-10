from __future__ import annotations
from .module import IRModule
from .operation import Operation
from .value import IRValue
from .type import TensorType
from ..graph.graph import Graph


def _to_ir_value(value) -> IRValue:
    return IRValue(f"%{value.id}", TensorType(value.spec.shape, value.spec.dtype))


def graph_to_ir(graph: Graph) -> IRModule:
    module = IRModule()
    values = {}

    for value in graph.inputs:
        ir_value = _to_ir_value(value)
        values[value.id] = ir_value
        module.inputs.append(ir_value)

    for value in graph.parameters:
        ir_value = _to_ir_value(value)
        values[value.id] = ir_value
        module.parameters.append(ir_value)

    for node in graph.nodes:
        operands = [values[value.id] for value in node.inputs]
        results = []
        for out in node.outputs:
            ir_value = _to_ir_value(out)
            values[out.id] = ir_value
            results.append(ir_value)
        module.ops.append(Operation(node.op, operands, results, dict(node.attrs)))

    module.outputs = [values[value.id] for value in graph.outputs]
    return module
