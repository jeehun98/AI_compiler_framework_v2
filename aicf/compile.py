from __future__ import annotations
from .context import CompileContext
from .graph.builder import capture_graph
from .graph.verifier import verify_graph
from .ir.from_graph import graph_to_ir
from .compiler.pass_manager import PassManager
from .compiler.passes.canonicalize import CanonicalizePass
from .compiler.passes.fusion import FusionPass
from .lowering.cuda.lower import lower_to_cuda
from .backend.cuda.codegen.generator import codegen
from .runtime.executable import Executable
from .diagnostics.events import emit


def compile(model_fn, input_specs, *, target="cuda", diagnostics=True):
    context = CompileContext(target=target, diagnostics=diagnostics)

    with capture_graph() as builder:
        inputs = [builder.input(spec) for spec in input_specs]
        output = model_fn(*inputs)
        builder.output(output)
        graph = builder.graph

    verify_graph(graph)
    emit(context, "graph.captured", graph)

    module = graph_to_ir(graph)
    emit(context, "ir.created", module)

    pm = PassManager([CanonicalizePass(), FusionPass()])
    module = pm.run(module, context)
    emit(context, "ir.optimized", module)

    if target != "cuda":
        raise NotImplementedError(f"target not supported yet: {target}")

    lowered = lower_to_cuda(module, context)
    emit(context, "lowering.finished", lowered)

    image = codegen(lowered)
    emit(context, "backend.codegen", image)
    return Executable(image)
