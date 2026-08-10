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
from .nn.module import Module


def compile(model, input_specs, *, target="cuda", diagnostics=True):
    context = CompileContext(target=target, diagnostics=diagnostics)

    with capture_graph() as builder:
        inputs = [builder.input(spec) for spec in input_specs]

        if isinstance(model, Module):
            builder.bind_parameters(model.named_parameters())

        output = model(*inputs)
        builder.output(output)
        graph = builder.graph

    verify_graph(graph)
    emit(context, "graph.captured", graph)
    emit(context, "graph.use_def", graph)

    module = graph_to_ir(graph)
    emit(context, "ir.created", module)

    ir_use_def = context.analyses.use_def(module)
    emit(
        context,
        "ir.use_def",
        {"module": module, "analysis": ir_use_def},
    )

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