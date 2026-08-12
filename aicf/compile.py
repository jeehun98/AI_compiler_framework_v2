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
from .backend.cuda.compiler import compile_with_nvrtc
from .backend.cuda.driver import load_with_cuda_driver
from .runtime.bindings import build_runtime_signature
from .runtime.executable import Executable
from .diagnostics.events import emit
from .nn.module import Module


def compile(
    model,
    input_specs,
    *,
    target="cuda",
    diagnostics=True,
    cuda_compile=False,
    cuda_arch=None,
    nvrtc_library=None,
    cuda_load=False,
    cuda_device=0,
    cuda_driver_library=None,
):
    context = CompileContext(target=target, diagnostics=diagnostics)
    input_specs = tuple(input_specs)

    named_parameters = (
        tuple(model.named_parameters())
        if isinstance(model, Module)
        else ()
    )

    with capture_graph() as builder:
        inputs = [builder.input(spec) for spec in input_specs]

        if isinstance(model, Module):
            builder.bind_parameters(named_parameters)

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

    if cuda_load and not cuda_compile:
        raise ValueError("cuda_load=True requires cuda_compile=True")

    compiled_image = None
    if cuda_compile:
        compiled_image = compile_with_nvrtc(
            image,
            arch=cuda_arch,
            library_path=nvrtc_library,
        )
        emit(context, "backend.compiled", compiled_image)

    loaded_image = None
    if cuda_load:
        loaded_image = load_with_cuda_driver(
            compiled_image,
            device_ordinal=cuda_device,
            library_path=cuda_driver_library,
        )
        emit(context, "backend.loaded", loaded_image)

    signature = build_runtime_signature(
        module,
        input_specs,
        named_parameters,
    )

    return Executable(
        image,
        signature,
        compiled_image=compiled_image,
        loaded_image=loaded_image,
    )