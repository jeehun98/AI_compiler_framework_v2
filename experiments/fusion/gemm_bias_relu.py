"""Sequential-style model declaration through the user-facing nn API."""
import numpy as np

from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners
from aicf.diagnostics.graph_dump import format_graph
from aicf.diagnostics.use_def_dump import format_use_def
from aicf.diagnostics.lowering_dump import format_cuda_lowering
from aicf.ir.printer import format_ir


model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
)


def print_event(event, payload):
    print(f"\n[{event}]")
    if event == "graph.captured":
        print(format_graph(payload))
    elif event == "graph.use_def":
        print(format_use_def(payload))
    elif event.startswith("ir.") or event.startswith("pass."):
        module = payload.get("module") if isinstance(payload, dict) else payload
        print(format_ir(module))
    elif event == "lowering.finished":
        print(format_cuda_lowering(payload))
    elif event == "backend.codegen":
        print(payload.code)
    else:
        print(payload)


if __name__ == "__main__":
    print("[model]")
    print(model)

    print("\n[parameters]")
    for name, parameter in model.named_parameters():
        print(
            f"{name}: shape={parameter.shape}, dtype={parameter.dtype}, "
            f"storage_nbytes={parameter.data.nbytes}"
        )

    add_listener(print_event)
    exe = compile(
        model,
        [TensorSpec((32, 64), "float32", "x")],
    )

    x = np.zeros((32, 64), dtype=np.float32)

    print("\n[runtime]")
    print(exe.run(x))
    clear_listeners()