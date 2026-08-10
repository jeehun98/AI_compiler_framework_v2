"""Sequential-style model declaration through the user-facing nn API."""
from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners
from aicf.diagnostics.graph_dump import format_graph
from aicf.ir.printer import format_ir


model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
)


def print_event(event, payload):
    print(f"\n[{event}]")
    if event == "graph.captured":
        print(format_graph(payload))
    elif event.startswith("ir.") or event.startswith("pass."):
        module = payload.get("module") if isinstance(payload, dict) else payload
        print(format_ir(module))
    else:
        print(payload)


if __name__ == "__main__":
    print("[model]")
    print(model)

    print("\n[parameters]")
    for name, parameter in model.named_parameters():
        print(f"{name}: shape={parameter.shape}, dtype={parameter.dtype}")

    add_listener(print_event)
    exe = compile(
        model,
        [TensorSpec((32, 64), "float32", "x")],
    )
    print("\n[runtime]")
    print(exe.run())
    clear_listeners()
