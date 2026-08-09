"""First vertical experiment target.

This is intentionally not a benchmark framework. It simply uses the framework
and observes its internal stages.
"""
from aicf import compile
from aicf.frontend.tensor import TensorSpec
from aicf.frontend.ops import gemm, bias_add, relu
from aicf.diagnostics.events import add_listener, clear_listeners
from aicf.diagnostics.graph_dump import format_graph
from aicf.ir.printer import format_ir


def model(x, w, b):
    return relu(bias_add(gemm(x, w), b))


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
    add_listener(print_event)
    exe = compile(
        model,
        [
            TensorSpec((32, 64), "float32", "x"),
            TensorSpec((64, 128), "float32", "w"),
            TensorSpec((128,), "float32", "b"),
        ],
    )
    print("\n[runtime]")
    print(exe.run())
    clear_listeners()
