"""Negative fusion probe: BiasAdd output has multiple users."""

from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners
from aicf.diagnostics.graph_dump import format_graph
from aicf.diagnostics.use_def_dump import format_use_def
from aicf.diagnostics.ir_use_def_dump import format_ir_use_def
from aicf.ir.printer import format_ir


class MultiUseBiasModel(nn.Module):
    """Create two ReLU users of one Linear/BiasAdd result."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 128)
        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()

    def forward(self, x):
        y = self.linear(x)

        # Both operations consume the same BiasAdd result.
        out = self.relu_a(y)
        self.relu_b(y)

        return out


model = MultiUseBiasModel()


def print_event(event, payload):
    print(f"\n[{event}]")

    if event == "graph.captured":
        print(format_graph(payload))
    elif event == "graph.use_def":
        print(format_use_def(payload))
    elif event == "ir.use_def":
        print(
            format_ir_use_def(
                payload["module"],
                payload["analysis"],
            )
        )
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
        print(
            f"{name}: "
            f"shape={parameter.shape}, "
            f"dtype={parameter.dtype}"
        )

    add_listener(print_event)

    try:
        exe = compile(
            model,
            [TensorSpec((32, 64), "float32", "x")],
        )

        print("\n[runtime]")
        print(exe.run())
    finally:
        clear_listeners()