from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners


def _compile_and_capture_decision(model):
    decisions = []

    def listener(event, payload):
        if event == "optimization.decision":
            decisions.append(payload)

    clear_listeners()
    add_listener(listener)

    try:
        executable = compile(
            model,
            [TensorSpec((2, 4), "float32", "x")],
        )
    finally:
        clear_listeners()

    assert len(decisions) == 1

    return decisions[0], executable


def test_gemm_bias_relu_is_legal_candidate():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    decision, executable = _compile_and_capture_decision(model)

    assert decision == {
        "candidate": "gemm_bias_relu",
        "found": True,
        "legal": True,
        "profitable": None,
        "selected": False,
        "reason": (
            "legal candidate; "
            "profitability and rewrite not implemented"
        ),
    }

    # v0.5 only decides legality.
    # The IR is still not rewritten.
    assert executable.run()["kernels"] == [
        "kernel_0_gemm",
        "kernel_1_bias_add",
        "kernel_2_relu",
    ]


def test_gemm_bias_relu_rejected_when_bias_result_has_multiple_uses():
    class MultiUseBiasModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.linear = nn.Linear(4, 8)
            self.relu_a = nn.ReLU()
            self.relu_b = nn.ReLU()

        def forward(self, x):
            y = self.linear(x)

            out = self.relu_a(y)
            self.relu_b(y)

            return out

    model = MultiUseBiasModel()

    decision, executable = _compile_and_capture_decision(model)

    assert decision == {
        "candidate": "gemm_bias_relu",
        "found": True,
        "legal": False,
        "profitable": None,
        "selected": False,
        "reason": "bias_add result has multiple uses",
    }

    # No rewrite or DCE yet, so both ReLU kernels remain.
    assert executable.run()["kernels"] == [
        "kernel_0_gemm",
        "kernel_1_bias_add",
        "kernel_2_relu",
        "kernel_3_relu",
    ]