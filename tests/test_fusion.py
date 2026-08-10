from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners


def _compile_and_capture(model):
    decisions = []
    optimized_modules = []

    def listener(event, payload):
        if event == "optimization.decision":
            decisions.append(payload)
        elif event == "ir.optimized":
            optimized_modules.append(payload)

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
    assert len(optimized_modules) == 1

    return decisions[0], optimized_modules[0], executable


def test_gemm_bias_relu_is_rewritten_when_legal():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    decision, module, executable = _compile_and_capture(model)

    assert decision["candidate"] == "gemm_bias_relu"
    assert decision["found"] is True
    assert decision["legal"] is True
    assert decision["profitable"] is None
    assert decision["selected"] is True
    assert decision["reason"] == (
        "selected by fixed fusion policy; IR rewrite applied"
    )

    assert [op.name for op in module.ops] == [
        "fused_gemm_bias_relu",
    ]

    fused = module.ops[0]
    assert len(fused.operands) == 3
    assert fused.operands[0] is module.inputs[0]
    assert fused.operands[1] is module.parameters[0]
    assert fused.operands[2] is module.parameters[1]
    assert fused.results == module.outputs

    assert executable.run()["kernels"] == [
        "kernel_0_fused_gemm_bias_relu",
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

    decision, module, executable = _compile_and_capture(model)

    assert decision["candidate"] == "gemm_bias_relu"
    assert decision["found"] is True
    assert decision["legal"] is False
    assert decision["profitable"] is None
    assert decision["selected"] is False
    assert decision["reason"] == "bias_add result has multiple uses"

    assert [op.name for op in module.ops] == [
        "gemm",
        "bias_add",
        "relu",
        "relu",
    ]

    assert executable.run()["kernels"] == [
        "kernel_0_gemm",
        "kernel_1_bias_add",
        "kernel_2_relu",
        "kernel_3_relu",
    ]