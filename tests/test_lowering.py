from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners


def _compile_and_capture_lowering(model):
    lowered = []

    def listener(event, payload):
        if event == "lowering.finished":
            lowered.append(payload)

    clear_listeners()
    add_listener(listener)

    try:
        executable = compile(
            model,
            [TensorSpec((2, 4), "float32", "x")],
        )
    finally:
        clear_listeners()

    assert len(lowered) == 1
    return lowered[0], executable


def test_fused_op_uses_cuda_epilogue_lowering_rule():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    lowered, executable = _compile_and_capture_lowering(model)

    assert len(lowered.kernels) == 1

    plan = lowered.kernels[0]
    assert plan.name == "kernel_0_fused_gemm_bias_relu"
    assert plan.source_op == "fused_gemm_bias_relu"
    assert plan.strategy == "gemm_epilogue_bias_relu"
    assert plan.inputs == ("%0", "%1", "%2")
    assert plan.outputs == ("%5",)

    assert plan.problem is not None
    assert plan.problem.m == 2
    assert plan.problem.n == 8
    assert plan.problem.k == 4
    assert plan.problem.dtype == "float32"

    assert plan.attrs["fused_ops"] == (
        "gemm",
        "bias_add",
        "relu",
    )

    assert executable.run()["kernels"] == [
        "kernel_0_fused_gemm_bias_relu",
    ]


def test_unfused_multi_use_path_uses_separate_cuda_rules():
    class MultiUseBiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 8)
            self.relu_a = nn.ReLU()
            self.relu_b = nn.ReLU()

        def forward(self, x):
            y = self.linear(x)
            out = self.relu_a(y)
            self.relu_b(y)
            return out

    lowered, executable = _compile_and_capture_lowering(
        MultiUseBiasModel()
    )

    assert [plan.source_op for plan in lowered.kernels] == [
        "gemm",
        "bias_add",
        "relu",
        "relu",
    ]

    assert [plan.strategy for plan in lowered.kernels] == [
        "gemm",
        "elementwise_bias_add",
        "elementwise_relu",
        "elementwise_relu",
    ]

    gemm_plan = lowered.kernels[0]
    assert gemm_plan.problem is not None
    assert (
        gemm_plan.problem.m,
        gemm_plan.problem.n,
        gemm_plan.problem.k,
    ) == (2, 8, 4)

    assert all(
        plan.problem is None
        for plan in lowered.kernels[1:]
    )

    assert executable.run()["kernels"] == [
        "kernel_0_gemm",
        "kernel_1_bias_add",
        "kernel_2_relu",
        "kernel_3_relu",
    ]