import numpy as np

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

    assert plan.tile is not None
    assert plan.tile.block_m == 128
    assert plan.tile.block_n == 128
    assert plan.tile.block_k == 32

    assert plan.schedule is not None
    assert plan.schedule.grid_m == 1
    assert plan.schedule.grid_n == 1
    assert plan.schedule.k_tiles == 1

    assert plan.block_mapping is not None
    assert plan.block_mapping.threads == 256
    assert plan.block_mapping.warps == 8
    assert plan.block_mapping.outputs_per_thread == 64

    assert plan.thread_mapping is not None
    assert plan.thread_mapping.traversal == "linear_strided"
    assert plan.thread_mapping.output_order == "row_major"

    assert plan.control_flow is not None
    assert plan.control_flow.output_traversal == "linear_strided"
    assert plan.control_flow.output_guard is True
    assert plan.control_flow.k_traversal == "tile_then_inner"
    assert plan.control_flow.k_tail_guard is True

    assert plan.epilogue is not None
    assert plan.epilogue.bias is True
    assert plan.epilogue.activation == "relu"

    assert plan.attrs["fused_ops"] == (
        "gemm",
        "bias_add",
        "relu",
    )

    runtime_x = np.zeros((2, 4), dtype=np.float32)
    assert executable.run(runtime_x)["kernels"] == [
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

    assert gemm_plan.tile is not None
    assert (
        gemm_plan.tile.block_m,
        gemm_plan.tile.block_n,
        gemm_plan.tile.block_k,
    ) == (128, 128, 32)

    assert gemm_plan.schedule is not None
    assert (
        gemm_plan.schedule.grid_m,
        gemm_plan.schedule.grid_n,
        gemm_plan.schedule.k_tiles,
    ) == (1, 1, 1)

    assert all(
        plan.problem is None
        for plan in lowered.kernels[1:]
    )

    assert all(
        plan.tile is None
        for plan in lowered.kernels[1:]
    )

    assert all(
        plan.schedule is None
        for plan in lowered.kernels[1:]
    )

    assert gemm_plan.block_mapping is not None
    assert (
        gemm_plan.block_mapping.threads,
        gemm_plan.block_mapping.warps,
        gemm_plan.block_mapping.outputs_per_thread,
    ) == (256, 8, 64)

    assert all(
        plan.block_mapping is None
        for plan in lowered.kernels[1:]
    )

    assert gemm_plan.thread_mapping is not None
    assert all(
        plan.thread_mapping is None
        for plan in lowered.kernels[1:]
    )

    assert gemm_plan.control_flow is not None
    assert all(
        plan.control_flow is None
        for plan in lowered.kernels[1:]
    )

    assert gemm_plan.epilogue is not None
    assert gemm_plan.epilogue.bias is False
    assert gemm_plan.epilogue.activation is None
    assert all(
        plan.epilogue is None
        for plan in lowered.kernels[1:]
    )

    runtime_x = np.zeros((2, 4), dtype=np.float32)
    assert executable.run(runtime_x)["kernels"] == [
        "kernel_0_gemm",
        "kernel_1_bias_add",
        "kernel_2_relu",
        "kernel_3_relu",
    ]


def test_gemm_schedule_uses_ceiling_division():
    model = nn.Sequential(
        nn.Linear(130, 257, bias=False),
    )

    lowered = []

    def listener(event, payload):
        if event == "lowering.finished":
            lowered.append(payload)

    clear_listeners()
    add_listener(listener)

    try:
        compile(
            model,
            [TensorSpec((129, 130), "float32", "x")],
        )
    finally:
        clear_listeners()

    assert len(lowered) == 1
    assert len(lowered[0].kernels) == 1

    plan = lowered[0].kernels[0]

    assert plan.problem is not None
    assert (
        plan.problem.m,
        plan.problem.n,
        plan.problem.k,
    ) == (129, 257, 130)

    assert plan.schedule is not None

    # ceil(129 / 128) = 2
    # ceil(257 / 128) = 3
    # ceil(130 / 32)  = 5
    assert (
        plan.schedule.grid_m,
        plan.schedule.grid_n,
        plan.schedule.k_tiles,
    ) == (2, 3, 5)


def test_gemm_block_mapping_covers_logical_output_tile():
    model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
    )

    lowered, _ = _compile_and_capture_lowering(model)
    plan = lowered.kernels[0]

    assert plan.tile is not None
    assert plan.block_mapping is not None

    logical_outputs = plan.tile.block_m * plan.tile.block_n
    mapped_capacity = (
        plan.block_mapping.threads
        * plan.block_mapping.outputs_per_thread
    )

    assert mapped_capacity >= logical_outputs
    assert (
        plan.block_mapping.outputs_per_thread - 1
    ) * plan.block_mapping.threads < logical_outputs


def test_gemm_thread_mapping_defines_explicit_row_major_coordinates():
    model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
    )

    lowered, _ = _compile_and_capture_lowering(model)
    plan = lowered.kernels[0]

    assert plan.thread_mapping is not None
    assert plan.thread_mapping.traversal == "linear_strided"
    assert plan.thread_mapping.output_order == "row_major"
    assert plan.thread_mapping.thread_axis == "x"
    assert plan.thread_mapping.block_m_axis == "y"
    assert plan.thread_mapping.block_n_axis == "x"

    # v0.13 contract: local_m = output // BN; local_n = output % BN.
    output = 257
    local_m = output // plan.tile.block_n
    local_n = output % plan.tile.block_n

    assert (local_m, local_n) == (2, 1)


def test_gemm_control_flow_elides_guards_for_exact_tiles():
    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
    )

    lowered = []

    def listener(event, payload):
        if event == "lowering.finished":
            lowered.append(payload)

    clear_listeners()
    add_listener(listener)

    try:
        compile(
            model,
            [TensorSpec((128, 64), "float32", "x")],
        )
    finally:
        clear_listeners()

    plan = lowered[0].kernels[0]
    assert plan.control_flow is not None
    assert plan.control_flow.output_guard is False
    assert plan.control_flow.k_tail_guard is False


def test_gemm_control_flow_marks_non_divisible_k_tail():
    model = nn.Sequential(
        nn.Linear(130, 128, bias=False),
    )

    lowered = []

    def listener(event, payload):
        if event == "lowering.finished":
            lowered.append(payload)

    clear_listeners()
    add_listener(listener)

    try:
        compile(
            model,
            [TensorSpec((128, 130), "float32", "x")],
        )
    finally:
        clear_listeners()

    plan = lowered[0].kernels[0]
    assert plan.schedule.k_tiles == 5
    assert plan.control_flow is not None
    assert plan.control_flow.output_guard is False
    assert plan.control_flow.k_tail_guard is True


def test_codegen_emits_naive_global_memory_gemm_kernel():
    model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
    )

    _, executable = _compile_and_capture_lowering(model)
    code = executable.image.code

    assert 'extern "C" __global__ void kernel_0_gemm(' in code
    assert "constexpr int M = 2;" in code
    assert "constexpr int N = 8;" in code
    assert "constexpr int K = 4;" in code
    assert "for (int output = static_cast<int>(threadIdx.x);" in code
    assert "if (row < M && col < N)" in code
    assert "for (int kt = 0; kt < K_TILES; ++kt)" in code
    assert "const int k = kt * BK + kk;" in code
    assert "if (k < K)" in code
    assert "acc += A[row * K + k] * B[k * N + col];" in code
    assert "float value = acc;" in code
    assert "C[row * N + col] = value;" in code
    assert "bias[col]" not in code


def test_codegen_emits_fused_bias_relu_epilogue():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    _, executable = _compile_and_capture_lowering(model)
    code = executable.image.code

    assert 'extern "C" __global__ void kernel_0_fused_gemm_bias_relu(' in code
    assert "const float* __restrict__ bias" in code
    assert "float value = acc + bias[col];" in code
    assert "value = value > float(0) ? value : float(0);" in code
    assert "C[row * N + col] = value;" in code


def test_codegen_elides_k_tail_guard_when_k_is_exact_multiple():
    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
    )

    lowered = []

    def listener(event, payload):
        if event == "lowering.finished":
            lowered.append(payload)

    clear_listeners()
    add_listener(listener)

    try:
        executable = compile(
            model,
            [TensorSpec((128, 64), "float32", "x")],
        )
    finally:
        clear_listeners()

    assert lowered[0].kernels[0].control_flow.k_tail_guard is False
    assert "if (k < K)" not in executable.image.code
    assert "acc += A[row * K + k] * B[k * N + col];" in executable.image.code