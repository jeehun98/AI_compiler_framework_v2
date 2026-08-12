from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUDAExecutableImage:
    kernels: list[str]
    code: str
    plans: list[object] = field(default_factory=list)


def _cuda_scalar_type(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    raise NotImplementedError(
        f"v0.15 CUDA source generation supports float32 only, got {dtype}"
    )


def _require_gemm_plan(plan):
    required = (
        plan.problem,
        plan.tile,
        plan.schedule,
        plan.block_mapping,
        plan.thread_mapping,
        plan.control_flow,
        plan.epilogue,
    )
    if any(item is None for item in required):
        raise ValueError(
            f"incomplete GEMM lowering plan for kernel {plan.name}"
        )


def _emit_gemm_kernel(plan) -> list[str]:
    """Emit the first semantically complete naive CUDA GEMM kernel.

    v0.15 deliberately uses direct row-major global-memory loads for every
    multiply-accumulate. It prioritizes a transparent mapping from the lowering
    plan to CUDA source over performance. Shared memory, register tiles and
    vectorized loads are later optimization steps.
    """

    _require_gemm_plan(plan)

    problem = plan.problem
    tile = plan.tile
    schedule = plan.schedule
    mapping = plan.block_mapping
    flow = plan.control_flow
    epilogue = plan.epilogue
    scalar = _cuda_scalar_type(problem.dtype)

    if epilogue.bias:
        signature = (
            f'extern "C" __global__ void {plan.name}(\n'
            f'    const {scalar}* __restrict__ A,\n'
            f'    const {scalar}* __restrict__ B,\n'
            f'    const {scalar}* __restrict__ bias,\n'
            f'    {scalar}* __restrict__ C) {{'
        )
    else:
        signature = (
            f'extern "C" __global__ void {plan.name}(\n'
            f'    const {scalar}* __restrict__ A,\n'
            f'    const {scalar}* __restrict__ B,\n'
            f'    {scalar}* __restrict__ C) {{'
        )

    lines = [
        signature,
        f"  constexpr int M = {problem.m};",
        f"  constexpr int N = {problem.n};",
        f"  constexpr int K = {problem.k};",
        f"  constexpr int BM = {tile.block_m};",
        f"  constexpr int BN = {tile.block_n};",
        f"  constexpr int BK = {tile.block_k};",
        f"  constexpr int K_TILES = {schedule.k_tiles};",
        "",
        "  for (int output = static_cast<int>(threadIdx.x);",
        "       output < BM * BN;",
        "       output += static_cast<int>(blockDim.x)) {",
        "    const int local_m = output / BN;",
        "    const int local_n = output % BN;",
        "    const int row = static_cast<int>(blockIdx.y) * BM + local_m;",
        "    const int col = static_cast<int>(blockIdx.x) * BN + local_n;",
    ]

    indent = "    "
    if flow.output_guard:
        lines.append("    if (row < M && col < N) {")
        indent = "      "

    lines.extend(
        [
            f"{indent}{scalar} acc = {scalar}(0);",
            f"{indent}for (int kt = 0; kt < K_TILES; ++kt) {{",
            f"{indent}  for (int kk = 0; kk < BK; ++kk) {{",
            f"{indent}    const int k = kt * BK + kk;",
        ]
    )

    if flow.k_tail_guard:
        lines.extend(
            [
                f"{indent}    if (k < K) {{",
                f"{indent}      acc += A[row * K + k] * B[k * N + col];",
                f"{indent}    }}",
            ]
        )
    else:
        lines.append(
            f"{indent}    acc += A[row * K + k] * B[k * N + col];"
        )

    lines.extend(
        [
            f"{indent}  }}",
            f"{indent}}}",
        ]
    )

    if epilogue.bias:
        lines.append(f"{indent}{scalar} value = acc + bias[col];")
    else:
        lines.append(f"{indent}{scalar} value = acc;")

    if epilogue.activation == "relu":
        lines.append(
            f"{indent}value = value > {scalar}(0) ? value : {scalar}(0);"
        )
    elif epilogue.activation is not None:
        raise NotImplementedError(
            f"unsupported GEMM epilogue activation: {epilogue.activation}"
        )

    lines.append(f"{indent}C[row * N + col] = value;")

    if flow.output_guard:
        lines.append("    }")

    lines.extend(["  }", "}"])
    return lines


def _emit_plan_header(plan) -> list[str]:
    lines = [
        f"// {plan.name}",
        f"// source_op: {plan.source_op}",
        f"// strategy: {plan.strategy}",
    ]

    if plan.problem is not None:
        lines.append(
            "// gemm_problem: "
            f"M={plan.problem.m}, "
            f"N={plan.problem.n}, "
            f"K={plan.problem.k}, "
            f"dtype={plan.problem.dtype}"
        )

    if plan.tile is not None:
        lines.append(
            "// gemm_tile: "
            f"BM={plan.tile.block_m}, "
            f"BN={plan.tile.block_n}, "
            f"BK={plan.tile.block_k}"
        )

    if plan.schedule is not None:
        lines.append(
            "// gemm_schedule: "
            f"grid_m={plan.schedule.grid_m}, "
            f"grid_n={plan.schedule.grid_n}, "
            f"k_tiles={plan.schedule.k_tiles}"
        )

    if plan.block_mapping is not None:
        lines.append(
            "// gemm_block_mapping: "
            f"threads={plan.block_mapping.threads}, "
            f"warps={plan.block_mapping.warps}, "
            f"outputs_per_thread={plan.block_mapping.outputs_per_thread}"
        )
        lines.append(
            "// launch_shape: "
            f"grid=({plan.schedule.grid_n}, {plan.schedule.grid_m}, 1), "
            f"block=({plan.block_mapping.threads}, 1, 1)"
        )

    if plan.thread_mapping is not None:
        lines.append(
            "// gemm_thread_mapping: "
            f"traversal={plan.thread_mapping.traversal}, "
            f"output_order={plan.thread_mapping.output_order}, "
            f"thread_axis={plan.thread_mapping.thread_axis}, "
            f"block_m_axis={plan.thread_mapping.block_m_axis}, "
            f"block_n_axis={plan.thread_mapping.block_n_axis}"
        )

    if plan.control_flow is not None:
        lines.append(
            "// gemm_control_flow: "
            f"output_traversal={plan.control_flow.output_traversal}, "
            f"output_guard={plan.control_flow.output_guard}, "
            f"k_traversal={plan.control_flow.k_traversal}, "
            f"k_tail_guard={plan.control_flow.k_tail_guard}"
        )

    if plan.epilogue is not None:
        lines.append(
            "// gemm_epilogue: "
            f"bias={plan.epilogue.bias}, "
            f"activation={plan.epilogue.activation}"
        )

    return lines


def codegen(lowered) -> CUDAExecutableImage:
    """Materialize CUDA kernel plans into source strings.

    GEMM-like plans now produce syntactically plausible CUDA C++ kernels with
    naive row-major global-memory loads, scalar accumulation, output stores and
    the planned bias/ReLU epilogue. Other op kinds remain explicit TODOs until
    their codegen paths are implemented.
    """

    kernel_names = [plan.name for plan in lowered.kernels]
    chunks: list[str] = []

    for plan in lowered.kernels:
        if chunks:
            chunks.append("")

        chunks.extend(_emit_plan_header(plan))

        if plan.problem is not None:
            chunks.append("")
            chunks.extend(_emit_gemm_kernel(plan))
        else:
            chunks.append(f"// TODO executable CUDA codegen for {plan.name}")

    return CUDAExecutableImage(
        kernels=kernel_names,
        code="\n".join(chunks),
        plans=list(lowered.kernels),
    )
