from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from .spec import KernelSpec, LaunchSpec


def generated_naive_gemm_bias_relu() -> KernelSpec:
    source = files("aicf_labs.backend_cuda.kernels").joinpath(
        "sources/generated_naive_gemm_bias_relu.cu"
    )
    return KernelSpec(
        name="generated_naive_gemm_bias_relu_v020",
        workload_kind="gemm_bias_relu",
        entry="kernel_0_fused_gemm_bias_relu",
        source_path=Path(str(source)),
        launch=LaunchSpec(grid=(1, 1, 1), block=(256, 1, 1)),
        architecture="sm_86",
        tags=("baseline", "v0.20", "generated_naive", "global_memory"),
    )
