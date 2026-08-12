from __future__ import annotations

from ..backend_cuda.implementations import GeneratedNaive
from ..contracts import (
    EnvironmentSpec,
    ExperimentRecord,
    RuntimeObservation,
    ValidationResult,
)
from ..frontend_lab.adapters import fused_gemm_bias_relu_workload, gemm_bias_relu_fusion_record


def v020_baseline_record() -> ExperimentRecord:
    workload = fused_gemm_bias_relu_workload(m=32, n=128, k=64)
    implementation = GeneratedNaive().spec

    return ExperimentRecord(
        experiment_id="baseline-v0.20-gemm-bias-relu",
        workload=workload,
        implementation=implementation,
        transformation=gemm_bias_relu_fusion_record(selected=True),
        environment=EnvironmentSpec(
            gpu_name="NVIDIA GeForce RTX 3080 Ti Laptop GPU",
            gpu_arch="compute_86",
            cuda_version="13.3",
            compiler="nvrtc",
            compiler_version="13.3",
        ),
        runtime=RuntimeObservation(
            values={
                "grid": (1, 1, 1),
                "block": (256, 1, 1),
                "status": "cuda_executed",
            }
        ),
        validation=ValidationResult(
            passed=True,
            max_abs_error=1.6391277313232422e-07,
            mean_abs_error=4.8260062612826005e-09,
        ),
    )
