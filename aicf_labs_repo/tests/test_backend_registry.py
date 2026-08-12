from aicf_labs.backend_cuda import default_registry
from aicf_labs.frontend_lab.adapters import fused_gemm_bias_relu_workload


def test_generated_naive_is_frozen_backend_candidate():
    workload = fused_gemm_bias_relu_workload(m=32, n=128, k=64)
    registry = default_registry()
    candidates = registry.candidates(workload)

    assert [candidate.spec.name for candidate in candidates] == ["generated_naive_v020"]
    kernel = candidates[0].plan(workload)
    assert kernel.entry == "kernel_0_fused_gemm_bias_relu"
    assert kernel.launch.block == (256, 1, 1)
