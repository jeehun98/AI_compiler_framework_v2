from aicf_labs.backend_cuda import default_registry
from aicf_labs.frontend_lab.adapters import fused_gemm_bias_relu_workload


def test_generated_naive_is_backend_candidate_not_core_pipeline():
    workload = fused_gemm_bias_relu_workload(m=32, n=128, k=64)
    registry = default_registry()
    candidates = registry.candidates(workload)

    assert [candidate.spec.name for candidate in candidates] == ["generated_naive"]
    plan = candidates[0].plan(workload)
    assert (plan.m, plan.n, plan.k) == (32, 128, 64)
    assert (plan.block_m, plan.block_n, plan.block_k, plan.threads) == (128, 128, 32, 256)
