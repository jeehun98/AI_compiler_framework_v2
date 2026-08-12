from pprint import pprint

from aicf_labs.backend_cuda.implementations import GeneratedNaive
from aicf_labs.frontend_lab.adapters import fused_gemm_bias_relu_workload


if __name__ == "__main__":
    workload = fused_gemm_bias_relu_workload(m=32, n=128, k=64)
    implementation = GeneratedNaive()
    print("[implementation]")
    pprint(implementation.spec)
    print("\n[plan]")
    pprint(implementation.plan(workload))
