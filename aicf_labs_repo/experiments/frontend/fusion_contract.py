from pprint import pprint

from aicf_labs.frontend_lab.adapters import (
    fused_gemm_bias_relu_workload,
    gemm_bias_relu_fusion_record,
)


if __name__ == "__main__":
    print("[transformation]")
    pprint(gemm_bias_relu_fusion_record())
    print("\n[workload]")
    pprint(fused_gemm_bias_relu_workload(m=32, n=128, k=64))
