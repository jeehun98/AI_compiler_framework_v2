from aicf_labs.frontend_lab.adapters import (
    fused_gemm_bias_relu_workload,
    gemm_bias_relu_fusion_record,
)


def test_v020_fused_adapter():
    workload = fused_gemm_bias_relu_workload(m=32, n=128, k=64)
    record = gemm_bias_relu_fusion_record()

    assert workload.kind == "gemm_bias_relu"
    assert workload.inputs[0].shape == (32, 64)
    assert workload.inputs[1].shape == (64, 128)
    assert workload.outputs[0].shape == (32, 128)
    assert record.before == ("gemm", "bias_add", "relu")
    assert record.after == ("fused_gemm_bias_relu",)
