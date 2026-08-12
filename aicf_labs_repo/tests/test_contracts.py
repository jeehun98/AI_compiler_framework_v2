from aicf_labs.contracts import NumericalContract, TensorSpec, WorkloadSpec


def test_workload_contract_is_implementation_independent():
    workload = WorkloadSpec(
        kind="gemm",
        inputs=(
            TensorSpec((2, 4), name="x"),
            TensorSpec((4, 8), name="weight"),
        ),
        outputs=(TensorSpec((2, 8), name="output"),),
        numerical=NumericalContract(rtol=1e-4, atol=1e-5),
    )
    assert workload.kind == "gemm"
    assert workload.inputs[0].shape == (2, 4)
    assert "cuda" not in workload.to_dict() if hasattr(workload, "to_dict") else True
