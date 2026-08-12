from aicf_labs.experiments import v020_baseline_record


def test_v020_baseline_is_frozen_as_experiment_record():
    record = v020_baseline_record()

    assert record.experiment_id == "baseline-v0.20-gemm-bias-relu"
    assert record.workload.kind == "gemm_bias_relu"
    assert record.implementation.name == "generated_naive_v020"
    assert record.validation is not None and record.validation.passed is True
    assert record.runtime.values["status"] == "cuda_executed"
