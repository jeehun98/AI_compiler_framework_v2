from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec


def test_sequential_model_registration():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    assert len(model) == 2
    assert [name for name, _ in model.named_parameters()] == ["0.weight", "0.bias"]


def test_mock_compile_pipeline():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    exe = compile(model, [TensorSpec((2, 4), name="x")])
    result = exe.run()
    assert result["status"] == "mock"
    assert result["kernels"] == [
        "kernel_0_gemm",
        "kernel_1_bias_add",
        "kernel_2_relu",
    ]
