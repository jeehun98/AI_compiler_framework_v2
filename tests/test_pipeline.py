from aicf import compile
from aicf.frontend.tensor import TensorSpec
from aicf.frontend.ops import gemm, bias_add, relu


def model(x, w, b):
    return relu(bias_add(gemm(x, w), b))


def test_mock_compile_pipeline():
    exe = compile(model, [
        TensorSpec((2, 4), name="x"),
        TensorSpec((4, 8), name="w"),
        TensorSpec((8,), name="b"),
    ])
    result = exe.run()
    assert result["status"] == "mock"
    assert len(result["kernels"]) == 3
