import pytest

from aicf_labs import Tensor, capture, nn, ops
from aicf_labs.frontend_lab.analysis.graph import analyze_graph_marks
from aicf_labs.frontend_lab.marking import OpMask


def test_sequential_linear_relu_builds_canonical_operator_graph():
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
    x = Tensor((32, 64), name="x")

    graph = capture(model, x)

    assert graph.operator_names() == ("gemm", "bias_add", "relu")
    assert [value.shape for value in graph.inputs] == [(32, 64)]
    assert [value.shape for value in graph.parameters] == [(64, 128), (128,)]
    assert graph.outputs[0].shape == (32, 128)


def test_native_graph_connects_to_existing_frontend_marking():
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
    graph = capture(model, Tensor((32, 64)))

    result = analyze_graph_marks(graph)

    assert result.operators == ("gemm", "bias_add", "relu")
    assert result.preserves(OpMask.SIDE_EFFECT_FREE)
    assert not result.preserves(OpMask.ELEMENTWISE)


def test_add_supports_broadcasting_and_tensor_syntax():
    lhs = Tensor((32, 128), name="lhs")
    rhs = Tensor((128,), name="rhs")

    via_function = ops.add(lhs, rhs)
    via_tensor = lhs + rhs

    assert via_function.shape == (32, 128)
    assert via_tensor.shape == (32, 128)


def test_add_rejects_incompatible_shape_or_dtype():
    with pytest.raises(ValueError, match="add shape mismatch"):
        ops.add(Tensor((32, 128)), Tensor((64,)))

    with pytest.raises(ValueError, match="add dtype mismatch"):
        ops.add(Tensor((32, 128), dtype="float32"), Tensor((32, 128), dtype="float16"))


def test_linear_relu_add_builds_residual_style_graph():
    class LinearReluAdd(nn.Module):
        def __init__(self):
            self.linear = nn.Linear(64, 128)
            self.relu = nn.ReLU()

        def forward(self, x, residual):
            return self.relu(self.linear(x)) + residual

    model = LinearReluAdd()
    x = Tensor((32, 64), name="x")
    residual = Tensor((32, 128), name="residual")

    graph = capture(model, x, residual)

    assert graph.operator_names() == ("gemm", "bias_add", "relu", "add")
    assert [value.shape for value in graph.inputs] == [(32, 64), (32, 128)]
    assert [value.shape for value in graph.parameters] == [(64, 128), (128,)]
    assert graph.outputs[0].shape == (32, 128)

    result = analyze_graph_marks(graph)
    assert result.operators == ("gemm", "bias_add", "relu", "add")
    assert result.preserves(OpMask.SIDE_EFFECT_FREE)
