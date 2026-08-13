from aicf_labs import Tensor, capture, nn
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
