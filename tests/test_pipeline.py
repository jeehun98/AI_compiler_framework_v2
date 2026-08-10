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


def test_graph_use_def_links():
    from aicf.graph.builder import capture_graph

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    with capture_graph() as builder:
        x = builder.input(TensorSpec((2, 4), name="x"))
        builder.bind_parameters(model.named_parameters())
        y = model(x)
        builder.output(y)
        graph = builder.graph

    gemm, bias_add, relu = graph.nodes
    x_value = graph.inputs[0]
    weight, bias = graph.parameters
    gemm_out = gemm.outputs[0]
    bias_out = bias_add.outputs[0]
    relu_out = relu.outputs[0]

    assert x_value.producer is None
    assert x_value.users == [gemm]
    assert weight.producer is None
    assert weight.users == [gemm]
    assert bias.producer is None
    assert bias.users == [bias_add]

    assert gemm_out.producer is gemm
    assert gemm_out.users == [bias_add]
    assert bias_out.producer is bias_add
    assert bias_out.users == [relu]
    assert relu_out.producer is relu
    assert relu_out.users == []


def test_ir_use_def_analysis():
    from aicf.graph.builder import capture_graph
    from aicf.ir.from_graph import graph_to_ir
    from aicf.compiler.analysis.use_def import build_use_def

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    with capture_graph() as builder:
        x = builder.input(TensorSpec((2, 4), name="x"))
        builder.bind_parameters(model.named_parameters())
        y = model(x)
        builder.output(y)
        graph = builder.graph

    module = graph_to_ir(graph)
    analysis = build_use_def(module)

    gemm, bias_add, relu = module.ops
    x_value = module.inputs[0]
    weight, bias = module.parameters
    gemm_out = gemm.results[0]
    bias_out = bias_add.results[0]
    relu_out = relu.results[0]

    assert analysis.producer(x_value) is None
    assert analysis.users(x_value) == (gemm,)
    assert analysis.users(weight) == (gemm,)
    assert analysis.users(bias) == (bias_add,)

    assert analysis.producer(gemm_out) is gemm
    assert analysis.users(gemm_out) == (bias_add,)
    assert analysis.use_count(gemm_out) == 1

    assert analysis.producer(bias_out) is bias_add
    assert analysis.users(bias_out) == (relu,)
    assert analysis.use_count(bias_out) == 1

    assert analysis.producer(relu_out) is relu
    assert analysis.users(relu_out) == ()
    assert analysis.use_count(relu_out) == 0