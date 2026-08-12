import numpy as np
import pytest

from aicf import compile, nn
from aicf.frontend.tensor import Parameter, TensorSpec


def test_parameter_owns_concrete_zero_initialized_host_storage():
    parameter = Parameter(TensorSpec((2, 3), "float32", "p"))

    assert parameter.data is not None
    assert parameter.data.shape == (2, 3)
    assert parameter.data.dtype == np.float32
    assert np.all(parameter.data == 0)

    replacement = np.arange(6, dtype=np.float32).reshape(2, 3)
    parameter.set_data(replacement)

    assert np.array_equal(parameter.data, replacement)
    assert parameter.data is not replacement


def test_runtime_binding_connects_ir_refs_to_input_parameters_and_output():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    weight = np.arange(32, dtype=np.float32).reshape(4, 8)
    bias = np.arange(8, dtype=np.float32)
    model[0].weight.set_data(weight)
    model[0].bias.set_data(bias)

    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    bindings = executable.bind(x)

    assert bindings.buffer("%0").slot.role == "input"
    assert bindings.buffer("%0").array is x

    assert bindings.buffer("%1").slot.name == "0.weight"
    assert bindings.buffer("%1").array is model[0].weight.data

    assert bindings.buffer("%2").slot.name == "0.bias"
    assert bindings.buffer("%2").array is model[0].bias.data

    # Fusion preserves the original ReLU result value %5 as the module output.
    output = bindings.buffer("%5")
    assert output.slot.role == "output"
    assert output.array.shape == (2, 8)
    assert output.array.dtype == np.float32

    plan = executable.image.plans[0]
    args = bindings.kernel_arguments(plan)

    assert plan.inputs == ("%0", "%1", "%2")
    assert plan.outputs == ("%5",)
    assert args[0] is x
    assert args[1] is model[0].weight.data
    assert args[2] is model[0].bias.data
    assert args[3] is output.array


def test_runtime_allocates_intermediate_buffers_for_unfused_path():
    class MultiUseBiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 8)
            self.relu_a = nn.ReLU()
            self.relu_b = nn.ReLU()

        def forward(self, x):
            y = self.linear(x)
            out = self.relu_a(y)
            self.relu_b(y)
            return out

    executable = compile(
        MultiUseBiasModel(),
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )

    bindings = executable.bind(
        np.zeros((2, 4), dtype=np.float32)
    )

    # Unfused IR contains %3 gemm, %4 bias_add, %5 relu output and %6 dead relu.
    assert bindings.buffer("%3").slot.role == "temporary"
    assert bindings.buffer("%4").slot.role == "temporary"
    assert bindings.buffer("%5").slot.role == "output"
    assert bindings.buffer("%6").slot.role == "temporary"

    for plan in executable.image.plans:
        args = bindings.kernel_arguments(plan)
        assert len(args) == len(plan.inputs) + len(plan.outputs)


def test_runtime_rejects_input_shape_or_dtype_mismatch():
    model = nn.Sequential(nn.Linear(4, 8, bias=False))
    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        executable.bind(np.zeros((3, 4), dtype=np.float32))

    with pytest.raises(TypeError, match="dtype mismatch"):
        executable.bind(np.zeros((2, 4), dtype=np.float64))


def test_runtime_run_reports_host_bound_launch_contract():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )
    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )

    result = executable.run(
        np.zeros((2, 4), dtype=np.float32)
    )

    assert result["status"] == "host_bound"
    assert result["launched"] is False
    assert result["kernels"] == ["kernel_0_fused_gemm_bias_relu"]
    assert result["launches"] == [
        {
            "kernel": "kernel_0_fused_gemm_bias_relu",
            "argument_refs": ["%0", "%1", "%2", "%5"],
            "grid": (1, 1, 1),
            "block": (256, 1, 1),
        }
    ]