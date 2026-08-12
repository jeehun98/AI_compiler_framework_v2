import ctypes

import numpy as np
import pytest

from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.backend.cuda.codegen.generator import CUDAExecutableImage
from aicf.backend.cuda.compiler import CUDACompiledImage, NVRTCCompiler
from aicf.backend.cuda.driver import CUDADriver


class FakeExecutionDriverLibrary:
    def __init__(self):
        self.calls = []
        self._current = 0x1111
        self._next_pointer = 0x100000
        self.memory = {}

    def initialize(self):
        self.calls.append(("initialize",))

    def driver_version(self):
        return 13030

    def device_count(self):
        return 1

    def device(self, ordinal):
        return 7

    def device_name(self, device):
        return "Fake RTX"

    def current_context(self):
        return self._current

    def retain_primary_context(self, device):
        return 0x2222

    def release_primary_context(self, device):
        self.calls.append(("release_primary_context", device))

    def set_current_context(self, context):
        self.calls.append(("set_current_context", context))
        self._current = context

    def load_module(self, ptx):
        return 0x3333

    def get_function(self, module, name):
        return 0x4444

    def unload_module(self, module):
        self.calls.append(("unload_module", module))

    def mem_alloc(self, nbytes):
        pointer = self._next_pointer
        self._next_pointer += max(int(nbytes), 0x1000)
        self.memory[pointer] = bytearray(int(nbytes))
        return pointer

    def mem_free(self, pointer):
        del self.memory[pointer]

    def memcpy_htod(self, dst_device, src_host, nbytes):
        self.memory[dst_device][:] = ctypes.string_at(
            int(src_host), int(nbytes)
        )

    def memcpy_dtoh(self, dst_host, src_device, nbytes):
        ctypes.memmove(
            int(dst_host),
            bytes(self.memory[src_device][: int(nbytes)]),
            int(nbytes),
        )

    def launch_kernel(
        self,
        function,
        *,
        grid,
        block,
        kernel_params,
        shared_mem_bytes=0,
        stream=0,
    ):
        kernel_params = tuple(kernel_params)
        self.calls.append(
            (
                "launch_kernel",
                function,
                tuple(grid),
                tuple(block),
                kernel_params,
            )
        )

        # Fake a completed kernel by writing a deterministic sequence into the
        # output allocation (the final pointer argument).
        output_pointer = kernel_params[-1]
        output_memory = self.memory[output_pointer]
        values = np.arange(
            len(output_memory) // np.dtype(np.float32).itemsize,
            dtype=np.float32,
        )
        output_memory[:] = values.tobytes()

    def synchronize(self):
        self.calls.append(("synchronize",))


def _compiled_image(kernel_name):
    source = CUDAExecutableImage(
        kernels=[kernel_name],
        code="",
        plans=[],
    )
    return CUDACompiledImage(
        source_image=source,
        ptx=b".version 8.8\n.target sm_86\n",
        compiler="nvrtc",
        compiler_version=(13, 3),
        options=("--gpu-architecture=compute_86",),
    )


def _attach_fake_loaded_image(executable):
    library = FakeExecutionDriverLibrary()
    loaded = CUDADriver(_library=library).load(
        _compiled_image(executable.image.kernels[0])
    )
    executable.loaded_image = loaded
    return library, loaded


def test_run_cuda_launches_synchronizes_and_copies_output():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )
    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )
    library, loaded = _attach_fake_loaded_image(executable)

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    execution = executable.run_cuda(x)

    assert len(execution.launches) == 1
    launch = execution.launches[0]
    assert launch.kernel == "kernel_0_fused_gemm_bias_relu"
    assert launch.function_handle == 0x4444
    assert launch.argument_refs == ("%0", "%1", "%2", "%5")
    assert launch.grid == (1, 1, 1)
    assert launch.block == (256, 1, 1)

    expected = np.arange(16, dtype=np.float32).reshape(2, 8)
    assert np.array_equal(execution.output, expected)

    launch_calls = [c for c in library.calls if c[0] == "launch_kernel"]
    assert len(launch_calls) == 1
    assert ("synchronize",) in library.calls
    assert len(library.memory) == 0

    summary = executable.run(x)
    assert summary["status"] == "cuda_executed"
    assert summary["launched"] is True
    assert summary["executed_kernels"] == [
        "kernel_0_fused_gemm_bias_relu"
    ]

    loaded.close()


def test_run_cuda_requires_loaded_image():
    executable = compile(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )

    with pytest.raises(RuntimeError, match="requires a loaded CUDA image"):
        executable.run_cuda(np.zeros((2, 4), dtype=np.float32))


def test_real_cuda_fused_gemm_matches_numpy_when_available():
    if not NVRTCCompiler.is_available() or not CUDADriver.is_available():
        pytest.skip("NVRTC and a CUDA Driver device are required")

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )
    weight = np.linspace(-0.2, 0.2, 32, dtype=np.float32).reshape(4, 8)
    bias = np.linspace(-0.1, 0.1, 8, dtype=np.float32)
    model[0].weight.set_data(weight)
    model[0].bias.set_data(bias)

    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
        cuda_compile=True,
        cuda_load=True,
    )

    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    reference = np.maximum(x @ weight + bias, np.float32(0))

    try:
        execution = executable.run_cuda(x)
        np.testing.assert_allclose(
            execution.output,
            reference,
            rtol=1e-4,
            atol=1e-5,
        )
        assert len(execution.launches) == 1
    finally:
        executable.close()