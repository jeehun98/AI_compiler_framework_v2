import ctypes

import numpy as np
import pytest

from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.backend.cuda.codegen.generator import CUDAExecutableImage
from aicf.backend.cuda.compiler import CUDACompiledImage, NVRTCCompiler
from aicf.backend.cuda.driver import CUDADriver


class FakeMemoryDriverLibrary:
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
        self.calls.append(("mem_alloc", int(nbytes), pointer))
        return pointer

    def mem_free(self, pointer):
        self.calls.append(("mem_free", pointer))
        del self.memory[pointer]

    def memcpy_htod(self, dst_device, src_host, nbytes):
        payload = ctypes.string_at(int(src_host), int(nbytes))
        self.memory[dst_device][:] = payload
        self.calls.append(("memcpy_htod", dst_device, int(nbytes)))

    def memcpy_dtoh(self, dst_host, src_device, nbytes):
        payload = bytes(self.memory[src_device][: int(nbytes)])
        ctypes.memmove(int(dst_host), payload, int(nbytes))
        self.calls.append(("memcpy_dtoh", src_device, int(nbytes)))


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
    library = FakeMemoryDriverLibrary()
    loaded = CUDADriver(_library=library).load(
        _compiled_image(executable.image.kernels[0])
    )
    executable.loaded_image = loaded
    return library, loaded


def test_device_bindings_allocate_upload_roundtrip_and_free():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )
    model[0].weight.set_data(
        np.arange(32, dtype=np.float32).reshape(4, 8)
    )
    model[0].bias.set_data(np.arange(8, dtype=np.float32))

    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )
    library, loaded = _attach_fake_loaded_image(executable)

    x = np.arange(8, dtype=np.float32).reshape(2, 4)

    with executable.bind_device(x) as device:
        assert list(device.buffers) == ["%0", "%1", "%2", "%5"]

        assert device.buffer("%0").uploaded is True
        assert device.buffer("%1").uploaded is True
        assert device.buffer("%2").uploaded is True
        assert device.buffer("%5").uploaded is False

        plan = executable.image.plans[0]
        pointers = device.kernel_arguments(plan)
        assert pointers == (
            device.buffer("%0").pointer,
            device.buffer("%1").pointer,
            device.buffer("%2").pointer,
            device.buffer("%5").pointer,
        )

        roundtrip = np.empty_like(x)
        device.copy_to_host("%0", roundtrip)
        assert np.array_equal(roundtrip, x)

        assert len(library.memory) == 4

    assert len(library.memory) == 0
    assert len([c for c in library.calls if c[0] == "mem_free"]) == 4

    loaded.close()


def test_device_binding_requires_loaded_cuda_image():
    executable = compile(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
    )

    with pytest.raises(RuntimeError, match="requires a loaded CUDA image"):
        executable.bind_device(np.zeros((2, 4), dtype=np.float32))


def test_real_cuda_device_memory_roundtrip_when_available():
    if not NVRTCCompiler.is_available() or not CUDADriver.is_available():
        pytest.skip("NVRTC and CUDA Driver with a device are required")

    executable = compile(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
        cuda_compile=True,
        cuda_arch="86",
        cuda_load=True,
    )

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    try:
        with executable.bind_device(x) as device:
            roundtrip = np.empty_like(x)
            device.copy_to_host("%0", roundtrip)
            assert np.array_equal(roundtrip, x)
            assert device.buffer("%0").pointer != 0
    finally:
        executable.close()