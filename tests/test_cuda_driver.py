import importlib

import numpy as np
import pytest

from aicf import nn
from aicf.frontend.tensor import TensorSpec
from aicf.backend.cuda.codegen.generator import CUDAExecutableImage
from aicf.backend.cuda.compiler import (
    CUDACompiledImage,
    NVRTCCompiler,
)
from aicf.backend.cuda.driver import (
    CUDADriver,
    CUDALoadedImage,
)


class FakeCUDADriverLibrary:
    def __init__(self, *, device_count=1):
        self.calls = []
        self._device_count = device_count
        self._current = 0x1111

    def initialize(self):
        self.calls.append(("initialize",))

    def driver_version(self):
        self.calls.append(("driver_version",))
        return 13030

    def device_count(self):
        self.calls.append(("device_count",))
        return self._device_count

    def device(self, ordinal):
        self.calls.append(("device", ordinal))
        return 7 + ordinal

    def device_name(self, device):
        self.calls.append(("device_name", device))
        return "Fake RTX"

    def current_context(self):
        self.calls.append(("current_context", self._current))
        return self._current

    def retain_primary_context(self, device):
        self.calls.append(("retain_primary_context", device))
        return 0x2222

    def release_primary_context(self, device):
        self.calls.append(("release_primary_context", device))

    def set_current_context(self, context):
        self.calls.append(("set_current_context", context))
        self._current = context

    def load_module(self, ptx):
        self.calls.append(("load_module", ptx))
        return 0x3333

    def get_function(self, module, name):
        self.calls.append(("get_function", module, name))
        return 0x4000 + len(
            [call for call in self.calls if call[0] == "get_function"]
        )

    def unload_module(self, module):
        self.calls.append(("unload_module", module))


def _compiled_image(*kernels):
    source = CUDAExecutableImage(
        kernels=list(kernels),
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


def test_cuda_driver_loads_ptx_and_resolves_kernel_handles():
    library = FakeCUDADriverLibrary()
    driver = CUDADriver(_library=library)

    loaded = driver.load(
        _compiled_image("kernel_a", "kernel_b"),
        device_ordinal=0,
    )

    assert loaded.device_ordinal == 0
    assert loaded.device_name == "Fake RTX"
    assert loaded.driver_version == 13030
    assert loaded.context_handle == 0x2222
    assert loaded.module_handle == 0x3333
    assert loaded.kernels == ["kernel_a", "kernel_b"]
    assert loaded.function("kernel_a").handle != 0
    assert loaded.function("kernel_b").handle != 0

    # Loading temporarily switches to the retained primary context, then
    # restores the context that was current before the module load.
    assert library._current == 0x1111
    assert ("set_current_context", 0x2222) in library.calls
    assert ("set_current_context", 0x1111) in library.calls

    loaded.close()
    assert loaded.closed is True
    assert ("unload_module", 0x3333) in library.calls
    assert ("release_primary_context", 7) in library.calls

    with pytest.raises(RuntimeError, match="closed"):
        loaded.function("kernel_a")


def test_cuda_driver_rejects_invalid_device_ordinal_before_context_retain():
    library = FakeCUDADriverLibrary(device_count=1)
    driver = CUDADriver(_library=library)

    with pytest.raises(ValueError, match="out of range"):
        driver.load(_compiled_image("kernel"), device_ordinal=1)

    assert not any(
        call[0] == "retain_primary_context"
        for call in library.calls
    )


def test_compile_pipeline_can_attach_optional_loaded_image(monkeypatch):
    compile_module = importlib.import_module("aicf.compile")

    def fake_compile_with_nvrtc(
        image,
        *,
        arch=None,
        library_path=None,
    ):
        return CUDACompiledImage(
            source_image=image,
            ptx=b"fake ptx",
            compiler="nvrtc",
            compiler_version=(13, 3),
            options=(),
        )

    class FakeLoadedImage:
        def __init__(self, compiled):
            self.compiled_image = compiled
            self.device_ordinal = 0
            self.device_name = "Fake RTX"
            self.driver_version = 13030
            self.context_handle = 0x22
            self.module_handle = 0x33
            self.kernels = list(compiled.kernels)
            self.closed = False

        def close(self):
            self.closed = True

    def fake_load_with_cuda_driver(
        compiled,
        *,
        device_ordinal=0,
        library_path=None,
    ):
        assert device_ordinal == 0
        assert library_path == "fake-driver"
        return FakeLoadedImage(compiled)

    monkeypatch.setattr(
        compile_module,
        "compile_with_nvrtc",
        fake_compile_with_nvrtc,
    )
    monkeypatch.setattr(
        compile_module,
        "load_with_cuda_driver",
        fake_load_with_cuda_driver,
    )

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    executable = compile_module.compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
        cuda_compile=True,
        cuda_load=True,
        cuda_driver_library="fake-driver",
    )

    result = executable.run(
        np.zeros((2, 4), dtype=np.float32)
    )

    assert result["status"] == "cuda_loaded"
    assert result["compiled"] is True
    assert result["loaded"] is True
    assert result["launched"] is False
    assert result["cuda_device"] == 0
    assert result["cuda_device_name"] == "Fake RTX"
    assert result["resolved_kernels"] == [
        "kernel_0_fused_gemm_bias_relu"
    ]

    loaded = executable.loaded_image
    executable.close()
    assert loaded.closed is True


def test_cuda_load_requires_cuda_compile():
    from aicf import compile

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    with pytest.raises(ValueError, match="requires cuda_compile"):
        compile(
            model,
            [TensorSpec((2, 4), "float32", "x")],
            diagnostics=False,
            cuda_load=True,
        )


def test_real_driver_can_load_generated_ptx_when_available():
    if not NVRTCCompiler.is_available() or not CUDADriver.is_available():
        pytest.skip("NVRTC and CUDA Driver with a device are required")

    from aicf import compile

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
    )

    executable = compile(
        model,
        [TensorSpec((2, 4), "float32", "x")],
        diagnostics=False,
        cuda_compile=True,
        cuda_arch="86",
        cuda_load=True,
        cuda_device=0,
    )

    try:
        loaded = executable.loaded_image
        assert isinstance(loaded, CUDALoadedImage)
        assert loaded.module_handle != 0
        assert loaded.function(
            "kernel_0_fused_gemm_bias_relu"
        ).handle != 0
    finally:
        executable.close()