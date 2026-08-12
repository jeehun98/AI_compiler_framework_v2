import importlib

import numpy as np
import pytest

from aicf import nn
from aicf.frontend.tensor import TensorSpec
from aicf.backend.cuda.codegen.generator import CUDAExecutableImage
from aicf.backend.cuda.compiler import (
    CUDACompileOptions,
    CUDACompiledImage,
    NVRTCCompileError,
    NVRTCCompiler,
)


class FakeNVRTCLibrary:
    def __init__(self):
        self.calls = []

    def version(self):
        return (13, 3)

    def compile(self, source, *, name, options):
        self.calls.append(
            {
                "source": source,
                "name": name,
                "options": tuple(options),
            }
        )
        return b".version 8.8\n.entry kernel() {}\n", "fake compile log"


def test_cuda_compile_options_materialize_nvrtc_arguments():
    options = CUDACompileOptions(
        arch="86",
        extra=("--use_fast_math",),
    )

    assert options.nvrtc_args() == (
        "--std=c++17",
        "--gpu-architecture=compute_86",
        "--use_fast_math",
    )

    assert CUDACompileOptions(arch="sm_86").nvrtc_args() == (
        "--std=c++17",
        "--gpu-architecture=sm_86",
    )

    with pytest.raises(ValueError, match="CUDA arch"):
        CUDACompileOptions(arch="ampere").nvrtc_args()


def test_nvrtc_compiler_wraps_ptx_as_compiled_image():
    library = FakeNVRTCLibrary()
    compiler = NVRTCCompiler(_library=library)
    source_image = CUDAExecutableImage(
        kernels=["kernel"],
        code='extern "C" __global__ void kernel() {}',
        plans=[],
    )

    compiled = compiler.compile(
        source_image,
        options=CUDACompileOptions(arch="compute_86"),
    )

    assert compiled.compiler == "nvrtc"
    assert compiled.compiler_version == (13, 3)
    assert compiled.kernels == ["kernel"]
    assert compiled.ptx.startswith(b".version")
    assert compiled.ptx_nbytes == len(compiled.ptx)
    assert compiled.log == "fake compile log"
    assert library.calls[0]["options"] == (
        "--std=c++17",
        "--gpu-architecture=compute_86",
    )


def test_nvrtc_compiler_rejects_unresolved_codegen():
    compiler = NVRTCCompiler(_library=FakeNVRTCLibrary())
    image = CUDAExecutableImage(
        kernels=["kernel_0_relu"],
        code="// TODO executable CUDA codegen for kernel_0_relu",
        plans=[],
        unresolved_kernels=["kernel_0_relu"],
    )

    with pytest.raises(NVRTCCompileError, match="kernel_0_relu"):
        compiler.compile(image)


def test_compile_pipeline_can_attach_optional_compiled_image(monkeypatch):
    compile_module = importlib.import_module("aicf.compile")

    def fake_compile_with_nvrtc(
        image,
        *,
        arch=None,
        library_path=None,
    ):
        assert arch == "86"
        assert library_path == "fake-nvrtc"
        return CUDACompiledImage(
            source_image=image,
            ptx=b"fake ptx",
            compiler="nvrtc",
            compiler_version=(13, 3),
            options=("--gpu-architecture=compute_86",),
        )

    monkeypatch.setattr(
        compile_module,
        "compile_with_nvrtc",
        fake_compile_with_nvrtc,
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
        cuda_arch="86",
        nvrtc_library="fake-nvrtc",
    )

    assert executable.compiled_image is not None

    result = executable.run(
        np.zeros((2, 4), dtype=np.float32)
    )
    assert result["status"] == "cuda_compiled"
    assert result["compiled"] is True
    assert result["launched"] is False
    assert result["compiler"] == "nvrtc"
    assert result["ptx_nbytes"] == len(b"fake ptx")


def test_real_nvrtc_can_compile_generated_fused_gemm_when_available():
    if not NVRTCCompiler.is_available():
        pytest.skip("NVRTC is not available on this machine")

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
    )

    compiled = executable.compiled_image
    assert compiled is not None
    assert compiled.ptx_nbytes > 0
    assert b"kernel_0_fused_gemm_bias_relu" in compiled.ptx