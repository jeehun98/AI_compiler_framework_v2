from pathlib import Path

from aicf_labs.backend_cuda.tools import ClangAstTool, CuobjdumpTool, NvccTool


def test_external_tool_command_builders_do_not_require_tool_installation():
    source = Path("kernel.cu")
    assert NvccTool().ptx_command(source, Path("kernel.ptx"), "sm_86")[:3] == (
        "nvcc",
        "-ptx",
        "-arch=sm_86",
    )
    assert CuobjdumpTool().sass_command(Path("kernel.cubin")) == (
        "cuobjdump",
        "--dump-sass",
        "kernel.cubin",
    )
    command = ClangAstTool().ast_command(source, "sm_86")
    assert "-ast-dump" in command
