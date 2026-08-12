from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class NVRTCUnavailableError(RuntimeError):
    """Raised when the NVRTC shared library cannot be found or loaded."""


class NVRTCCompileError(RuntimeError):
    """Raised when NVRTC rejects generated CUDA C++ source."""

    def __init__(self, message: str, *, log: str = "") -> None:
        if log:
            message = f"{message}\n\nNVRTC log:\n{log}"
        super().__init__(message)
        self.log = log


@dataclass(frozen=True)
class CUDACompileOptions:
    """Small compile contract for the generated CUDA C++ source.

    v0.17 keeps target selection deliberately narrow. `arch` is optional;
    when omitted NVRTC chooses its default target. A caller may pass either
    `86` or an NVRTC architecture spelling such as `compute_86` / `sm_86`.
    """

    arch: str | None = None
    std: str = "c++17"
    extra: tuple[str, ...] = ()

    def nvrtc_args(self) -> tuple[str, ...]:
        args = [f"--std={self.std}"]

        arch = _normalize_arch(self.arch)
        if arch is not None:
            args.append(f"--gpu-architecture={arch}")

        args.extend(self.extra)
        return tuple(args)


@dataclass(frozen=True)
class CUDACompiledImage:
    """Device-code artifact produced from a CUDAExecutableImage.

    v0.17 stops at PTX production. Loading the PTX with the CUDA Driver API,
    resolving CUfunction handles and launching kernels are runtime work for the
    next stage.
    """

    source_image: object
    ptx: bytes
    compiler: str
    compiler_version: tuple[int, int] | None
    options: tuple[str, ...]
    log: str = ""

    @property
    def kernels(self) -> list[str]:
        return list(getattr(self.source_image, "kernels", []))

    @property
    def plans(self) -> list[object]:
        return list(getattr(self.source_image, "plans", []))

    @property
    def ptx_nbytes(self) -> int:
        return len(self.ptx)

    @property
    def ptx_text(self) -> str:
        return self.ptx.decode("utf-8")


def _normalize_arch(arch: str | None) -> str | None:
    if arch is None:
        return None

    arch = str(arch).strip()
    if not arch:
        return None

    if arch.isdigit():
        return f"compute_{arch}"

    if arch.startswith("compute_") or arch.startswith("sm_"):
        suffix = arch.split("_", 1)[1]
        if suffix.isdigit():
            return arch

    raise ValueError(
        "CUDA arch must look like '86', 'compute_86', or 'sm_86'"
    )


def _existing_glob(pattern: str) -> list[str]:
    return [path for path in glob.glob(pattern) if os.path.isfile(path)]


def _candidate_nvrtc_libraries() -> list[str]:
    candidates: list[str] = []

    explicit = os.environ.get("AICF_NVRTC_LIBRARY")
    if explicit:
        candidates.append(explicit)

    roots: list[Path] = []
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))

    if os.name == "nt":
        for root in roots:
            candidates.extend(
                sorted(
                    _existing_glob(str(root / "bin" / "nvrtc64_*.dll")),
                    reverse=True,
                )
            )

        # CUDA's bin directory is commonly on PATH even when CUDA_PATH is not
        # visible to a child process. Search only for the NVRTC DLL name.
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            if entry:
                candidates.extend(
                    sorted(
                        _existing_glob(str(Path(entry) / "nvrtc64_*.dll")),
                        reverse=True,
                    )
                )
    else:
        for root in roots:
            candidates.extend(
                sorted(
                    _existing_glob(str(root / "lib64" / "libnvrtc.so*")),
                    reverse=True,
                )
            )
            candidates.extend(
                sorted(
                    _existing_glob(str(root / "lib" / "libnvrtc.so*")),
                    reverse=True,
                )
            )

    found = ctypes.util.find_library("nvrtc")
    if found:
        candidates.append(found)

    # Preserve discovery order while removing duplicates.
    return list(dict.fromkeys(candidates))


def find_nvrtc_library() -> str:
    candidates = _candidate_nvrtc_libraries()
    if not candidates:
        raise NVRTCUnavailableError(
            "NVRTC library was not found. Install the CUDA Toolkit or set "
            "AICF_NVRTC_LIBRARY to the nvrtc shared-library path."
        )
    return candidates[0]


class _NVRTCLibrary:
    NVRTC_SUCCESS = 0

    def __init__(self, library_path: str | None = None) -> None:
        path = library_path or find_nvrtc_library()
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as exc:
            raise NVRTCUnavailableError(
                f"failed to load NVRTC library: {path}"
            ) from exc

        self.path = path
        self._bind_api()

    def _bind_api(self) -> None:
        lib = self._lib

        lib.nvrtcVersion.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.nvrtcVersion.restype = ctypes.c_int

        lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p

        lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.nvrtcCreateProgram.restype = ctypes.c_int

        lib.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.nvrtcCompileProgram.restype = ctypes.c_int

        lib.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.nvrtcGetProgramLogSize.restype = ctypes.c_int

        lib.nvrtcGetProgramLog.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.nvrtcGetProgramLog.restype = ctypes.c_int

        lib.nvrtcGetPTXSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.nvrtcGetPTXSize.restype = ctypes.c_int

        lib.nvrtcGetPTX.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.nvrtcGetPTX.restype = ctypes.c_int

        lib.nvrtcDestroyProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nvrtcDestroyProgram.restype = ctypes.c_int

    def _error_string(self, result: int) -> str:
        raw = self._lib.nvrtcGetErrorString(result)
        if raw is None:
            return f"NVRTC error {result}"
        return raw.decode("utf-8", errors="replace")

    def _check(self, result: int, action: str) -> None:
        if result != self.NVRTC_SUCCESS:
            raise NVRTCCompileError(
                f"{action} failed: {self._error_string(result)}"
            )

    def version(self) -> tuple[int, int]:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self._check(
            self._lib.nvrtcVersion(
                ctypes.byref(major),
                ctypes.byref(minor),
            ),
            "nvrtcVersion",
        )
        return major.value, minor.value

    def _program_log(self, program: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        result = self._lib.nvrtcGetProgramLogSize(
            program,
            ctypes.byref(size),
        )
        if result != self.NVRTC_SUCCESS or size.value <= 1:
            return ""

        buffer = ctypes.create_string_buffer(size.value)
        result = self._lib.nvrtcGetProgramLog(program, buffer)
        if result != self.NVRTC_SUCCESS:
            return ""

        return buffer.value.decode("utf-8", errors="replace")

    def compile(
        self,
        source: str,
        *,
        name: str,
        options: Iterable[str],
    ) -> tuple[bytes, str]:
        program = ctypes.c_void_p()

        result = self._lib.nvrtcCreateProgram(
            ctypes.byref(program),
            source.encode("utf-8"),
            name.encode("utf-8"),
            0,
            None,
            None,
        )
        self._check(result, "nvrtcCreateProgram")

        try:
            encoded_options = [
                option.encode("utf-8")
                for option in options
            ]
            if encoded_options:
                option_array = (ctypes.c_char_p * len(encoded_options))(
                    *encoded_options
                )
                option_ptr = option_array
            else:
                option_ptr = None

            result = self._lib.nvrtcCompileProgram(
                program,
                len(encoded_options),
                option_ptr,
            )
            log = self._program_log(program)

            if result != self.NVRTC_SUCCESS:
                raise NVRTCCompileError(
                    "nvrtcCompileProgram failed: "
                    f"{self._error_string(result)}",
                    log=log,
                )

            ptx_size = ctypes.c_size_t()
            self._check(
                self._lib.nvrtcGetPTXSize(
                    program,
                    ctypes.byref(ptx_size),
                ),
                "nvrtcGetPTXSize",
            )

            ptx_buffer = ctypes.create_string_buffer(ptx_size.value)
            self._check(
                self._lib.nvrtcGetPTX(program, ptx_buffer),
                "nvrtcGetPTX",
            )

            # nvrtcGetPTXSize includes the trailing NUL byte.
            ptx = bytes(ptx_buffer.raw).rstrip(b"\0")
            return ptx, log
        finally:
            # Destruction failure should not mask the primary compile result.
            self._lib.nvrtcDestroyProgram(ctypes.byref(program))


class NVRTCCompiler:
    """Compile a complete generated CUDA source image into PTX."""

    def __init__(
        self,
        library_path: str | None = None,
        *,
        _library=None,
    ) -> None:
        self._library = (
            _library
            if _library is not None
            else _NVRTCLibrary(library_path)
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            path = find_nvrtc_library()
            _NVRTCLibrary(path)
        except NVRTCUnavailableError:
            return False
        return True

    @property
    def version(self) -> tuple[int, int] | None:
        version = getattr(self._library, "version", None)
        if version is None:
            return None
        return version()

    def compile(
        self,
        image,
        *,
        options: CUDACompileOptions | None = None,
        program_name: str = "aicf_generated.cu",
    ) -> CUDACompiledImage:
        unresolved = tuple(
            getattr(image, "unresolved_kernels", ())
        )
        if unresolved:
            joined = ", ".join(unresolved)
            raise NVRTCCompileError(
                "cannot compile CUDA image because executable codegen is "
                f"not implemented for: {joined}"
            )

        options = options or CUDACompileOptions()
        nvrtc_args = options.nvrtc_args()

        ptx, log = self._library.compile(
            image.code,
            name=program_name,
            options=nvrtc_args,
        )

        return CUDACompiledImage(
            source_image=image,
            ptx=ptx,
            compiler="nvrtc",
            compiler_version=self.version,
            options=nvrtc_args,
            log=log,
        )


def compile_with_nvrtc(
    image,
    *,
    arch: str | None = None,
    extra_options: Iterable[str] = (),
    library_path: str | None = None,
) -> CUDACompiledImage:
    compiler = NVRTCCompiler(library_path)
    options = CUDACompileOptions(
        arch=arch,
        extra=tuple(extra_options),
    )
    return compiler.compile(image, options=options)
