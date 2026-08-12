from __future__ import annotations

import ctypes
import ctypes.util
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable


class CUDADriverUnavailableError(RuntimeError):
    """Raised when the CUDA Driver shared library cannot be loaded."""


class CUDADriverError(RuntimeError):
    """Raised when a CUDA Driver API call returns a non-success CUresult."""

    def __init__(
        self,
        message: str,
        *,
        result: int | None = None,
        error_name: str | None = None,
        error_string: str | None = None,
    ) -> None:
        details = []
        if error_name:
            details.append(error_name)
        if error_string:
            details.append(error_string)
        if result is not None:
            details.append(f"code={result}")
        if details:
            message = f"{message}: " + " | ".join(details)
        super().__init__(message)
        self.result = result
        self.error_name = error_name
        self.error_string = error_string


def _driver_library_candidates() -> list[str]:
    candidates: list[str] = []

    explicit = os.environ.get("AICF_CUDA_DRIVER_LIBRARY")
    if explicit:
        candidates.append(explicit)

    if os.name == "nt":
        found = ctypes.util.find_library("nvcuda")
        if found:
            candidates.append(found)
        candidates.append("nvcuda.dll")
    else:
        found = ctypes.util.find_library("cuda")
        if found:
            candidates.append(found)
        candidates.extend(("libcuda.so.1", "libcuda.so"))

    return list(dict.fromkeys(candidates))


def _library_loader():
    if os.name == "nt":
        return getattr(ctypes, "WinDLL", ctypes.CDLL)
    return ctypes.CDLL


def find_cuda_driver_library() -> str:
    loader = _library_loader()
    errors = []
    for candidate in _driver_library_candidates():
        try:
            loader(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        return candidate

    detail = "\n".join(errors)
    message = (
        "CUDA Driver library was not found. Install an NVIDIA display driver "
        "or set AICF_CUDA_DRIVER_LIBRARY to the driver-library path."
    )
    if detail:
        message += f"\nTried:\n{detail}"
    raise CUDADriverUnavailableError(message)


def _handle_value(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    raw = getattr(value, "value", value)
    return int(raw or 0)


_CUdeviceptr = ctypes.c_uint64


class _CUDADriverLibrary:
    CUDA_SUCCESS = 0

    def __init__(self, library_path: str | None = None) -> None:
        path = library_path or find_cuda_driver_library()
        loader = _library_loader()
        try:
            self._lib = loader(path)
        except OSError as exc:
            raise CUDADriverUnavailableError(
                f"failed to load CUDA Driver library: {path}"
            ) from exc

        self.path = path
        self._bind_api()

    def _symbol(self, *names: str):
        for name in names:
            symbol = getattr(self._lib, name, None)
            if symbol is not None:
                return symbol
        raise CUDADriverUnavailableError(
            "CUDA Driver library is missing required entry point: "
            + " or ".join(names)
        )

    def _bind_api(self) -> None:
        self._cuInit = self._symbol("cuInit")
        self._cuInit.argtypes = [ctypes.c_uint]
        self._cuInit.restype = ctypes.c_int

        self._cuDriverGetVersion = self._symbol("cuDriverGetVersion")
        self._cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._cuDriverGetVersion.restype = ctypes.c_int

        self._cuDeviceGetCount = self._symbol("cuDeviceGetCount")
        self._cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._cuDeviceGetCount.restype = ctypes.c_int

        self._cuDeviceGet = self._symbol("cuDeviceGet")
        self._cuDeviceGet.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self._cuDeviceGet.restype = ctypes.c_int

        self._cuDeviceGetName = self._symbol("cuDeviceGetName")
        self._cuDeviceGetName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._cuDeviceGetName.restype = ctypes.c_int

        self._cuDevicePrimaryCtxRetain = self._symbol(
            "cuDevicePrimaryCtxRetain"
        )
        self._cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self._cuDevicePrimaryCtxRetain.restype = ctypes.c_int

        self._cuDevicePrimaryCtxRelease = self._symbol(
            "cuDevicePrimaryCtxRelease_v2",
            "cuDevicePrimaryCtxRelease",
        )
        self._cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self._cuDevicePrimaryCtxRelease.restype = ctypes.c_int

        self._cuCtxGetCurrent = self._symbol("cuCtxGetCurrent")
        self._cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._cuCtxGetCurrent.restype = ctypes.c_int

        self._cuCtxSetCurrent = self._symbol("cuCtxSetCurrent")
        self._cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self._cuCtxSetCurrent.restype = ctypes.c_int

        self._cuModuleLoadData = self._symbol("cuModuleLoadData")
        self._cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._cuModuleLoadData.restype = ctypes.c_int

        self._cuModuleGetFunction = self._symbol("cuModuleGetFunction")
        self._cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._cuModuleGetFunction.restype = ctypes.c_int

        self._cuModuleUnload = self._symbol("cuModuleUnload")
        self._cuModuleUnload.argtypes = [ctypes.c_void_p]
        self._cuModuleUnload.restype = ctypes.c_int

        # Use the v2 memory ABI consistently. CUDA's Driver API versioning
        # rules require resources allocated with cuMemAlloc_v2 to be released
        # with the matching cuMemFree_v2 entry point.
        self._cuMemAlloc = self._symbol("cuMemAlloc_v2")
        self._cuMemAlloc.argtypes = [
            ctypes.POINTER(_CUdeviceptr),
            ctypes.c_size_t,
        ]
        self._cuMemAlloc.restype = ctypes.c_int

        self._cuMemFree = self._symbol("cuMemFree_v2")
        self._cuMemFree.argtypes = [_CUdeviceptr]
        self._cuMemFree.restype = ctypes.c_int

        self._cuMemcpyHtoD = self._symbol("cuMemcpyHtoD_v2")
        self._cuMemcpyHtoD.argtypes = [
            _CUdeviceptr,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._cuMemcpyHtoD.restype = ctypes.c_int

        self._cuMemcpyDtoH = self._symbol("cuMemcpyDtoH_v2")
        self._cuMemcpyDtoH.argtypes = [
            ctypes.c_void_p,
            _CUdeviceptr,
            ctypes.c_size_t,
        ]
        self._cuMemcpyDtoH.restype = ctypes.c_int

        self._cuGetErrorName = self._symbol("cuGetErrorName")
        self._cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._cuGetErrorName.restype = ctypes.c_int

        self._cuGetErrorString = self._symbol("cuGetErrorString")
        self._cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._cuGetErrorString.restype = ctypes.c_int

    def _error_text(self, result: int) -> tuple[str | None, str | None]:
        name = ctypes.c_char_p()
        description = ctypes.c_char_p()

        name_result = self._cuGetErrorName(result, ctypes.byref(name))
        text_result = self._cuGetErrorString(
            result,
            ctypes.byref(description),
        )

        name_text = None
        description_text = None
        if name_result == self.CUDA_SUCCESS and name.value:
            name_text = name.value.decode("utf-8", errors="replace")
        if text_result == self.CUDA_SUCCESS and description.value:
            description_text = description.value.decode(
                "utf-8",
                errors="replace",
            )
        return name_text, description_text

    def _check(self, result: int, action: str) -> None:
        if result == self.CUDA_SUCCESS:
            return
        name, description = self._error_text(result)
        raise CUDADriverError(
            f"{action} failed",
            result=result,
            error_name=name,
            error_string=description,
        )

    def initialize(self) -> None:
        self._check(self._cuInit(0), "cuInit")

    def driver_version(self) -> int:
        version = ctypes.c_int()
        self._check(
            self._cuDriverGetVersion(ctypes.byref(version)),
            "cuDriverGetVersion",
        )
        return int(version.value)

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(
            self._cuDeviceGetCount(ctypes.byref(count)),
            "cuDeviceGetCount",
        )
        return int(count.value)

    def device(self, ordinal: int) -> int:
        device = ctypes.c_int()
        self._check(
            self._cuDeviceGet(ctypes.byref(device), int(ordinal)),
            "cuDeviceGet",
        )
        return int(device.value)

    def device_name(self, device: int) -> str:
        buffer = ctypes.create_string_buffer(256)
        self._check(
            self._cuDeviceGetName(buffer, len(buffer), int(device)),
            "cuDeviceGetName",
        )
        return buffer.value.decode("utf-8", errors="replace")

    def retain_primary_context(self, device: int) -> int:
        context = ctypes.c_void_p()
        self._check(
            self._cuDevicePrimaryCtxRetain(
                ctypes.byref(context),
                int(device),
            ),
            "cuDevicePrimaryCtxRetain",
        )
        return _handle_value(context)

    def release_primary_context(self, device: int) -> None:
        self._check(
            self._cuDevicePrimaryCtxRelease(int(device)),
            "cuDevicePrimaryCtxRelease",
        )

    def current_context(self) -> int | None:
        context = ctypes.c_void_p()
        self._check(
            self._cuCtxGetCurrent(ctypes.byref(context)),
            "cuCtxGetCurrent",
        )
        value = _handle_value(context)
        return value or None

    def set_current_context(self, context: int | None) -> None:
        handle = ctypes.c_void_p(0 if context is None else int(context))
        self._check(
            self._cuCtxSetCurrent(handle),
            "cuCtxSetCurrent",
        )

    def load_module(self, ptx: bytes) -> int:
        if not ptx:
            raise ValueError("cannot load empty PTX")

        image = ptx if ptx.endswith(b"\0") else ptx + b"\0"
        buffer = ctypes.create_string_buffer(image)
        module = ctypes.c_void_p()
        self._check(
            self._cuModuleLoadData(
                ctypes.byref(module),
                ctypes.cast(buffer, ctypes.c_void_p),
            ),
            "cuModuleLoadData",
        )
        return _handle_value(module)

    def get_function(self, module: int, name: str) -> int:
        function = ctypes.c_void_p()
        self._check(
            self._cuModuleGetFunction(
                ctypes.byref(function),
                ctypes.c_void_p(int(module)),
                name.encode("utf-8"),
            ),
            f"cuModuleGetFunction({name})",
        )
        return _handle_value(function)

    def unload_module(self, module: int) -> None:
        self._check(
            self._cuModuleUnload(ctypes.c_void_p(int(module))),
            "cuModuleUnload",
        )

    def mem_alloc(self, nbytes: int) -> int:
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("CUDA device allocation size must be positive")

        pointer = _CUdeviceptr()
        self._check(
            self._cuMemAlloc(ctypes.byref(pointer), nbytes),
            "cuMemAlloc_v2",
        )
        return int(pointer.value)

    def mem_free(self, pointer: int) -> None:
        pointer = int(pointer)
        if pointer == 0:
            return
        self._check(
            self._cuMemFree(_CUdeviceptr(pointer)),
            "cuMemFree_v2",
        )

    def memcpy_htod(
        self,
        dst_device: int,
        src_host: int,
        nbytes: int,
    ) -> None:
        self._check(
            self._cuMemcpyHtoD(
                _CUdeviceptr(int(dst_device)),
                ctypes.c_void_p(int(src_host)),
                int(nbytes),
            ),
            "cuMemcpyHtoD_v2",
        )

    def memcpy_dtoh(
        self,
        dst_host: int,
        src_device: int,
        nbytes: int,
    ) -> None:
        self._check(
            self._cuMemcpyDtoH(
                ctypes.c_void_p(int(dst_host)),
                _CUdeviceptr(int(src_device)),
                int(nbytes),
            ),
            "cuMemcpyDtoH_v2",
        )


@dataclass(frozen=True)
class CUDAKernelHandle:
    name: str
    handle: int


@dataclass
class CUDALoadedImage:
    """PTX loaded into one retained CUDA primary context.

    v0.18 resolves module/function handles only. Device memory and kernel
    launch are intentionally deferred. The loaded module must remain alive for
    its function handles to stay valid, so the artifact owns that lifetime and
    exposes close().
    """

    compiled_image: object
    device_ordinal: int
    device_name: str
    driver_version: int | None
    context_handle: int
    module_handle: int
    functions: dict[str, CUDAKernelHandle]
    _library: object = field(repr=False)
    _device: int = field(repr=False)
    closed: bool = False

    @property
    def kernels(self) -> list[str]:
        return list(self.functions)

    def function(self, name: str) -> CUDAKernelHandle:
        if self.closed:
            raise RuntimeError("CUDA loaded image is closed")
        try:
            return self.functions[name]
        except KeyError as exc:
            raise KeyError(f"CUDA kernel handle not found: {name}") from exc

    @property
    def driver_api(self):
        if self.closed:
            raise RuntimeError("CUDA loaded image is closed")
        return self._library

    @contextmanager
    def activate_context(self):
        """Temporarily make this image's retained primary context current."""

        if self.closed:
            raise RuntimeError("CUDA loaded image is closed")

        previous = self._library.current_context()
        self._library.set_current_context(self.context_handle)
        try:
            yield self._library
        finally:
            self._library.set_current_context(previous)

    def close(self) -> None:
        if self.closed:
            return

        previous = self._library.current_context()
        try:
            self._library.set_current_context(self.context_handle)
            self._library.unload_module(self.module_handle)
        finally:
            try:
                self._library.set_current_context(previous)
            finally:
                self._library.release_primary_context(self._device)
                self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("CUDA loaded image is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class CUDADriver:
    """Minimal Driver API boundary for loading PTX and resolving kernels."""

    def __init__(
        self,
        library_path: str | None = None,
        *,
        _library=None,
    ) -> None:
        self._library = (
            _library
            if _library is not None
            else _CUDADriverLibrary(library_path)
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            driver = cls()
            driver._library.initialize()
            return driver._library.device_count() > 0
        except (CUDADriverUnavailableError, CUDADriverError):
            return False

    def load(
        self,
        compiled_image,
        *,
        device_ordinal: int = 0,
    ) -> CUDALoadedImage:
        library = self._library
        library.initialize()

        count = library.device_count()
        if device_ordinal < 0 or device_ordinal >= count:
            raise ValueError(
                f"CUDA device ordinal {device_ordinal} is out of range; "
                f"available device count is {count}"
            )

        device = library.device(device_ordinal)
        device_name = library.device_name(device)
        driver_version = library.driver_version()

        previous = library.current_context()
        context = library.retain_primary_context(device)
        module = None

        try:
            library.set_current_context(context)
            module = library.load_module(compiled_image.ptx)

            functions = {}
            for kernel_name in compiled_image.kernels:
                handle = library.get_function(module, kernel_name)
                functions[kernel_name] = CUDAKernelHandle(
                    name=kernel_name,
                    handle=handle,
                )
        except Exception:
            if module is not None:
                try:
                    library.unload_module(module)
                except Exception:
                    pass
            try:
                library.set_current_context(previous)
            finally:
                library.release_primary_context(device)
            raise
        else:
            library.set_current_context(previous)

        return CUDALoadedImage(
            compiled_image=compiled_image,
            device_ordinal=device_ordinal,
            device_name=device_name,
            driver_version=driver_version,
            context_handle=context,
            module_handle=module,
            functions=functions,
            _library=library,
            _device=device,
        )


def load_with_cuda_driver(
    compiled_image,
    *,
    device_ordinal: int = 0,
    library_path: str | None = None,
) -> CUDALoadedImage:
    driver = CUDADriver(library_path)
    return driver.load(
        compiled_image,
        device_ordinal=device_ordinal,
    )
