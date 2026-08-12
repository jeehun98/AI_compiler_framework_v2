from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bindings import BoundBuffer, RuntimeBindings


@dataclass
class DeviceBuffer:
    """One CUDA allocation corresponding to one runtime IR value."""

    host: BoundBuffer
    pointer: int
    uploaded: bool = False

    @property
    def ref(self) -> str:
        return self.host.ref

    @property
    def nbytes(self) -> int:
        return self.host.nbytes

    @property
    def role(self) -> str:
        return self.host.slot.role

    @property
    def name(self) -> str:
        return self.host.slot.name


@dataclass
class DeviceBindings:
    """Per-invocation mapping from IR values to CUDA device pointers.

    v0.19 deliberately allocates one device buffer per host-bound IR value.
    Inputs and parameters are uploaded immediately. Outputs and temporaries are
    allocated but left uninitialized until a kernel writes them. Buffer reuse,
    asynchronous copies and streams are deferred.
    """

    host_bindings: RuntimeBindings
    loaded_image: object
    buffers: dict[str, DeviceBuffer]
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        host_bindings: RuntimeBindings,
        loaded_image,
    ) -> "DeviceBindings":
        buffers: dict[str, DeviceBuffer] = {}

        try:
            with loaded_image.activate_context() as driver:
                for ref, host in host_bindings.buffers.items():
                    pointer = driver.mem_alloc(host.nbytes)
                    device = DeviceBuffer(
                        host=host,
                        pointer=pointer,
                    )
                    buffers[ref] = device

                    if host.slot.role in {"input", "parameter"}:
                        array = _contiguous_host_array(host.array)
                        driver.memcpy_htod(
                            pointer,
                            int(array.ctypes.data),
                            int(array.nbytes),
                        )
                        device.uploaded = True
        except Exception:
            if buffers:
                _free_device_buffers(
                    loaded_image,
                    buffers,
                    suppress_errors=True,
                )
            raise

        return cls(
            host_bindings=host_bindings,
            loaded_image=loaded_image,
            buffers=buffers,
        )

    def buffer(self, ref: str) -> DeviceBuffer:
        if self.closed:
            raise RuntimeError("CUDA device bindings are closed")
        try:
            return self.buffers[ref]
        except KeyError as exc:
            raise KeyError(
                f"no CUDA device buffer is bound for IR value {ref}"
            ) from exc

    def kernel_arguments(self, plan) -> tuple[int, ...]:
        """Return CUdeviceptr values in the lowered kernel argument order."""

        refs = (*plan.inputs, *plan.outputs)
        return tuple(self.buffer(ref).pointer for ref in refs)

    def copy_to_host(
        self,
        ref: str,
        destination: np.ndarray | None = None,
    ) -> np.ndarray:
        """Copy one device buffer into host memory.

        If destination is omitted, the runtime-bound host array for the value is
        used. This method is primarily the D2H primitive that v0.20 will call
        after kernel execution.
        """

        device = self.buffer(ref)
        host_array = (
            device.host.array
            if destination is None
            else np.asarray(destination)
        )

        if tuple(host_array.shape) != tuple(device.host.array.shape):
            raise ValueError(
                f"D2H destination shape mismatch for {ref}: "
                f"expected {device.host.array.shape}, got {host_array.shape}"
            )
        if host_array.dtype != device.host.array.dtype:
            raise TypeError(
                f"D2H destination dtype mismatch for {ref}: "
                f"expected {device.host.array.dtype}, got {host_array.dtype}"
            )
        if not host_array.flags.c_contiguous:
            raise ValueError("D2H destination must be C-contiguous")

        with self.loaded_image.activate_context() as driver:
            driver.memcpy_dtoh(
                int(host_array.ctypes.data),
                device.pointer,
                device.nbytes,
            )

        return host_array

    def copy_outputs_to_host(self) -> tuple[np.ndarray, ...]:
        outputs = []
        for device in self.buffers.values():
            if device.role == "output":
                outputs.append(self.copy_to_host(device.ref))
        return tuple(outputs)

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "ref": device.ref,
                "name": device.name,
                "role": device.role,
                "nbytes": device.nbytes,
                "device_ptr": f"0x{device.pointer:x}",
                "uploaded": device.uploaded,
            }
            for device in self.buffers.values()
        ]

    def close(self) -> None:
        if self.closed:
            return
        _free_device_buffers(self.loaded_image, self.buffers)
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("CUDA device bindings are closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _contiguous_host_array(array: np.ndarray) -> np.ndarray:
    if array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array)


def _free_device_buffers(
    loaded_image,
    buffers: dict[str, DeviceBuffer],
    *,
    suppress_errors: bool = False,
) -> None:
    # Device allocations belong to the context in which they were created.
    # Free in reverse allocation order while the image's retained primary
    # context is current.
    first_error = None
    with loaded_image.activate_context() as driver:
        for device in reversed(tuple(buffers.values())):
            try:
                driver.mem_free(device.pointer)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

    if first_error is not None and not suppress_errors:
        raise first_error
