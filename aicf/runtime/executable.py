from __future__ import annotations

from dataclasses import dataclass, field

from .bindings import RuntimeBindings, RuntimeSignature
from .cuda_execution import CUDAExecutionResult, execute_cuda, launch_shape
from .device_bindings import DeviceBindings


@dataclass
class Executable:
    image: object
    signature: RuntimeSignature
    compiled_image: object | None = None
    loaded_image: object | None = None
    last_cuda_execution: CUDAExecutionResult | None = field(
        default=None,
        repr=False,
    )

    def bind(self, *args) -> RuntimeBindings:
        """Bind concrete host inputs, parameters and result buffers."""

        return self.signature.bind(args)

    def bind_device(self, *args) -> DeviceBindings:
        """Allocate per-value CUDA buffers and upload inputs/parameters."""

        if self.loaded_image is None:
            raise RuntimeError(
                "CUDA device binding requires a loaded CUDA image; "
                "compile with cuda_compile=True and cuda_load=True"
            )

        host_bindings = self.bind(*args)
        return DeviceBindings.allocate(
            host_bindings,
            self.loaded_image,
        )

    def run_cuda(self, *args) -> CUDAExecutionResult:
        """Execute the lowered kernels synchronously on the loaded CUDA device."""

        result = execute_cuda(self, *args)
        self.last_cuda_execution = result
        return result

    def run(self, *args, **kwargs):
        if kwargs:
            raise TypeError("v0.20 runtime accepts positional inputs only")

        bindings = self.bind(*args)

        launches = []
        for plan in getattr(self.image, "plans", []):
            # Resolve the exact host-side argument contract for diagnostics.
            bindings.kernel_arguments(plan)

            launch = {
                "kernel": plan.name,
                "argument_refs": [*plan.inputs, *plan.outputs],
            }

            if plan.schedule is not None and plan.block_mapping is not None:
                grid, block = launch_shape(plan)
                launch["grid"] = grid
                launch["block"] = block

            launches.append(launch)

        compiled = self.compiled_image is not None
        loaded = self.loaded_image is not None
        launched = self.last_cuda_execution is not None
        status = (
            "cuda_executed"
            if launched
            else "cuda_loaded"
            if loaded
            else "cuda_compiled"
            if compiled
            else "host_bound"
        )
        result = {
            "status": status,
            "compiled": compiled,
            "loaded": loaded,
            "launched": launched,
            "kernels": list(getattr(self.image, "kernels", [])),
            "buffers": bindings.summary(),
            "launches": launches,
        }

        if compiled:
            result["compiler"] = getattr(
                self.compiled_image,
                "compiler",
                None,
            )
            result["ptx_nbytes"] = getattr(
                self.compiled_image,
                "ptx_nbytes",
                0,
            )

        if loaded:
            result["cuda_device"] = getattr(
                self.loaded_image,
                "device_ordinal",
                None,
            )
            result["cuda_device_name"] = getattr(
                self.loaded_image,
                "device_name",
                None,
            )
            result["driver_version"] = getattr(
                self.loaded_image,
                "driver_version",
                None,
            )
            result["resolved_kernels"] = list(
                getattr(self.loaded_image, "kernels", [])
            )

        if launched:
            result["executed_kernels"] = [
                launch.kernel
                for launch in self.last_cuda_execution.launches
            ]
            result["output_count"] = len(
                self.last_cuda_execution.outputs
            )

        return result

    def close(self) -> None:
        loaded = self.loaded_image
        if loaded is not None:
            close = getattr(loaded, "close", None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False