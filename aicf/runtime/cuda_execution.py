from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device_bindings import DeviceBindings


@dataclass(frozen=True)
class CUDALaunchRecord:
    kernel: str
    function_handle: int
    argument_refs: tuple[str, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


@dataclass
class CUDAExecutionResult:
    """Result of one synchronous CUDA execution.

    v0.20 intentionally uses the default stream and synchronizes the current
    CUDA context before copying outputs back to host memory. Streams, events,
    async copies and overlapping execution are deferred to later runtime work.
    """

    outputs: tuple[np.ndarray, ...]
    launches: tuple[CUDALaunchRecord, ...]
    device_ordinal: int
    device_name: str

    @property
    def output(self) -> np.ndarray:
        if len(self.outputs) != 1:
            raise RuntimeError(
                f"expected exactly one CUDA output, got {len(self.outputs)}"
            )
        return self.outputs[0]


def launch_shape(plan) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if plan.schedule is None or plan.block_mapping is None:
        raise RuntimeError(
            f"CUDA launch plan is incomplete for kernel {plan.name}"
        )

    grid = (
        int(plan.schedule.grid_n),
        int(plan.schedule.grid_m),
        1,
    )
    block = (
        int(plan.block_mapping.threads),
        1,
        1,
    )
    return grid, block


def execute_cuda(executable, *args) -> CUDAExecutionResult:
    loaded = executable.loaded_image
    if loaded is None:
        raise RuntimeError(
            "CUDA execution requires a loaded CUDA image; compile with "
            "cuda_compile=True and cuda_load=True"
        )

    host_bindings = executable.bind(*args)
    launches: list[CUDALaunchRecord] = []

    with DeviceBindings.allocate(host_bindings, loaded) as device_bindings:
        with loaded.activate_context() as driver:
            for plan in getattr(executable.image, "plans", []):
                function = loaded.function(plan.name)
                kernel_params = device_bindings.kernel_arguments(plan)
                grid, block = launch_shape(plan)

                driver.launch_kernel(
                    function.handle,
                    grid=grid,
                    block=block,
                    kernel_params=kernel_params,
                )

                launches.append(
                    CUDALaunchRecord(
                        kernel=plan.name,
                        function_handle=function.handle,
                        argument_refs=tuple((*plan.inputs, *plan.outputs)),
                        grid=grid,
                        block=block,
                    )
                )

            # cuLaunchKernel is asynchronous with respect to host execution.
            # v0.20 chooses the simplest correctness-first boundary: wait for
            # all work in the current context before any D2H output copy.
            driver.synchronize()

        outputs = device_bindings.copy_outputs_to_host()

    return CUDAExecutionResult(
        outputs=outputs,
        launches=tuple(launches),
        device_ordinal=int(loaded.device_ordinal),
        device_name=str(loaded.device_name),
    )