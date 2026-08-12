from __future__ import annotations

from dataclasses import dataclass

from .bindings import RuntimeBindings, RuntimeSignature


@dataclass
class Executable:
    image: object
    signature: RuntimeSignature
    compiled_image: object | None = None

    def bind(self, *args) -> RuntimeBindings:
        """Bind concrete host inputs, parameters and result buffers."""

        return self.signature.bind(args)

    def run(self, *args, **kwargs):
        if kwargs:
            raise TypeError("v0.17 runtime accepts positional inputs only")

        bindings = self.bind(*args)

        launches = []
        for plan in getattr(self.image, "plans", []):
            # Resolve the exact argument list now. Actual host->device transfer,
            # module loading and cuLaunchKernel are intentionally deferred.
            bindings.kernel_arguments(plan)

            launch = {
                "kernel": plan.name,
                "argument_refs": [*plan.inputs, *plan.outputs],
            }

            if (
                plan.schedule is not None
                and plan.block_mapping is not None
            ):
                launch["grid"] = (
                    plan.schedule.grid_n,
                    plan.schedule.grid_m,
                    1,
                )
                launch["block"] = (
                    plan.block_mapping.threads,
                    1,
                    1,
                )

            launches.append(launch)

        compiled = self.compiled_image is not None
        result = {
            "status": "cuda_compiled" if compiled else "host_bound",
            "compiled": compiled,
            "launched": False,
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

        return result