from __future__ import annotations

from dataclasses import asdict

from ..backend_cuda.implementations import GeneratedNaive
from ..contracts import WorkloadSpec


def describe_v020_codegen_baseline(workload: WorkloadSpec) -> dict[str, object]:
    """Return the frozen planning identity of the old v0.20 codegen path.

    Real CUDA source/NVRTC/Driver execution will be migrated behind the
    generated_naive backend implementation in later rearchitecture steps.
    """

    implementation = GeneratedNaive()
    return {
        "implementation": implementation.spec.name,
        "source": implementation.spec.source,
        "plan": asdict(implementation.plan(workload)),
    }
