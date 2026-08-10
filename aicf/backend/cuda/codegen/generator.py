from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUDAExecutableImage:
    kernels: list[str]
    code: str
    plans: list[object] = field(default_factory=list)


def codegen(lowered) -> CUDAExecutableImage:
    """Materialize CUDA kernel plans into a mock executable image.

    Real CUDA C++/PTX generation is intentionally deferred. The backend now
    consumes target-side kernel plans, including explicit GEMM problem shapes,
    instead of raw source-IR operation names.
    """

    kernel_names = [plan.name for plan in lowered.kernels]

    chunks = []
    for plan in lowered.kernels:
        chunks.extend(
            [
                f"// {plan.name}",
                f"// source_op: {plan.source_op}",
                f"// strategy: {plan.strategy}",
            ]
        )

        if plan.problem is not None:
            chunks.append(
                "// gemm_problem: "
                f"M={plan.problem.m}, "
                f"N={plan.problem.n}, "
                f"K={plan.problem.k}, "
                f"dtype={plan.problem.dtype}"
            )

        chunks.append(f"// TODO codegen for {plan.name}")

    return CUDAExecutableImage(
        kernels=kernel_names,
        code="\n".join(chunks),
        plans=list(lowered.kernels),
    )