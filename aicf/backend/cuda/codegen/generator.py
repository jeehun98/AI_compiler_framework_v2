from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUDAExecutableImage:
    kernels: list[str]
    code: str
    plans: list[object] = field(default_factory=list)


def codegen(lowered) -> CUDAExecutableImage:
    """Materialize CUDA kernel plans into a mock executable image.

    Real CUDA C++/PTX generation is still deferred. The backend now receives
    the logical GEMM problem, tile and schedule plus a minimal physical
    thread-block mapping. Concrete thread coordinates and CUDA source emission
    are still deferred.
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

        if plan.tile is not None:
            chunks.append(
                "// gemm_tile: "
                f"BM={plan.tile.block_m}, "
                f"BN={plan.tile.block_n}, "
                f"BK={plan.tile.block_k}"
            )

        if plan.schedule is not None:
            chunks.append(
                "// gemm_schedule: "
                f"grid_m={plan.schedule.grid_m}, "
                f"grid_n={plan.schedule.grid_n}, "
                f"k_tiles={plan.schedule.k_tiles}"
            )

        if plan.block_mapping is not None:
            chunks.append(
                "// gemm_block_mapping: "
                f"threads={plan.block_mapping.threads}, "
                f"warps={plan.block_mapping.warps}, "
                "outputs_per_thread="
                f"{plan.block_mapping.outputs_per_thread}"
            )

        chunks.append(f"// TODO codegen for {plan.name}")

    return CUDAExecutableImage(
        kernels=kernel_names,
        code="\n".join(chunks),
        plans=list(lowered.kernels),
    )