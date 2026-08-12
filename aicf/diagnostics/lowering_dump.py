from __future__ import annotations


def format_cuda_lowering(lowered) -> str:
    if not lowered.kernels:
        return "<no kernels>"

    lines = []

    for kernel in lowered.kernels:
        lines.append(f"{kernel.name}:")
        lines.append(f"  source_op = {kernel.source_op}")
        lines.append(f"  strategy  = {kernel.strategy}")

        if kernel.problem is not None:
            lines.append(
                "  problem   = "
                f"GEMM(M={kernel.problem.m}, "
                f"N={kernel.problem.n}, "
                f"K={kernel.problem.k}, "
                f"dtype={kernel.problem.dtype})"
            )

        if kernel.tile is not None:
            lines.append(
                "  tile      = "
                f"GEMMTile(BM={kernel.tile.block_m}, "
                f"BN={kernel.tile.block_n}, "
                f"BK={kernel.tile.block_k})"
            )

        if kernel.schedule is not None:
            lines.append(
                "  schedule  = "
                f"GEMMSchedule(grid_m={kernel.schedule.grid_m}, "
                f"grid_n={kernel.schedule.grid_n}, "
                f"k_tiles={kernel.schedule.k_tiles})"
            )

        if kernel.block_mapping is not None:
            lines.append(
                "  mapping   = "
                f"GEMMBlockMapping(threads={kernel.block_mapping.threads}, "
                f"warps={kernel.block_mapping.warps}, "
                "outputs_per_thread="
                f"{kernel.block_mapping.outputs_per_thread})"
            )

        lines.append(f"  inputs    = [{', '.join(kernel.inputs)}]")
        lines.append(f"  outputs   = [{', '.join(kernel.outputs)}]")

        if kernel.attrs:
            lines.append(f"  attrs     = {kernel.attrs}")

    return "\n".join(lines)