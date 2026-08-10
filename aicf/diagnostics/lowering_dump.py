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

        lines.append(f"  inputs    = [{', '.join(kernel.inputs)}]")
        lines.append(f"  outputs   = [{', '.join(kernel.outputs)}]")

        if kernel.attrs:
            lines.append(f"  attrs     = {kernel.attrs}")

    return "\n".join(lines)