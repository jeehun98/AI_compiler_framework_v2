from __future__ import annotations


def format_cuda_execution(result) -> str:
    lines = [
        f"device      = {result.device_ordinal}",
        f"device_name = {result.device_name}",
        "launches    =",
    ]

    for launch in result.launches:
        lines.extend(
            [
                f"  {launch.kernel}:",
                f"    function = 0x{launch.function_handle:x}",
                f"    args     = [{', '.join(launch.argument_refs)}]",
                f"    grid     = {launch.grid}",
                f"    block    = {launch.block}",
            ]
        )

    lines.append(f"outputs     = {len(result.outputs)}")
    for index, output in enumerate(result.outputs):
        lines.append(
            f"  [{index}] shape={tuple(output.shape)}, "
            f"dtype={output.dtype}, nbytes={output.nbytes}"
        )

    return "\n".join(lines)