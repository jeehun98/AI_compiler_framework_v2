from __future__ import annotations


def format_cuda_loaded_image(loaded) -> str:
    kernels = getattr(loaded, "kernels", [])
    functions = getattr(loaded, "functions", {})

    lines = [
        f"device      = {getattr(loaded, 'device_ordinal', '<unknown>')}",
        f"device_name = {getattr(loaded, 'device_name', '<unknown>')}",
        f"driver_ver  = {getattr(loaded, 'driver_version', '<unknown>')}",
        f"context     = 0x{getattr(loaded, 'context_handle', 0):x}",
        f"module      = 0x{getattr(loaded, 'module_handle', 0):x}",
        "functions   =",
    ]

    if not kernels:
        lines.append("  <none>")
    else:
        for name in kernels:
            function = functions[name]
            lines.append(f"  {name} -> 0x{function.handle:x}")

    return "\n".join(lines)