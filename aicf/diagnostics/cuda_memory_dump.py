from __future__ import annotations


def format_cuda_device_bindings(bindings) -> str:
    lines = []
    for item in bindings.summary():
        lines.append(
            f"{item['ref']} {item['name']}: "
            f"role={item['role']}, "
            f"nbytes={item['nbytes']}, "
            f"device_ptr={item['device_ptr']}, "
            f"uploaded={item['uploaded']}"
        )
    return "\n".join(lines)