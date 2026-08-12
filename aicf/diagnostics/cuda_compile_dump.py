from __future__ import annotations


def format_cuda_compilation(compiled) -> str:
    version = getattr(compiled, "compiler_version", None)
    if version is None:
        version_text = "<unknown>"
    else:
        version_text = f"{version[0]}.{version[1]}"

    kernels = ", ".join(getattr(compiled, "kernels", [])) or "<none>"
    options = " ".join(getattr(compiled, "options", ())) or "<none>"

    lines = [
        f"compiler    = {getattr(compiled, 'compiler', '<unknown>')}",
        f"version     = {version_text}",
        f"kernels     = [{kernels}]",
        f"options     = {options}",
        f"ptx_nbytes  = {getattr(compiled, 'ptx_nbytes', 0)}",
    ]

    log = getattr(compiled, "log", "")
    if log:
        lines.append("log         =")
        lines.extend(f"  {line}" for line in log.splitlines())

    return "\n".join(lines)