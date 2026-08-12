from __future__ import annotations

import argparse
from pathlib import Path

from aicf_labs.backend_cuda import ObservationInputs, generated_naive_gemm_bias_relu, observe_kernel


def _read(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8", errors="replace") if path else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ast")
    parser.add_argument("--ptx")
    parser.add_argument("--sass")
    args = parser.parse_args()

    kernel = generated_naive_gemm_bias_relu()
    observation = observe_kernel(
        kernel,
        ObservationInputs(
            ast_dump=_read(args.ast),
            ptx=_read(args.ptx),
            sass=_read(args.sass),
        ),
    )

    print("[kernel]")
    print(f"name={kernel.name}")
    print(f"entry={kernel.entry}")
    print(f"source={kernel.source_path}")
    print(f"launch=grid{kernel.launch.grid} block{kernel.launch.block}")

    for layer in (observation.source, observation.ast, observation.ptx, observation.sass):
        if layer is None:
            continue
        print()
        print(f"[{layer.layer}]")
        for key, value in layer.metrics.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
