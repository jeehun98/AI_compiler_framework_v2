"""Sequential-style model declaration through the user-facing nn API."""
import argparse

import numpy as np

from aicf import compile, nn
from aicf.frontend.tensor import TensorSpec
from aicf.diagnostics.events import add_listener, clear_listeners
from aicf.diagnostics.graph_dump import format_graph
from aicf.diagnostics.use_def_dump import format_use_def
from aicf.diagnostics.ir_use_def_dump import format_ir_use_def
from aicf.diagnostics.lowering_dump import format_cuda_lowering
from aicf.diagnostics.cuda_compile_dump import format_cuda_compilation
from aicf.diagnostics.cuda_driver_dump import format_cuda_loaded_image
from aicf.diagnostics.cuda_memory_dump import format_cuda_device_bindings
from aicf.diagnostics.cuda_launch_dump import format_cuda_execution
from aicf.ir.printer import format_ir


model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
)


def print_event(event, payload):
    print(f"\n[{event}]")
    if event == "graph.captured":
        print(format_graph(payload))
    elif event == "graph.use_def":
        print(format_use_def(payload))
    elif event == "ir.use_def":
        print(format_ir_use_def(payload["module"], payload["analysis"]))
    elif event.startswith("ir.") or event.startswith("pass."):
        module = payload.get("module") if isinstance(payload, dict) else payload
        print(format_ir(module))
    elif event == "lowering.finished":
        print(format_cuda_lowering(payload))
    elif event == "backend.codegen":
        print(payload.code)
    elif event == "backend.compiled":
        print(format_cuda_compilation(payload))
    elif event == "backend.loaded":
        print(format_cuda_loaded_image(payload))
    else:
        print(payload)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nvrtc",
        action="store_true",
        help="compile generated CUDA C++ to PTX with NVRTC",
    )
    parser.add_argument(
        "--cuda-load",
        action="store_true",
        help="load compiled PTX with the CUDA Driver API and resolve kernels",
    )
    parser.add_argument(
        "--cuda-run",
        action="store_true",
        help="compile, load and execute the generated CUDA kernel",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device ordinal used by --cuda-load",
    )
    parser.add_argument(
        "--cuda-arch",
        default=None,
        help="optional NVRTC target, e.g. 86 or compute_86",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("[model]")
    print(model)

    print("\n[parameters]")
    for name, parameter in model.named_parameters():
        print(
            f"{name}: shape={parameter.shape}, dtype={parameter.dtype}, "
            f"storage_nbytes={parameter.data.nbytes}"
        )

    # Use deterministic, non-zero data so the first real GPU execution is a
    # meaningful numerical check rather than a trivial all-zero result.
    model[0].weight.set_data(
        np.linspace(
            -0.02,
            0.02,
            64 * 128,
            dtype=np.float32,
        ).reshape(64, 128)
    )
    model[0].bias.set_data(
        np.linspace(-0.1, 0.1, 128, dtype=np.float32)
    )

    add_listener(print_event)
    exe = compile(
        model,
        [TensorSpec((32, 64), "float32", "x")],
        cuda_compile=(args.nvrtc or args.cuda_load or args.cuda_run),
        cuda_arch=args.cuda_arch,
        cuda_load=(args.cuda_load or args.cuda_run),
        cuda_device=args.cuda_device,
    )

    x = np.linspace(
        -1.0,
        1.0,
        32 * 64,
        dtype=np.float32,
    ).reshape(32, 64)

    if args.cuda_run:
        execution = exe.run_cuda(x)
        print("\n[cuda.execution]")
        print(format_cuda_execution(execution))

        reference = np.maximum(
            x @ model[0].weight.data + model[0].bias.data,
            np.float32(0),
        )
        output = execution.output
        difference = np.abs(output - reference)
        validation = {
            "allclose": bool(
                np.allclose(
                    output,
                    reference,
                    rtol=1e-4,
                    atol=1e-5,
                )
            ),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
        print("\n[numerical.validation]")
        print(validation)

    elif args.cuda_load:
        with exe.bind_device(x) as device_bindings:
            print("\n[device.bindings]")
            print(format_cuda_device_bindings(device_bindings))

            roundtrip = np.empty_like(x)
            device_bindings.copy_to_host("%0", roundtrip)
            print("\n[device.roundtrip]")
            print({"input_equal": bool(np.array_equal(roundtrip, x))})

    print("\n[runtime]")
    print(exe.run(x))
    exe.close()
    clear_listeners()