# Backend CUDA Observation

The backend does not begin by generating a new optimized kernel. It begins with a fixed CUDA implementation and asks how the mathematical code is represented and executed.

## Kernel definition

`backend_cuda/kernels/KernelSpec` records the kernel entry, source, workload kind, architecture, compile options, and launch configuration.

The frozen v0.20 `GEMM + Bias + ReLU` kernel is the first observation target.

## Observation layers

```text
CUDA source
→ AST
→ PTX
→ SASS
→ runtime
```

`backend_cuda/observation/` contains parsers that summarize already-collected artifacts. The parsers are deliberately separate from the tools that produce those artifacts.

## Tool adapters

`backend_cuda/tools/` provides small adapters/command builders for:

- `clang++` AST dump
- `nvcc` PTX/cubin generation
- `cuobjdump` SASS/resource dump
- Nsight Compute (`ncu`) profiling

Tool availability is environment-dependent. The core repository tests do not require these tools.

## Intended experiment loop

```text
Kernel A
→ collect source/AST/PTX/SASS/runtime observations
→ identify a concrete execution hypothesis
→ write/select Kernel B
→ collect the same observations
→ compare artifacts and measurements
```

The important result is not only that B is faster, but which source-level change produced which PTX/SASS/runtime change.
