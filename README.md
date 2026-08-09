# AICF Skeleton v0.1

A deliberately small **AI framework / compiler skeleton**. The goal is not to
implement performance features yet, but to establish the places where each
feature will live and where each layer can be observed.

## Pipeline

```text
User Model / Ops
      ↓
Frontend
      ↓
Graph
      ↓
High-level IR
      ↓
PassManager / Compiler decisions
      ↓
Optimized IR
      ↓
Lowering
      ↓
CUDA Backend / Codegen
      ↓
Runtime
      ↓
Hardware
```

`diagnostics/` observes the pipeline. `experiments/` exercises the pipeline.
Neither is the center of the framework.

## Directory roles

```text
aicf/
├─ frontend/       # Tensor, Module, user-facing ops, future capture/tracing
├─ graph/          # Graph / Node / Value and graph validation
├─ ir/             # Internal representation and printer
├─ compiler/       # PassManager, analyses, optimization passes
├─ lowering/       # Decisions already made above → target representation
├─ backend/        # CUDA target, kernels, code generation
├─ runtime/        # Executable, executor, memory/runtime state
├─ diagnostics/    # Graph/IR dump, decision logging, tracing hooks
└─ profiler/       # Timer, NVTX, future Nsight Compute integration

experiments/       # Small programs that probe framework behavior
tests/             # Correctness/structural tests
```

## What is real in v0.1

- user-facing symbolic `TensorSpec`
- graph capture context
- `Graph / Node / Value`
- graph → IR conversion
- `IRModule / Operation / IRValue / TensorType`
- textual IR printer
- `Pass` and `PassManager`
- mock `CanonicalizePass`
- mock `FusionPass` that only detects `gemm → bias_add → relu` and logs a decision
- placeholder CUDA lowering
- placeholder CUDA codegen
- mock runtime executable
- cross-layer diagnostics event hook

## What is intentionally NOT implemented

- real tensor storage or eager execution
- autograd
- robust shape/type inference
- alias/mutation analysis
- actual fusion rewrite
- legality / profitability model
- tile/thread mapping
- CUDA kernels
- PTX/SASS generation
- allocator / CUDA Graph runtime
- NVTX/NCU integration

Those become individual implementation/study tasks as the framework grows.

## First experiment

```bash
python -m experiments.fusion.gemm_bias_relu
```

Expected conceptual output:

```text
graph.captured
      ↓
ir.created
      ↓
canonicalize
      ↓
fusion candidate detected, rewrite not implemented
      ↓
ir.optimized
      ↓
CUDA lowering placeholder
      ↓
backend placeholder
      ↓
mock runtime
```

## Development rule

When adding a feature, implement it in the layer that owns the decision:

- semantic/model behavior → frontend/graph
- optimization decision → compiler
- execution strategy already selected → lowering
- target implementation → backend
- launch/memory/lifetime → runtime
- observation only → diagnostics/profiler

This keeps `lowering` from becoming a second optimizer and keeps experiments
from becoming the framework itself.
