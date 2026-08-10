# AICF Skeleton v0.2

A minimal AI framework/compiler skeleton intended to be filled in feature by feature.
The current focus is **structure**, not production execution.

## User-facing model API

The first structural revision uses an `nn`-style model declaration instead of passing
weights and bias as explicit model-function inputs.

```python
from aicf import nn

model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
)
```

Compilation receives only the actual user/program input:

```python
from aicf import compile
from aicf.frontend.tensor import TensorSpec

exe = compile(
    model,
    [TensorSpec((32, 64), "float32", "x")],
)
```

`Linear` owns its model state:

```text
Sequential
└─ Linear
   ├─ weight : Parameter[64, 128]
   └─ bias   : Parameter[128]
└─ ReLU
```

The graph therefore separates:

```text
user input       model state          temporary values
   %0          %1, %2                 %3, %4, %5
    │            │                         │
    └────────────┴── GEMM → BiasAdd → ReLU
```

## Architecture

```text
User Model / nn API
        │
        ▼
Frontend Tensor + Ops
        │
        ▼
Graph Capture
        │
        ▼
Graph / Node / Value
        │
        ▼
Graph → IR
        │
        ▼
PassManager
        │
        ├─ CanonicalizePass      [mock]
        └─ FusionPass            [detect only]
        │
        ▼
CUDA Lowering                    [mock]
        │
        ▼
CUDA Backend / Codegen           [mock]
        │
        ▼
Runtime                          [mock]
```

Cross-cutting observation lives in `diagnostics/` and `profiler/`.
Experiments use the framework rather than becoming the framework itself.

## Package layout

```text
aicf/
├─ nn/                    # user-facing Module / Sequential / layers
│  ├─ module.py
│  ├─ containers.py
│  └─ layers.py
├─ frontend/              # symbolic Tensor/Parameter and primitive ops
├─ graph/                 # Graph / Node / Value / capture builder
├─ ir/                    # compiler IR
├─ compiler/              # analyses, passes, PassManager
├─ lowering/              # high-level IR -> target representation
├─ backend/cuda/          # CUDA target/codegen placeholders
├─ runtime/               # executable/runtime placeholders
├─ diagnostics/           # graph/IR/decision observability
└─ profiler/              # timing/NVTX/NCU placeholders

experiments/
├─ capture/
├─ fusion/
├─ lowering/
└─ hardware/
```

## Current implemented slice

```text
nn.Sequential
  ↓
n.Linear + nn.ReLU
  ↓
Module child/parameter registration
  ↓
Graph inputs vs parameters
  ↓
GEMM → BiasAdd → ReLU
  ↓
IR
  ↓
Fusion candidate detection
  ↓
Mock CUDA lowering/runtime
```

The following are intentionally incomplete and are intended to be implemented one at
a time while studying the corresponding framework/compiler concept:

- real Tensor storage and execution
- parameter initialization / state_dict
- use-def chains
- graph mutation APIs
- fusion legality and profitability
- IR rewrite
- shape/type inference system
- target-specific lowering
- CUDA code generation
- memory planning
- autograd and training state
- actual profiling integration

## Run

From the project root:

```powershell
python -m experiments.fusion.gemm_bias_relu
```

## Tests

```powershell
python -m pytest -q
```
