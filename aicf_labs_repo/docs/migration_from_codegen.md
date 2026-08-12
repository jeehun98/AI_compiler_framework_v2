# Migration from the v0.20 codegen-centered project

The v0.20 vertical slice remains the baseline, but its components move to new ownership boundaries.

| Previous area | New destination | Meaning |
|---|---|---|
| `nn/`, `frontend/`, `graph/`, `ir/`, fusion analysis/pass | `frontend_lab/` | high-level model/graph reasoning |
| fused IR → GEMM M/N/K extraction | frontend adapter + `WorkloadSpec` | computation contract extraction |
| CUDA problem/tile/schedule/mapping plan | `codegen_prototype/` | generated-naive implementation planning |
| CUDA source emitter | `codegen_prototype/` | baseline implementation generator |
| NVRTC | `backend_cuda/compile` (future move) | CUDA compilation infrastructure |
| CUDA Driver API | `backend_cuda/driver` (future move) | device/module infrastructure |
| device bindings / launch runtime | `backend_cuda/runtime` (future move) | backend execution infrastructure |
| numerical comparison | `backend_cuda/validation` + contracts | implementation-independent validation result |
| diagnostics | experiment/artifact records | reproducible observations |

## Migration rule

Do not move all legacy files at once. New contracts are introduced first, then legacy capabilities are wrapped behind adapters, and only then physically moved.

The `legacy_v020/` directory in this repository contains selected source snapshots for migration/reference. It is deliberately not importable as the new core package.
