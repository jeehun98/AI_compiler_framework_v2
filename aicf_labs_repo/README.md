# AICF Labs

AICF Labs is a separate experimental repository built after the original AICF v0.20 vertical slice.

The repository deliberately has **two independent laboratories**.

```text
Frontend Lab
operator mathematical properties
→ binary marking
→ mask propagation / candidate screening
→ detailed transformation analysis

Backend CUDA Lab
fixed CUDA kernel
→ CUDA source / AST / PTX / SASS
→ runtime observation
→ optimization hypothesis
```

They meet only through shared contracts such as `WorkloadSpec`, `ImplementationSpec`, and `ExperimentRecord`.

## Frontend principle

The first frontend primitive is not a CUDA lowering rule. It is an operator-property vocabulary:

```text
bit 0 COMMUTATIVE
bit 1 ASSOCIATIVE
bit 2 ELEMENTWISE
...
```

Each operator has its own binary mark. A cheap mask propagation pass can intersect properties across an operator region and preserve or eliminate optimization possibilities. A surviving bit is a **screening signal**, not a proof of final legality.

## Backend principle

The backend starts from an already defined CUDA kernel. It observes how the same mathematical computation appears at different execution layers:

```text
CUDA source
→ CUDA AST
→ PTX
→ SASS
→ runtime / profiler counters
```

Optimization comes after observation. The repository therefore separates artifact collection tools from artifact analyzers.

## v0.20 baseline

The original end-to-end generated `GEMM + Bias + ReLU` CUDA kernel is preserved as a frozen baseline kernel under:

```text
src/aicf_labs/backend_cuda/kernels/sources/
```

The old codegen/lowering/runtime implementation remains under `legacy_v020/` only as migration reference. It is not the center of this repository.

## Install / test

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

## First experiments

Frontend mask propagation:

```powershell
python -m experiments.frontend.operator_mask_walk
```

Backend CUDA-source observation:

```powershell
python -m experiments.backend.observe_generated_naive
```

If AST/PTX/SASS text artifacts have already been collected:

```powershell
python -m experiments.backend.observe_generated_naive --ast out.ast --ptx out.ptx --sass out.sass
```

See `docs/project_direction.md`, `docs/architecture.md`, `docs/frontend_marking.md`, and `docs/backend_observation.md`.
