# Architecture

## Purpose

AICF Labs separates two questions that should not be forced into one compiler/codegen pipeline.

```text
Frontend: what transformations are mathematically/structurally possible?
Backend: how is a fixed mathematical CUDA computation actually executed?
```

## Repository boundary

```text
src/aicf_labs/
├─ contracts/          shared experiment/workload records
├─ frontend_lab/
│  ├─ marking/         bit vocabulary + operator-specific masks
│  └─ analysis/        propagation and candidate screening
├─ backend_cuda/
│  ├─ kernels/         fixed CUDA kernels to observe
│  ├─ observation/     CUDA/AST/PTX/SASS parsers
│  ├─ tools/           artifact-producing external tool adapters
│  └─ implementations/ backend implementation identities
└─ codegen_prototype/  compatibility/reference only
```

## Frontend flow

```text
Mask Definition
→ Operator Mark Definition
→ mask propagation
→ candidate region
→ detailed legality / semantic validation (future)
→ transformation record
```

## Backend flow

```text
KernelSpec
→ CUDA source observation
→ AST artifact observation
→ PTX artifact observation
→ SASS artifact observation
→ runtime observation
→ comparison / hypothesis (future)
```

No backend optimization is implied simply because an artifact was observed. Observation is the first-class output.
