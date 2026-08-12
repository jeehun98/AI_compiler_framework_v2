# Repository Layout

```text
aicf_labs_repo/
├─ src/aicf_labs/
│  ├─ contracts/
│  ├─ frontend_lab/
│  │  ├─ marking/
│  │  ├─ analysis/
│  │  └─ adapters/          # legacy bridge only
│  └─ backend_cuda/
│     ├─ kernels/           # fixed implementations to observe
│     ├─ observation/       # source/AST/PTX/SASS analyzers
│     ├─ tools/             # clang/nvcc/cuobjdump/ncu adapters
│     └─ implementations/   # implementation identities/catalog
├─ experiments/
│  ├─ frontend/
│  ├─ backend/
│  └─ connected/
├─ baselines/
├─ legacy_v020/             # old codegen/runtime migration source
├─ docs/
├─ scripts/
└─ tests/
```

The absence of a central `codegen_prototype/` package is intentional. Code generation is one possible backend implementation source, not the mandatory center of AICF Labs.
