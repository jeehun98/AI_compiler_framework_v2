# AICF Labs

AICF Labs is the restructured project that follows the v0.20 end-to-end compiler/codegen milestone.

The project intentionally separates two optimization questions:

- **Frontend Lab** — what computation should be performed?
- **Backend CUDA Lab** — how should a fixed computation be executed on the GPU?

The two labs share **contracts**, not implementation code.

```text
Model / Graph
    ↓
Frontend Lab
    ↓
WorkloadSpec
    ↓
Backend CUDA Lab
    ↓
Implementation + Artifact + Measurement
    ↓
ExperimentRecord
```

The previous v0.20 codegen path is retained as a **generated_naive baseline implementation**, not as the center of the repository.

## Initial milestone

The first milestone is architectural rather than performance-oriented:

1. freeze the v0.20 baseline,
2. introduce WorkloadSpec / ImplementationSpec / ExperimentRecord,
3. adapt the old fused GEMM+Bias+ReLU path into `generated_naive`,
4. allow frontend-only, backend-only, and connected experiments to run independently.

## Install

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Layout

See `docs/architecture.md` and `docs/migration_from_codegen.md`.
