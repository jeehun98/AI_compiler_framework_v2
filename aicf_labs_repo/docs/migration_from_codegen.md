# Migration from the original AICF codegen project

The original project proved a complete vertical slice:

```text
Model → Graph → IR → Fusion → CUDA lowering → CUDA source → NVRTC → PTX → Driver → GPU execution → validation
```

That path is preserved, but it no longer defines the architecture of AICF Labs.

## What was extracted

- the generated naive fused GEMM/Bias/ReLU CUDA kernel as a fixed backend observation baseline;
- the v0.20 NVRTC/Driver/device-memory/runtime files under `legacy_v020/` as migration references;
- shared workload, environment, validation, artifact, and experiment contracts.

## What changed

Frontend research now starts from operator mathematical-property marking, not code generation.

Backend research now starts from an existing CUDA kernel and its artifacts, not from a requirement that every workload pass through the old generator.

The old code generator may still be used as one controlled implementation source, but it is not a mandatory path.
