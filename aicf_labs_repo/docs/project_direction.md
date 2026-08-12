# Project Direction

AICF Labs exists to keep two optimization research questions independent while allowing their results to be connected later.

## Frontend Lab

Question: **what mathematical transformations remain possible across operators?**

The first mechanism is binary operator marking:

```text
Mask vocabulary
→ per-operator mask
→ mask propagation
→ candidate screening
→ detailed legality / semantic validation
```

The mask is intentionally a cheap first filter. A surviving property means the corresponding optimization possibility has not yet been eliminated; it is not final proof of fusion legality.

## Backend CUDA Lab

Question: **how is an already-defined mathematical CUDA kernel actually represented and executed on the GPU?**

```text
fixed CUDA kernel
→ CUDA source observation
→ AST observation
→ PTX observation
→ SASS observation
→ runtime/profiler observation
→ execution hypothesis
```

The first output is observation, not an automatic rewrite. Optimized kernels may later be introduced as explicit comparison candidates.

## Connection

The two labs share workload and experiment contracts, not an obligatory code-generation pipeline.

```text
Frontend decision ─┐
                   ├→ Workload / Experiment Record
Backend observation┘
```

The original AICF v0.20 codegen path remains valuable as a proven end-to-end baseline, but its generator/runtime implementation lives under `legacy_v020/` and is not the architecture of the new repository.
