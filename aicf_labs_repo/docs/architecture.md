# Architecture

## Core boundary

```text
Frontend Lab
  model / graph / analysis / pattern / transform
                  │
                  ▼
             WorkloadSpec
                  │
                  ▼
Backend CUDA Lab
  implementation / compile / artifact / profile / validate
                  │
                  ▼
           ExperimentRecord
```

### Frontend Lab

Owns model/graph semantics and transformation decisions. It must not require a CUDA implementation.

### Contracts

Owns stable, implementation-independent data structures shared by both sides.

### Backend CUDA Lab

Owns implementations of an already-defined workload. It must not require the original model or graph.

### Codegen Prototype

The old generated CUDA path is one backend implementation candidate. It is useful as:

- a correctness baseline,
- a controlled code-generation experiment,
- a CUDA/PTX/SASS learning artifact.

It is not the mandatory path for every workload.
