from __future__ import annotations

import re

from .model import LayerObservation


def observe_cuda_source(source: str) -> LayerObservation:
    return LayerObservation(
        layer="cuda_source",
        metrics={
            "global_kernels": len(re.findall(r"\b__global__\b", source)),
            "for_loops": len(re.findall(r"\bfor\s*\(", source)),
            "if_statements": len(re.findall(r"\bif\s*\(", source)),
            "thread_idx_refs": source.count("threadIdx."),
            "block_idx_refs": source.count("blockIdx."),
            "shared_declarations": len(re.findall(r"\b__shared__\b", source)),
            "sync_threads": source.count("__syncthreads"),
            "fma_like_expressions": len(re.findall(r"\+=\s*[^;]+\*[^;]+;", source)),
        },
    )
