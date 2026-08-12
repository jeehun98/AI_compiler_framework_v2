from __future__ import annotations

from abc import ABC, abstractmethod

from ...contracts import ImplementationSpec, WorkloadSpec


class CUDAImplementation(ABC):
    spec: ImplementationSpec

    @abstractmethod
    def supports(self, workload: WorkloadSpec) -> bool:
        raise NotImplementedError
