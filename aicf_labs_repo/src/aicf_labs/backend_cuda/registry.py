from __future__ import annotations

from .implementations import CUDAImplementation, GeneratedNaive


class ImplementationRegistry:
    def __init__(self) -> None:
        self._implementations: dict[str, CUDAImplementation] = {}

    def register(self, implementation: CUDAImplementation) -> None:
        name = implementation.spec.name
        if name in self._implementations:
            raise ValueError(f"implementation already registered: {name}")
        self._implementations[name] = implementation

    def get(self, name: str) -> CUDAImplementation:
        try:
            return self._implementations[name]
        except KeyError as exc:
            raise KeyError(f"unknown implementation: {name}") from exc

    def candidates(self, workload) -> tuple[CUDAImplementation, ...]:
        return tuple(
            implementation
            for implementation in self._implementations.values()
            if implementation.supports(workload)
        )


def default_registry() -> ImplementationRegistry:
    registry = ImplementationRegistry()
    registry.register(GeneratedNaive())
    return registry
