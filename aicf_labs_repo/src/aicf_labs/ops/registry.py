from __future__ import annotations

from .base import OperatorDefinition

_REGISTRY: dict[str, OperatorDefinition] = {}


def register(definition: OperatorDefinition) -> OperatorDefinition:
    if definition.name in _REGISTRY:
        raise ValueError(f"operator already registered: {definition.name}")
    _REGISTRY[definition.name] = definition
    return definition


def get(name: str) -> OperatorDefinition:
    return _REGISTRY[name]


def all_definitions() -> tuple[OperatorDefinition, ...]:
    return tuple(_REGISTRY.values())
