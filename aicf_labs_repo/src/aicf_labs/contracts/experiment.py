from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .artifact import ArtifactSet
from .environment import EnvironmentSpec
from .implementation import ImplementationSpec
from .observation import RuntimeObservation, StaticObservation
from .transformation import TransformationRecord
from .workload import WorkloadSpec


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    workload: WorkloadSpec
    implementation: ImplementationSpec
    transformation: TransformationRecord | None = None
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    artifacts: ArtifactSet = field(default_factory=ArtifactSet)
    static: StaticObservation = field(default_factory=StaticObservation)
    runtime: RuntimeObservation = field(default_factory=RuntimeObservation)
    validation: ValidationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
