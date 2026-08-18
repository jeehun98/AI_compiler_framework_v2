"""Value objects that connect frontend plans to backend execution evidence."""

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
import re


_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _validate_stable_id(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not _STABLE_ID.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a stable ID containing only letters, numbers, "
            "'.', '_', ':', or '-'"
        )


def _validate_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_text_tuple(
    values: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = True,
    stable_ids: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    for value in values:
        if stable_ids:
            _validate_stable_id(value, field_name)
        else:
            _validate_nonempty_text(value, field_name)


def _validate_typed_tuple(
    values: tuple[object, ...],
    expected_type: type,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if not all(isinstance(value, expected_type) for value in values):
        raise TypeError(
            f"{field_name} must contain only {expected_type.__name__} objects"
        )


class DecisionKind(Enum):
    """The logical transformation represented by an optimization decision."""

    SEMANTIC_FUSION = "semantic_fusion"
    DECOMPOSITION = "decomposition"
    LAYOUT_TRANSFORM = "layout_transform"
    OTHER = "other"


class BindingStatus(Enum):
    """Whether a planned unit has a usable backend implementation selection."""

    UNBOUND = "unbound"
    SELECTED = "selected"
    UNAVAILABLE = "unavailable"


class EvidenceSource(Enum):
    """Origin of an observed execution fact."""

    RUNTIME_TRACE = "runtime_trace"
    PROFILER = "profiler"
    BINARY_SASS = "binary_sass"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"


class ValidationStatus(Enum):
    """Observed validation outcome, kept separate from plan verification."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    UNKNOWN = "unknown"


class LatencyUnit(Enum):
    """Unit attached to an observed latency value."""

    NANOSECONDS = "ns"
    MICROSECONDS = "us"
    MILLISECONDS = "ms"
    SECONDS = "s"


@dataclass(frozen=True)
class Attribute:
    """A small immutable extension point for environment or configuration data."""

    key: str
    value: str

    def __post_init__(self) -> None:
        _validate_nonempty_text(self.key, "Attribute.key")
        _validate_nonempty_text(self.value, "Attribute.value")


def _validate_attributes(
    attributes: tuple[Attribute, ...], field_name: str
) -> None:
    _validate_typed_tuple(attributes, Attribute, field_name)
    keys = [attribute.key for attribute in attributes]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} contains duplicate keys")


@dataclass(frozen=True)
class OptimizationDecision:
    """Why logical computations were transformed into planned execution units."""

    id: str
    kind: DecisionKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    semantic_preservation: tuple[str, ...]
    expected_effects: tuple[str, ...]
    references: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        _validate_stable_id(self.id, "OptimizationDecision.id")
        if not isinstance(self.kind, DecisionKind):
            raise TypeError("OptimizationDecision.kind must be a DecisionKind")
        _validate_text_tuple(
            self.inputs,
            "OptimizationDecision.inputs",
            allow_empty=False,
            stable_ids=True,
        )
        _validate_text_tuple(
            self.outputs,
            "OptimizationDecision.outputs",
            allow_empty=False,
            stable_ids=True,
        )
        _validate_text_tuple(
            self.preconditions,
            "OptimizationDecision.preconditions",
            allow_empty=False,
        )
        _validate_text_tuple(
            self.semantic_preservation,
            "OptimizationDecision.semantic_preservation",
            allow_empty=False,
        )
        _validate_text_tuple(
            self.expected_effects,
            "OptimizationDecision.expected_effects",
            allow_empty=False,
        )
        _validate_attributes(self.references, "OptimizationDecision.references")


@dataclass(frozen=True)
class ValueSpec:
    """A planned input/output reference with optional shape, dtype, and layout."""

    id: str
    shape: tuple[int | str, ...] | None = None
    dtype: str | None = None
    layout: str | None = None

    def __post_init__(self) -> None:
        _validate_stable_id(self.id, "ValueSpec.id")
        if self.shape is not None:
            if not isinstance(self.shape, tuple):
                raise TypeError("ValueSpec.shape must be a tuple or None")
            for dimension in self.shape:
                if isinstance(dimension, bool) or not isinstance(
                    dimension, (int, str)
                ):
                    raise TypeError(
                        "ValueSpec.shape dimensions must be integers or symbols"
                    )
                if isinstance(dimension, int) and dimension < 0:
                    raise ValueError(
                        "ValueSpec.shape integer dimensions must be non-negative"
                    )
                if isinstance(dimension, str) and not dimension.strip():
                    raise ValueError("ValueSpec.shape symbols must not be empty")
        if self.dtype is not None:
            _validate_nonempty_text(self.dtype, "ValueSpec.dtype")
        if self.layout is not None:
            _validate_nonempty_text(self.layout, "ValueSpec.layout")


@dataclass(frozen=True)
class PlannedExecutionUnit:
    """A logical backend work unit; it is not proof of a kernel launch."""

    id: str
    logical_operator_ids: tuple[str, ...]
    inputs: tuple[ValueSpec, ...]
    outputs: tuple[ValueSpec, ...]
    decision_ids: tuple[str, ...] = ()
    expected_kernel_launches: int | None = None
    implementation_binding_id: str | None = None

    def __post_init__(self) -> None:
        _validate_stable_id(self.id, "PlannedExecutionUnit.id")
        _validate_text_tuple(
            self.logical_operator_ids,
            "PlannedExecutionUnit.logical_operator_ids",
            allow_empty=False,
            stable_ids=True,
        )
        _validate_typed_tuple(self.inputs, ValueSpec, "PlannedExecutionUnit.inputs")
        _validate_typed_tuple(
            self.outputs,
            ValueSpec,
            "PlannedExecutionUnit.outputs",
            allow_empty=False,
        )
        _validate_text_tuple(
            self.decision_ids,
            "PlannedExecutionUnit.decision_ids",
            stable_ids=True,
        )
        if self.expected_kernel_launches is not None:
            if isinstance(self.expected_kernel_launches, bool) or not isinstance(
                self.expected_kernel_launches, int
            ):
                raise TypeError(
                    "PlannedExecutionUnit.expected_kernel_launches must be an "
                    "integer or None"
                )
            if self.expected_kernel_launches < 0:
                raise ValueError(
                    "PlannedExecutionUnit.expected_kernel_launches must be "
                    "non-negative"
                )
        if self.implementation_binding_id is not None:
            _validate_stable_id(
                self.implementation_binding_id,
                "PlannedExecutionUnit.implementation_binding_id",
            )


@dataclass(frozen=True)
class ExecutionPlan:
    """An ordered collection of planned execution units."""

    id: str
    units: tuple[PlannedExecutionUnit, ...]

    def __post_init__(self) -> None:
        _validate_stable_id(self.id, "ExecutionPlan.id")
        _validate_typed_tuple(
            self.units,
            PlannedExecutionUnit,
            "ExecutionPlan.units",
            allow_empty=False,
        )


@dataclass(frozen=True)
class ImplementationBinding:
    """A lowering record that binds one planned unit to an implementation ref."""

    id: str
    unit_id: str
    backend: str
    status: BindingStatus
    target: str | None = None
    implementation_ref: str | None = None
    selection_reason: str | None = None
    configuration: tuple[Attribute, ...] = ()

    def __post_init__(self) -> None:
        _validate_stable_id(self.id, "ImplementationBinding.id")
        _validate_stable_id(self.unit_id, "ImplementationBinding.unit_id")
        _validate_nonempty_text(self.backend, "ImplementationBinding.backend")
        if not isinstance(self.status, BindingStatus):
            raise TypeError("ImplementationBinding.status must be a BindingStatus")
        if self.target is not None:
            _validate_nonempty_text(self.target, "ImplementationBinding.target")
        if self.implementation_ref is not None:
            _validate_stable_id(
                self.implementation_ref,
                "ImplementationBinding.implementation_ref",
            )
        if self.selection_reason is not None:
            _validate_nonempty_text(
                self.selection_reason,
                "ImplementationBinding.selection_reason",
            )
        _validate_attributes(
            self.configuration,
            "ImplementationBinding.configuration",
        )

        if self.status is BindingStatus.SELECTED:
            if self.implementation_ref is None:
                raise ValueError(
                    "A selected ImplementationBinding requires implementation_ref"
                )
            if self.selection_reason is None:
                raise ValueError(
                    "A selected ImplementationBinding requires selection_reason"
                )
        elif self.status is BindingStatus.UNBOUND and self.implementation_ref is not None:
            raise ValueError(
                "An unbound ImplementationBinding cannot name an implementation"
            )


@dataclass(frozen=True)
class ArtifactReference:
    """A repository-relative artifact path with explicit provenance."""

    path: str
    source: EvidenceSource

    def __post_init__(self) -> None:
        _validate_nonempty_text(self.path, "ArtifactReference.path")
        if not isinstance(self.source, EvidenceSource):
            raise TypeError("ArtifactReference.source must be an EvidenceSource")
        if "\\" in self.path:
            raise ValueError(
                "ArtifactReference.path must use repository-relative POSIX separators"
            )
        posix_path = PurePosixPath(self.path)
        windows_path = PureWindowsPath(self.path)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError("ArtifactReference.path must be repository-relative")
        if ".." in posix_path.parts:
            raise ValueError("ArtifactReference.path cannot contain path traversal")
        if self.path.startswith("./"):
            raise ValueError("ArtifactReference.path cannot begin with './'")


@dataclass(frozen=True)
class ValidationResult:
    """Validation observed during execution, independent of plan comparison."""

    status: ValidationStatus
    details: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ValidationStatus):
            raise TypeError("ValidationResult.status must be a ValidationStatus")
        if self.details is not None:
            _validate_nonempty_text(self.details, "ValidationResult.details")


@dataclass(frozen=True)
class ExecutionEvidence:
    """Runtime or binary facts observed independently from frontend expectations."""

    id: str
    subject_id: str
    sources: tuple[EvidenceSource, ...]
    environment: tuple[Attribute, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    observed_kernel_launches: int | None = None
    observed_kernel_names: tuple[str, ...] | None = None
    observed_instruction_features: tuple[str, ...] | None = None
    latency: float | None = None
    latency_unit: LatencyUnit | None = None
    validation: ValidationResult | None = None

    def __post_init__(self) -> None:
        _validate_stable_id(self.id, "ExecutionEvidence.id")
        _validate_stable_id(self.subject_id, "ExecutionEvidence.subject_id")
        _validate_typed_tuple(
            self.sources,
            EvidenceSource,
            "ExecutionEvidence.sources",
            allow_empty=False,
        )
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("ExecutionEvidence.sources contains duplicates")
        _validate_attributes(self.environment, "ExecutionEvidence.environment")
        _validate_typed_tuple(
            self.artifacts,
            ArtifactReference,
            "ExecutionEvidence.artifacts",
        )
        for artifact in self.artifacts:
            if artifact.source not in self.sources:
                raise ValueError(
                    "ExecutionEvidence artifact source must be listed in sources"
                )

        if self.observed_kernel_launches is not None:
            if isinstance(self.observed_kernel_launches, bool) or not isinstance(
                self.observed_kernel_launches, int
            ):
                raise TypeError(
                    "ExecutionEvidence.observed_kernel_launches must be an "
                    "integer or None"
                )
            if self.observed_kernel_launches < 0:
                raise ValueError(
                    "ExecutionEvidence.observed_kernel_launches must be non-negative"
                )
        if self.observed_kernel_names is not None:
            _validate_text_tuple(
                self.observed_kernel_names,
                "ExecutionEvidence.observed_kernel_names",
            )
        if self.observed_instruction_features is not None:
            _validate_text_tuple(
                self.observed_instruction_features,
                "ExecutionEvidence.observed_instruction_features",
            )

        if self.latency is None and self.latency_unit is not None:
            raise ValueError("ExecutionEvidence.latency_unit requires latency")
        if self.latency is not None:
            if isinstance(self.latency, bool) or not isinstance(
                self.latency, (int, float)
            ):
                raise TypeError("ExecutionEvidence.latency must be numeric or None")
            if self.latency < 0:
                raise ValueError("ExecutionEvidence.latency must be non-negative")
            if not isinstance(self.latency_unit, LatencyUnit):
                raise ValueError(
                    "ExecutionEvidence.latency requires a LatencyUnit"
                )
        if self.validation is not None and not isinstance(
            self.validation, ValidationResult
        ):
            raise TypeError(
                "ExecutionEvidence.validation must be ValidationResult or None"
            )


__all__ = (
    "ArtifactReference",
    "Attribute",
    "BindingStatus",
    "DecisionKind",
    "EvidenceSource",
    "ExecutionEvidence",
    "ExecutionPlan",
    "ImplementationBinding",
    "LatencyUnit",
    "OptimizationDecision",
    "PlannedExecutionUnit",
    "ValidationResult",
    "ValidationStatus",
    "ValueSpec",
)
