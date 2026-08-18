"""Concrete implementation descriptions separated from operator semantics."""

from dataclasses import dataclass

from .masks import HardwareMask, Observation


@dataclass(frozen=True)
class SassEvidence:
    """One inspected SASS fact, including explicit absence when inspected."""

    file: str
    observation: str
    instruction: str
    status: Observation

    def __post_init__(self) -> None:
        if not self.file:
            raise ValueError("SassEvidence.file must not be empty")
        if not self.observation:
            raise ValueError("SassEvidence.observation must not be empty")
        if not self.instruction:
            raise ValueError("SassEvidence.instruction must not be empty")
        if not isinstance(self.status, Observation):
            raise TypeError("SassEvidence.status must be an Observation")


@dataclass(frozen=True)
class Implementation:
    """A particular CUDA implementation that may realize an operator."""

    name: str
    source_file: str
    kernel_name: str
    input_dtype: str
    output_dtype: str
    hardware: HardwareMask
    sass_evidence: tuple[SassEvidence, ...] = ()

    def __post_init__(self) -> None:
        text_fields = {
            "name": self.name,
            "source_file": self.source_file,
            "kernel_name": self.kernel_name,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
        }
        for field_name, value in text_fields.items():
            if not value:
                raise ValueError(f"Implementation.{field_name} must not be empty")
        if not isinstance(self.hardware, HardwareMask):
            raise TypeError("Implementation.hardware must be a HardwareMask")
        if not isinstance(self.sass_evidence, tuple):
            raise TypeError("Implementation.sass_evidence must be a tuple")
        if not all(isinstance(item, SassEvidence) for item in self.sass_evidence):
            raise TypeError(
                "Implementation.sass_evidence must contain only SassEvidence"
            )
