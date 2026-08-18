"""Opt-in end-to-end test for a real add.exe execution trace.

The default test suite remains portable. Set ``AICF_RUN_CUDA_E2E=1`` to run
the prebuilt CUDA executable and print each plan/lowering/evidence artifact.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import unittest

from aicf_labs import (
    ArtifactReference,
    Attribute,
    BindingStatus,
    EvidenceSource,
    ExecutionEvidence,
    ExecutionPlan,
    ImplementationBinding,
    LatencyUnit,
    PlanEvidenceComparison,
    PlannedExecutionUnit,
    TraceRecord,
    ValidationResult,
    ValidationStatus,
    ValueSpec,
    VerificationStatus,
    compare_plan_to_evidence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CUDA_E2E_ENABLED = os.environ.get("AICF_RUN_CUDA_E2E") == "1"

_GPU_PATTERN = re.compile(
    r"^GPU: (?P<name>.+) \((?P<arch>sm_\d+)\)$",
    re.MULTILINE,
)
_ELEMENTS_PATTERN = re.compile(r"^Elements: (?P<count>\d+) FP32$", re.MULTILINE)
_LATENCY_PATTERN = re.compile(
    r"^Elementwise add: (?P<latency>\d+(?:\.\d+)?) us$",
    re.MULTILINE,
)
_BANDWIDTH_PATTERN = re.compile(
    r"^Effective bandwidth: (?P<bandwidth>\d+(?:\.\d+)?) GB/s$",
    re.MULTILINE,
)
_VALIDATION_PATTERN = re.compile(
    r"^Validation mismatches: (?P<mismatches>\d+) / (?P<count>\d+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class AddCudaRun:
    """Artifacts collected while one real add executable is traced."""

    command: tuple[str, ...]
    stdout: str
    gpu_name: str
    gpu_arch: str
    elements: int
    validation_elements: int
    mismatches: int
    latency_us: float
    bandwidth_gbps: float
    sass_path: Path
    profiler_path: Path
    plan: ExecutionPlan
    binding: ImplementationBinding
    direct_evidence: ExecutionEvidence
    profiler_evidence: ExecutionEvidence
    trace: TraceRecord
    comparison: PlanEvidenceComparison

    def render(self) -> str:
        """Return a readable stage-by-stage view of all collected artifacts."""

        unit = self.plan.units[0]
        artifact_lines = tuple(
            f"    - {artifact.source.value}: {artifact.path}"
            for evidence in self.trace.evidence
            for artifact in evidence.artifacts
        )
        return "\n".join(
            (
                "=== add CUDA trace end-to-end artifacts ===",
                "[1/6] frontend plan",
                f"  plan_id: {self.plan.id}",
                f"  unit_id: {unit.id}",
                f"  logical_operators: {', '.join(unit.logical_operator_ids)}",
                f"  expected_kernel_launches: {unit.expected_kernel_launches}",
                "[2/6] lowering",
                f"  binding_id: {self.binding.id}",
                f"  backend/target: {self.binding.backend}/{self.binding.target}",
                f"  implementation_ref: {self.binding.implementation_ref}",
                "[3/6] direct CUDA process",
                f"  command: {subprocess.list2cmdline(self.command)}",
                f"  gpu: {self.gpu_name} ({self.gpu_arch})",
                f"  elements: {self.elements}",
                f"  latency_us: {self.latency_us}",
                f"  bandwidth_gbps: {self.bandwidth_gbps}",
                f"  validation_mismatches: {self.mismatches} / "
                f"{self.validation_elements}",
                "  stdout:",
                *(f"    {line}" for line in self.stdout.rstrip().splitlines()),
                "[4/6] binary and repository artifacts",
                *artifact_lines,
                "  observed_kernel_names: add_fp32",
                "  observed_instruction_features: FADD",
                "  profiler_note: pre-existing report; this test does not regenerate it",
                "[5/6] execution evidence",
                f"  direct_evidence_id: {self.direct_evidence.id}",
                "  sources: "
                + ", ".join(source.value for source in self.direct_evidence.sources),
                "  observed_kernel_launches: None (stdout does not expose it)",
                f"  validation: {self.direct_evidence.validation.status.value}",
                "[6/6] expected versus observed",
                f"  status: {self.comparison.kernel_launches.status.value}",
                f"  expected: {self.comparison.kernel_launches.expected}",
                f"  observed: {self.comparison.kernel_launches.observed}",
            )
        )


def _required_match(pattern: re.Pattern[str], stdout: str, label: str) -> re.Match[str]:
    match = pattern.search(stdout)
    if match is None:
        raise AssertionError(f"add.exe output is missing {label!r}:\n{stdout}")
    return match


def run_add_cuda_trace(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    elements: int = 1024,
    iterations: int = 1,
    validation_elements: int = 128,
    seed: int = 12345,
) -> AddCudaRun:
    """Run add.exe once and connect its observed outputs to trace objects."""

    executable_path = repository_root / "operators" / "add" / "build" / "add.exe"
    sass_path = repository_root / "operators" / "add" / "artifacts" / "add.sass"
    profiler_path = repository_root / "operators" / "add" / "runtime" / "add.ncu-rep"
    for required_path in (executable_path, sass_path, profiler_path):
        if not required_path.is_file():
            raise AssertionError(f"required CUDA artifact is missing: {required_path}")

    unit = PlannedExecutionUnit(
        id="plan.add.fp32.e2e.unit0",
        logical_operator_ids=("add.e2e",),
        inputs=(
            ValueSpec("value.add.e2e.a", shape=(elements,), dtype="fp32"),
            ValueSpec("value.add.e2e.b", shape=(elements,), dtype="fp32"),
        ),
        outputs=(
            ValueSpec("value.add.e2e.output", shape=(elements,), dtype="fp32"),
        ),
        expected_kernel_launches=1,
        implementation_binding_id="lowering.add.cuda.e2e",
    )
    plan = ExecutionPlan(id="plan.add.fp32.e2e", units=(unit,))

    command = (
        str(executable_path),
        str(elements),
        str(iterations),
        str(validation_elements),
        str(seed),
    )
    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "add.exe failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    gpu_match = _required_match(_GPU_PATTERN, completed.stdout, "GPU")
    elements_match = _required_match(_ELEMENTS_PATTERN, completed.stdout, "Elements")
    latency_match = _required_match(
        _LATENCY_PATTERN,
        completed.stdout,
        "Elementwise add latency",
    )
    bandwidth_match = _required_match(
        _BANDWIDTH_PATTERN,
        completed.stdout,
        "Effective bandwidth",
    )
    validation_match = _required_match(
        _VALIDATION_PATTERN,
        completed.stdout,
        "Validation mismatches",
    )
    if "Validation: PASSED" not in completed.stdout:
        raise AssertionError(f"add.exe validation did not pass:\n{completed.stdout}")

    observed_elements = int(elements_match.group("count"))
    observed_validation_elements = int(validation_match.group("count"))
    mismatches = int(validation_match.group("mismatches"))
    if observed_elements != elements:
        raise AssertionError(
            f"add.exe reported {observed_elements} elements, expected {elements}"
        )
    if observed_validation_elements != validation_elements:
        raise AssertionError(
            "add.exe reported "
            f"{observed_validation_elements} validation elements, expected "
            f"{validation_elements}"
        )

    sass_text = sass_path.read_text(encoding="utf-8")
    if "add_fp32" not in sass_text or not re.search(r"\bFADD\b", sass_text):
        raise AssertionError(
            f"SASS artifact does not contain add_fp32 and FADD: {sass_path}"
        )

    gpu_name = gpu_match.group("name")
    gpu_arch = gpu_match.group("arch")
    binding = ImplementationBinding(
        id="lowering.add.cuda.e2e",
        unit_id=unit.id,
        backend="cuda",
        target=gpu_arch,
        implementation_ref="operator:add",
        selection_reason="Run the repository's prebuilt scalar FP32 add executable.",
        status=BindingStatus.SELECTED,
        configuration=(
            Attribute("elements", str(elements)),
            Attribute("iterations", str(iterations)),
            Attribute("validation_elements", str(validation_elements)),
            Attribute("seed", str(seed)),
        ),
    )
    direct_evidence = ExecutionEvidence(
        id="evidence.add.cuda.e2e.direct",
        subject_id=binding.id,
        sources=(
            EvidenceSource.BENCHMARK,
            EvidenceSource.VALIDATION,
            EvidenceSource.BINARY_SASS,
        ),
        environment=(
            Attribute("gpu_name", gpu_name),
            Attribute("gpu_arch", gpu_arch),
        ),
        artifacts=(
            ArtifactReference(
                "operators/add/artifacts/add.sass",
                EvidenceSource.BINARY_SASS,
            ),
        ),
        observed_kernel_launches=None,
        observed_kernel_names=("add_fp32",),
        observed_instruction_features=("FADD",),
        latency=float(latency_match.group("latency")),
        latency_unit=LatencyUnit.MICROSECONDS,
        validation=ValidationResult(
            ValidationStatus.PASSED,
            f"add.exe reported {mismatches} mismatches across "
            f"{observed_validation_elements} validation elements.",
        ),
    )
    profiler_evidence = ExecutionEvidence(
        id="evidence.add.cuda.repository_profiler",
        subject_id=binding.id,
        sources=(EvidenceSource.PROFILER,),
        artifacts=(
            ArtifactReference(
                "operators/add/runtime/add.ncu-rep",
                EvidenceSource.PROFILER,
            ),
        ),
    )
    trace = TraceRecord(
        logical_operator_ids=("add.e2e",),
        plans=(plan,),
        bindings=(binding,),
        evidence=(direct_evidence, profiler_evidence),
    )
    comparison = compare_plan_to_evidence(unit, direct_evidence)

    return AddCudaRun(
        command=command,
        stdout=completed.stdout,
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        elements=observed_elements,
        validation_elements=observed_validation_elements,
        mismatches=mismatches,
        latency_us=direct_evidence.latency,
        bandwidth_gbps=float(bandwidth_match.group("bandwidth")),
        sass_path=sass_path,
        profiler_path=profiler_path,
        plan=plan,
        binding=binding,
        direct_evidence=direct_evidence,
        profiler_evidence=profiler_evidence,
        trace=trace,
        comparison=comparison,
    )


@unittest.skipUnless(
    CUDA_E2E_ENABLED,
    "set AICF_RUN_CUDA_E2E=1 to run the prebuilt CUDA executable",
)
class AddCudaTraceEndToEndTests(unittest.TestCase):
    """Execute one real CUDA run and verify every collected stage artifact."""

    artifacts: AddCudaRun

    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = run_add_cuda_trace()
        print("\n" + cls.artifacts.render(), flush=True)

    def test_direct_process_output_and_validation(self) -> None:
        self.assertIn("Validation: PASSED", self.artifacts.stdout)
        self.assertEqual(self.artifacts.elements, 1024)
        self.assertEqual(self.artifacts.mismatches, 0)
        self.assertGreaterEqual(self.artifacts.latency_us, 0.0)
        self.assertGreaterEqual(self.artifacts.bandwidth_gbps, 0.0)

    def test_plan_lowering_and_evidence_references_are_valid(self) -> None:
        self.artifacts.trace.validate()
        self.assertEqual(
            self.artifacts.plan.units[0].implementation_binding_id,
            self.artifacts.binding.id,
        )
        self.assertEqual(
            self.artifacts.binding.implementation_ref,
            "operator:add",
        )
        self.assertEqual(
            self.artifacts.direct_evidence.subject_id,
            self.artifacts.binding.id,
        )

    def test_sass_and_profiler_artifacts_exist(self) -> None:
        self.assertTrue(self.artifacts.sass_path.is_file())
        self.assertTrue(self.artifacts.profiler_path.is_file())
        self.assertEqual(
            self.artifacts.direct_evidence.observed_kernel_names,
            ("add_fp32",),
        )
        self.assertEqual(
            self.artifacts.direct_evidence.observed_instruction_features,
            ("FADD",),
        )

    def test_unmeasured_launch_count_remains_unobserved(self) -> None:
        self.assertIsNone(
            self.artifacts.direct_evidence.observed_kernel_launches
        )
        self.assertIs(
            self.artifacts.comparison.kernel_launches.status,
            VerificationStatus.UNOBSERVED,
        )
        self.assertEqual(
            self.artifacts.comparison.kernel_launches.expected,
            1,
        )
        self.assertIsNone(
            self.artifacts.comparison.kernel_launches.observed
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
