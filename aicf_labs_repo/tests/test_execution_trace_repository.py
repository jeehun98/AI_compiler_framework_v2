"""Repository integration tests for loose trace artifact references."""

from pathlib import Path
import unittest

from aicf_labs import (
    ArtifactReference,
    Attribute,
    BindingStatus,
    EvidenceSource,
    ExecutionEvidence,
    ExecutionPlan,
    ImplementationBinding,
    PlannedExecutionUnit,
    TraceRecord,
    ValueSpec,
    VerificationStatus,
    compare_plan_to_evidence,
)


class RepositoryExecutionTraceTests(unittest.TestCase):
    def test_existing_add_artifacts_are_loosely_referenced(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        unit = PlannedExecutionUnit(
            id="plan.add.fp32.unit0",
            logical_operator_ids=("add.0",),
            inputs=(ValueSpec("value.a"), ValueSpec("value.b")),
            outputs=(ValueSpec("value.y"),),
            expected_kernel_launches=1,
            implementation_binding_id="lowering.add.cuda.sm86",
        )
        plan = ExecutionPlan(id="plan.add.fp32", units=(unit,))
        binding = ImplementationBinding(
            id="lowering.add.cuda.sm86",
            unit_id=unit.id,
            backend="cuda",
            target="sm_86",
            implementation_ref="operator:add",
            selection_reason="Reference the existing scalar add experiment.",
            status=BindingStatus.SELECTED,
        )
        evidence = ExecutionEvidence(
            id="evidence.add.sm86.repository",
            subject_id=binding.id,
            sources=(EvidenceSource.PROFILER, EvidenceSource.BINARY_SASS),
            environment=(Attribute("gpu_arch", "sm_86"),),
            artifacts=(
                ArtifactReference(
                    "operators/add/runtime/add.ncu-rep",
                    EvidenceSource.PROFILER,
                ),
                ArtifactReference(
                    "operators/add/artifacts/add.sass",
                    EvidenceSource.BINARY_SASS,
                ),
            ),
            observed_kernel_launches=None,
            observed_kernel_names=("add_fp32",),
            observed_instruction_features=("FADD",),
        )
        trace = TraceRecord(
            logical_operator_ids=("add.0",),
            plans=(plan,),
            bindings=(binding,),
            evidence=(evidence,),
        )

        for artifact in trace.evidence[0].artifacts:
            self.assertTrue((repository_root / artifact.path).is_file())
        sass_text = (
            repository_root / "operators/add/artifacts/add.sass"
        ).read_text(encoding="utf-8-sig")
        self.assertIn("Function", sass_text)
        self.assertIn("add_fp32", sass_text)
        self.assertIn("FADD R9, R4, R3", sass_text)

        comparison = compare_plan_to_evidence(plan, evidence)
        self.assertIs(
            comparison.kernel_launches.status,
            VerificationStatus.UNOBSERVED,
        )


if __name__ == "__main__":
    unittest.main()
