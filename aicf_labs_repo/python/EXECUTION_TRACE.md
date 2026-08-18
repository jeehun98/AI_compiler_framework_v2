# Execution trace model

`aicf_labs`의 trace 계층은 frontend의 논리적 계획, backend lowering, 실제 실행
관측을 연결하되 같은 사실로 합치지 않는다.

```text
logical operator IDs
  -> OptimizationDecision
  -> ExecutionPlan / PlannedExecutionUnit
  -> ImplementationBinding
  -> ExecutionEvidence
  -> PlanEvidenceComparison
```

## Responsibility boundary

Frontend는 `OptimizationDecision`에 어떤 논리 연산을 왜 변환했는지 기록하고,
`PlannedExecutionUnit`에 backend로 전달하려는 작업과 기대 kernel launch 수를
기록한다. 이 계획은 kernel이 실제로 실행되었다는 증거가 아니다.

Backend는 `ImplementationBinding`에 planned unit을 어떤 backend/target의 안정적
implementation reference로 lowering했는지 기록한다. 기존 Python
`Implementation`은 operator에 속한 구체 구현 설명이고, binding은 그 구현을
선택하는 별도 record다. 구현이 아직 없으면 `BindingStatus.UNBOUND`, 알려진
target에 사용할 수 없으면 `UNAVAILABLE`로 남길 수 있다.

Runtime probing은 `ExecutionEvidence`에 profiler, runtime trace, binary/SASS,
validation, benchmark에서 실제로 관찰한 사실만 기록한다. 관찰하지 않은 launch
count, latency, validation 결과는 `None`이며 실패를 뜻하지 않는다.

## Three different kinds of fusion

- **Semantic fusion**: `OptimizationDecision(kind=SEMANTIC_FUSION)`에 기록된 frontend
  계획이다.
- **Kernel fusion**: runtime trace/profiler가 보여 준 kernel launch 수와 이름의
  관측이다.
- **Instruction fusion**: binary/SASS에서 확인한 `FFMA`, `HFMA2`, `HMMA` 같은
  instruction feature다.

Semantic fusion unit 하나를 계획해도 단일 CUDA kernel 실행이나 fused instruction을
자동으로 의미하지 않는다. 세 사실은 서로 다른 record와 provenance를 가진다.

## Expected versus observed

`PlannedExecutionUnit.expected_kernel_launches`와
`ExecutionEvidence.observed_kernel_launches`는 별도 값이다.
`compare_plan_to_evidence()`는 현재 이 두 값만 비교하고 다음 상태를 반환한다.

- `CONFIRMED`: 기대값과 관측값이 일치
- `MISMATCH`: 관측됐지만 기대값과 불일치
- `UNOBSERVED`: 기대값은 있으나 관측값이 없음
- `NOT_APPLICABLE`: 계획에 기대값 자체가 정의되지 않음

Instruction feature나 memory traffic 비교는 아직 자동화하지 않는다.

## Relationship to operator knowledge

`operators/*/operator.json`, `knowledge/index.json`,
`knowledge/schemas/operator.schema.json`은 CUDA operator 의미, 구현, artifact의
canonical 지식 계층이다. Trace object는 이 파일을 복제하거나 schema를 확장하지
않는다. `implementation_ref="operator:add"` 같은 안정적 참조와
repository-relative `ArtifactReference`만 보유한다.

Artifact path는 객체 ID가 아니다. 절대 경로, Windows separator와 `..` traversal은
거부한다. `TraceRecord`는 메모리 안에서 logical/decision/plan unit/binding/evidence
ID 중복과 깨진 내부 참조를 검사하지만, 외부 implementation ref나 artifact 파일이
실제로 존재하는지는 강제하지 않는다.

## Minimal example

```python
from aicf_labs import (
    BindingStatus,
    EvidenceSource,
    ExecutionEvidence,
    ExecutionPlan,
    ImplementationBinding,
    PlannedExecutionUnit,
    TraceRecord,
    ValueSpec,
    compare_plan_to_evidence,
)

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
    selection_reason="Use the existing scalar FP32 add experiment.",
    status=BindingStatus.SELECTED,
)
evidence = ExecutionEvidence(
    id="evidence.add.sm86.run001",
    subject_id=binding.id,
    sources=(EvidenceSource.RUNTIME_TRACE,),
    observed_kernel_launches=1,
    observed_kernel_names=("add_fp32",),
)
trace = TraceRecord(
    logical_operator_ids=("add.0",),
    plans=(plan,),
    bindings=(binding,),
    evidence=(evidence,),
)

result = compare_plan_to_evidence(unit, evidence)
```

Semantic fusion을 계획했지만 verified fused backend가 없다면 여러 logical operator
ID를 한 unit에 두고 UNBOUND binding을 연결한다. Evidence는 만들지 않으며 비교
결과는 `UNOBSERVED`다. 기존 GEMM/add artifact를 fused implementation이라고
연결하지 않는다.

## Direct CUDA end-to-end test

`tests/test_add_cuda_trace_e2e.py`는 기존 prebuilt `operators/add/build/add.exe`를
작은 입력으로 직접 실행하는 opt-in 테스트다. 테스트는 다음 산출물을 순서대로
검증하고 console에 출력한다.

1. frontend `ExecutionPlan`과 `PlannedExecutionUnit`
2. `operator:add`를 선택한 CUDA `ImplementationBinding`
3. 실제 process command, stdout, GPU/architecture, latency, bandwidth, validation
4. `add.sass`의 `add_fp32`/`FADD`와 기존 `add.ncu-rep` 파일 참조
5. direct-run evidence와 별도의 pre-existing profiler evidence
6. 기대 launch 수와 관측 launch 수 비교

일반 실행 stdout에는 실제 kernel launch 횟수가 없으므로 테스트는 이를 추측하지
않고 `observed_kernel_launches=None`, 비교 결과 `UNOBSERVED`로 유지한다. 또한 기존
`add.ncu-rep`는 이 테스트 실행에서 새로 생성된 report라고 주장하지 않고 별도의
repository profiler evidence로만 연결한다.

Repository root의 PowerShell에서 다음과 같이 실행한다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\python).Path
$env:AICF_RUN_CUDA_E2E = "1"
python .\tests\test_add_cuda_trace_e2e.py -v
Remove-Item Env:AICF_RUN_CUDA_E2E
```

`AICF_RUN_CUDA_E2E`가 없으면 portable 기본 test discovery에서는 이 test class가
명시적으로 skip된다. 환경 변수를 지정했는데 executable 또는 artifact가 없거나
CUDA 실행/validation이 실패하면 skip하지 않고 테스트 실패로 보고한다.

## Deliberately not implemented

- graph pattern matching 또는 자동 fusion 결정
- CUDA dispatch, compilation, graph execution
- prebuilt executable의 build 또는 profiler report 재생성
- implementation selection/autotuning
- Nsight report parsing 또는 SASS semantic analysis
- instruction/memory-traffic 자동 verification
- repository scanner 또는 persistent registry
- JSON serialization/schema
- fused CUDA kernel/code generation

이 계층은 표현, 명시적 비교, 메모리 내 참조 검증만 제공한다.
