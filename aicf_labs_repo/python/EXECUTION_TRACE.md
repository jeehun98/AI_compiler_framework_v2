# Execution trace model

`aicf_labs`의 trace 계층은 frontend의 논리적 계획, backend lowering, 실제 실행
관측을 연결하되 같은 사실로 합치지 않는다.

이 문서의 `TraceRecord`는 operator invocation capture나 compiler graph가 아니다.
모델 실행에서 생성되는 계산 기록은 `frontend.py`의 `ExecutionGraph`가 담당하고,
rewrite/semantic reference verification 이후의 plan·binding·runtime evidence만 이
backend trace 계층에 연결한다. 두 표현을 합치지 않은 이유는 실제 계산 edge와
관측 provenance의 수명과 검증 규칙이 다르기 때문이다.

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
Backend 선택 자체를 아직 수행하지 않은 `UNBOUND` record에서는 `backend=None`도
허용한다. `SELECTED`와 `UNAVAILABLE`에는 실제 backend 이름이 반드시 필요하다.
`UNAVAILABLE`은 해당 backend에서 lookup을 수행했지만 구현을 찾지 못했다는
결과이므로 `implementation_ref`를 가질 수 없다.

Runtime probing은 `ExecutionEvidence`에 profiler, runtime trace, binary/SASS,
validation, benchmark에서 실제로 관찰한 사실만 기록한다. 관찰하지 않은 launch
count, latency, validation 결과는 `None`이며 실패를 뜻하지 않는다.

Operator feature mask는 backend trace/evidence 계층에 전달하지 않는다. Mask는
rewrite 이전의 candidate screening에만 쓰이며, `OptimizationDecision`은 rule 조건과
semantic verification을 통과한 결정을 기록한다. `compare_plan_to_evidence()`는 그
결정의 의미를 다시 증명하지 않고 backend 결과만 비교한다.

현재 frontend bridge인 `record_linear_relu_semantic_fusion()`은 legal
`MatMul -> Add -> ReLU` rewrite와 동등성 비교가 모두 성공한 경우에만
`SEMANTIC_FUSION` decision을 만든다. 기존 schema의 output reference를 만족시키기
위해 logical plan unit도 생성하지만 kernel launch 수를 예상하지 않고 binding과
evidence를 만들지 않는다. 대신 unit에는 backend와 implementation을 아직 선택하지
않았다는 `UNBOUND` binding을 연결한다. 따라서 이 record는 semantic rewrite가
일어났다는 기록일 뿐 backend fusion의 관측이 아니다.

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
`knowledge/schemas/operator.schema.json`은 등록이 완료된 CUDA operator 의미, 구현,
artifact의 canonical 지식 계층이다. 현재 index에 등록된 operator는 `abs`, `neg`,
`relu`뿐이다. Trace object는 이 파일을 복제하거나 schema를 확장하지 않는다.
`implementation_ref`는 stable string만 검증하며 외부 metadata 존재 여부를
역참조하지 않는다. 따라서 Add E2E의 `operator:add`는 기존 실험을 식별하는 느슨한
참조이지, 현재 knowledge index 등록을 뜻하지 않는다.

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

Semantic fusion 뒤 backend selection을 아직 시도하지 않았다면 backend가 없는
`UNBOUND` binding을 연결하고 expected launch 수는 정의하지 않는다. 이 경우 비교
결과는 `NOT_APPLICABLE`이다.
Selection을 특정 backend에서 시도했지만 verified fused implementation이 없다면
여러 logical operator ID를 한 unit에 두고 그 backend의 `UNAVAILABLE` binding을
연결할 수 있다. Evidence는 만들지 않으며 기존 GEMM/add artifact를 fused
implementation이라고 연결하지 않는다.

## Explicit implementation selection

`select_implementation()`은 현재 유일한 production selection 경로다. 호출자가
trace의 정확한 `PlannedExecutionUnit`, owning `Operator`, 그 operator에 등록된
`Implementation`, backend와 optional target을 직접 전달한다.

```python
selected_trace = select_implementation(
    trace,
    unit,
    relu_operator,
    relu_operator.implementations[0],
    backend="cuda",
    target="sm_86",
)
```

함수는 기존 `UNBOUND` binding과 같은 ID의 `SELECTED` binding을 가진 새
`TraceRecord`를 반환한다. Unit reference와 원본 trace는 변경하지 않는다. 선택된
reference는 operator-local implementation name을 포함한
`operator:relu:implementation:fp32_scalar` 형식이며, 기존 `configuration`에
`selection_mode=explicit`을 기록한다.

현재 `Implementation` 모델은 CUDA source/kernel만 설명하므로 backend는 `cuda`만
허용한다. 등록 여부, unit/binding reference, operator arity, 명시된 input/output
dtype만 검사한다. `PlannedExecutionUnit`에는 logical operator identity field가
없으므로 전달된 operator가 unit의 실제 의미와 동일하다는 사실까지 검증하지는
않는다.

Selection은 implementation 존재나 실행의 증거가 아니다. 함수는 plan의
`expected_kernel_launches`, evidence, artifact, runtime observation을 생성하거나
변경하지 않는다.

## Existing CUDA asset boundary

Repository의 기존 CUDA workflow는 서로 다른 산출물을 만든다.

```text
operators/<op>/<op>.cu
  -> tools/operator/build.ps1   -> build/<op>.exe
  -> tools/operator/observe.ps1 -> artifacts/<op>.cubin, <op>.sass
  -> tools/operator/measure.ps1 -> runtime/<op>.*
```

이 tool들은 Python `Implementation`, `ImplementationBinding`, `ExecutionEvidence`를
자동 생성하지 않는다. Source는 구현 선언, executable/cubin/SASS는 build·static
artifact, runtime report는 특정 probe 실행의 관측이다. 각 파일은 해당 단계의
증거일 뿐 다음 단계의 사실을 자동으로 증명하지 않는다.

현재 `LinearReluOperator`에는 등록된 `Implementation`이 없고 repository에도
`operators/linear_relu` 또는 GEMM+bias+ReLU epilogue kernel이 없다. 기존
`gemm`, `add`, `relu` artifact 세트를 하나의 fused implementation으로 연결하지
않는다. 그러므로 semantic-fusion record는 backend lookup 전 `UNBOUND`를 유지한다.
향후 CUDA lookup을 실제로 수행했지만 fused implementation을 찾지 못한 경우에만
`backend="cuda"`, `status=UNAVAILABLE`, `implementation_ref=None`을 기록한다.

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

- backend trace object 내부의 graph pattern matching 또는 자동 fusion 결정
- CUDA dispatch, compilation, graph execution
- prebuilt executable의 build 또는 profiler report 재생성
- implementation selection/autotuning
- Nsight report parsing 또는 SASS semantic analysis
- instruction/memory-traffic 자동 verification
- repository scanner 또는 persistent registry
- JSON serialization/schema
- fused CUDA kernel/code generation

이 계층은 표현, 명시적 비교, 메모리 내 참조 검증만 제공한다.
