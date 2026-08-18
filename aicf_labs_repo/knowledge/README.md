# Operator knowledge model

이 계층은 기존 CUDA source, cubin/SASS, executable, Nsight Compute report를
재생성하지 않고 지식 객체로 연결한다. JSON은 Windows PowerShell에서 외부 YAML
parser 없이 읽을 수 있다는 이유로 선택했다. Canonical contract는
`schemas/operator.schema.json`이다.

## Responsibility boundary

`OperatorRecord` 상위 필드는 구현과 무관한 의미를 표현한다.

- `identity`, `semantics`: 이름, arity, 수학식, semantic kind
- `tensor`: input/output shape, rank, axis, permutation, layout 관계
- `algebra`: 수학적 성질
- `numerical`: 부동소수점 및 special-value 성질
- `fusion`: materialization/synchronization 조건을 포함한 fusion 가능성
- `decomposition`: 복합 operator의 수학적 구성 단계

`implementations[]`는 특정 source kernel variant의 사실을 표현한다.

- `dtype`, `validation`, `launch`, `codegen`
- `hardware`, `memory`, `parallel`
- static cubin에서 추출한 `sass`
- runtime report에 기록된 `measurements`

Tensor Core, SFU, shared memory, vector access, tile, launch, instruction, runtime
metric은 implementation property다. Operator-level semantic 객체에 이 정보를
올리지 않는다.

## Provenance and tri-state values

Implementation property group은 `declared`, `observed`, `inferred`를 분리한다.

- `declared`: source와 build option에서 확인한 구현 의도
- `observed`: SASS 또는 runtime artifact에 실제로 존재하는 사실
- `inferred`: source 의미와 관찰 evidence를 함께 사용해 내린 제한된 판단

Non-null provenance property에는 하나 이상의 evidence가 필요하다. 각 evidence는
repository-relative `file`, 안정적인 `locator`, 파일 안에서 literal로 확인되는
`match`, 과장하지 않은 `observation`을 가진다. 검증기는 파일 존재뿐 아니라
`match` 문자열도 검사한다.

Boolean 계열은 다음 의미를 지킨다.

- `true`: 현재 범위에서 성립함
- `false`: 현재 범위에서 성립하지 않음
- `null`: 비적용, 미판정 또는 현재 evidence로 확인 불가

예를 들어 unary operator의 `commutative`는 `false`가 아니라 `null`이다.
수학적 결합법칙은 `algebra.mathematically_associative`, 부동소수점 재결합
안전성은 `numerical.floating_point_reassociation_safe`에 각각 기록한다.

## Property classes

Semantic/tensor/algebraic/numerical/fusion Boolean은 검색과 rule prefilter에
사용한다. 축, shape equation, permutation, dtype, tolerance, launch rule,
instruction count, runtime metric, fusion 조건은 구조화 필드에 둔다. 모든 값을
mask로 압축하지 않는다.

SASS feature extraction은 mnemonic 관찰만 보고한다. `HMMA`는 Tensor Core
instruction evidence지만 operator가 GEMM임을 증명하지 않는다. `MUFU.EX2`도
exponential-family instruction evidence일 뿐 softmax나 sigmoid를 단독으로
확정하지 않는다. Motif는 source semantics를 보강하는 evidence이며 수학적
의미의 독립 증명이 아니다.

`sass.counts.instruction_count`는 현재 cuobjdump text의 offset instruction slot을
세므로 terminal padding `NOP`도 포함한다. Register count는 static text에서
안정적으로 얻지 못하면 `null`로 두고, Nsight Compute의 registers-per-thread는
measurement property로 별도 기록한다.

## Registering a new operator

1. `operators/<name>/<name>.cu`에서 expression, dtype, shape mapping, kernel symbol,
   launch mapping을 확인한다.
2. static `.sass`와 가능한 runtime files를 읽는다. 이름만 보고 property를
   생성하지 않는다.
3. `operators/<name>/operator.json`을 schema에 맞춰 작성한다. `identity.name`은
   directory 이름과 같아야 한다.
4. implementation의 non-null `observed` property마다 실제 evidence를 연결한다.
5. validator를 실행한 뒤 index builder를 실행한다.

```powershell
.\tools\knowledge\validate_operator_metadata.ps1 `
  -Path .\operators\sqrt\operator.json
.\tools\knowledge\build_operator_index.ps1
.\tools\knowledge\test_operator_knowledge.ps1
```

## Adding an implementation variant

같은 수학적 operator의 새 kernel은 기존 상위 의미를 복제하지 않고
`implementations[]`에 새 항목으로 추가한다. Variant마다 source kernel, dtype,
validation tolerance, launch, codegen/hardware/memory/parallel provenance, SASS,
measurement artifacts를 독립적으로 기록한다. 예를 들어 `naive_fp32`와
`wmma_fp16_fp32`의 `uses_tensor_core`는 서로 다른 값을 가질 수 있다.

## Updating SASS evidence

기존 observe workflow로 artifact를 의도적으로 갱신한 경우 먼저 extractor로 새
fact를 확인한다.

```powershell
.\tools\knowledge\extract_sass_features.ps1 `
  -Sass .\operators\relu\artifacts\relu.sass
```

그 다음 metadata의 instruction/features/counts/motifs와 evidence locator를
갱신하고 validator를 실행한다. Validator는 metadata와 extractor 결과를 다시
대조한다. Extractor는 기존 `sass_dataflow.ps1`을 교체하지 않는다. 전자는 안정적
mnemonic/count/motif inventory, 후자는 kernel별 GPR def-use 분석을 담당한다.

## Missing runtime evidence

Runtime artifact가 없으면 `measurements.artifacts`의 해당 경로를 `null`로 둔다.
관찰할 수 없는 measurement property도 `null`로 두고 evidence를 꾸며내지 않는다.
모든 runtime artifact가 `null`인데 non-null observed runtime value가 있으면
validator가 실패한다. 새 측정이 필요할 때만 기존
`tools/operator/measure.ps1` workflow를 별도로 실행한다.

## Using metadata for fusion

Fusion 후보 선택은 먼저 semantic/tensor/fusion flag로 거르고 `conditions`를
검사한다. Elementwise producer/consumer라도 output element mapping이 바뀌거나,
global synchronization/materialization이 필요하거나, dtype/special-value 정책이
달라지면 fusion 가능으로 단정하지 않는다. Implementation 선택 단계에서는
resource 사용, memory behavior, launch mapping, runtime bottleneck을 함께 본다.
현재 metadata는 fusion decision evidence이며 fusion kernel 생성을 수행하지 않는다.

## Validation performed by the tool

`validate_operator_metadata.ps1`은 다음을 한 번에 검사한다.

- JSON Schema의 required/type/enum/pattern/additional-property 계약
- operator 이름과 directory 이름 일치
- unary `commutative: null`
- hardware/memory/parallel property의 implementation-level 위치와 tri-state 값
- non-null provenance와 evidence 연결
- 모든 source/artifact/runtime 참조와 evidence literal 존재
- static SASS 재추출 결과와 recorded feature/count/instruction/motif 일치
- runtime artifact 없이 관찰 수치가 기록되지 않았는지 여부

Schema와 validator 변경 후에는 기존 공통 PowerShell script도 parser로 확인하고,
CUDA/Nsight 도구가 있는 환경에서는 원래 build/run/observe/measure workflow를
별도로 회귀 실행한다.
