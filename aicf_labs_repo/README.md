# AICF Labs

CUDA operator의 implementation/validation과 실제 machine/runtime observation을
분리한 최소 실험 저장소다. Operator registry나 별도 config 없이
`operators/<name>/` directory convention만 사용한다.

## Operator lifecycle

Implementation / Validation:

```text
CUDA source -> build -> run -> numerical validation / benchmark
```

Observation / Measurement:

```text
CUDA source -> observe -> CUBIN line metadata -> SASS
            -> measure -> runtime report -> correlate
```

Canonical correlation은 SASS GPR dataflow와 runtime PC의 SourceCounters를
연결한다. PTX는 `-IncludePtx`에서만 생성하는 optional diagnostic이다.

## Common workflow

공통 script는 다음 convention으로 경로를 찾는다.

```text
source:           operators/<name>/<name>.cu
additional CUDA:  operators/<name>/*.cu
binary:           operators/<name>/build/<name>.exe
static artifacts: operators/<name>/artifacts/
runtime evidence: operators/<name>/runtime/
```

저장소 루트에서 FMA에 직접 적용하는 예:

```powershell
.\tools\operator\build.ps1 -Operator fma
.\tools\operator\run.ps1 -Operator fma `
  -Arguments @("1048576", "20", "1024", "12345")
.\tools\operator\observe.ps1 -Operator fma
.\tools\operator\measure.ps1 -Operator fma `
  -Arguments @("1048576", "1", "120", "12345")
.\operators\fma\correlate.ps1
```

기존 FMA entry point도 compatibility wrapper로 유지한다.

```powershell
.\operators\fma\build.ps1
.\operators\fma\run.ps1
.\operators\fma\observe.ps1
.\operators\fma\measure.ps1
.\operators\fma\correlate.ps1
```

## Adding an operator

새 basic operator에 필요한 최소 파일은
`operators/<name>/<name>.cu` 하나다. CPU reference나 별도 numerical policy가
필요하면 같은 directory에 `validation.cu`와 최소 header를 추가한다.
`tools/operator/build.ps1`은 primary source와 같은 directory의 추가 `*.cu`
translation unit을 함께 빌드한다.

실제 검증 후에만 `VALIDATION.md`를 작성한다. FMA의 복잡한 fusion policy를
다른 operator에 공통 interface로 강제하지 않는다.

## FMA specialization

FMA는 separated MUL + ADD와 fused FMA를 비교하는 optimization experiment다.

- `fma.cu`: CUDA kernels, benchmark, CLI
- `validation.cu`: test data, CPU reference, tolerance/classification, reporting
- `validation.hpp`: 두 translation unit 사이의 최소 interface
- `correlate.ps1`: HFMA2 observation point와 기대 GPR edge
- `build/run/observe/measure.ps1`: common workflow를 호출하는 wrapper

FMA의 canonical static/runtime evidence:

```text
operators/fma/
├─ artifacts/
│  ├─ fma.cubin
│  ├─ fma.sass
│  └─ fma.ptx                  # optional
└─ runtime/
   ├─ fma.ncu-rep
   ├─ fma.txt
   ├─ fma.csv
   ├─ fma_detailed.ncu-rep
   ├─ fma_detailed_sass.txt
   └─ fma_correlation.txt
```

FMA correlation은 `fma.cu:70`의 `HFMA2`와 다음 edge를 검증한다.

```text
0x00a0 --R2--> 0x00f0
0x00c0 --R5--> 0x00f0
0x00d0 --R6--> 0x00f0
0x00f0 --R11--> 0x0100
```

## Analysis boundary

SASS dataflow는 같은 kernel의 straight-line instruction order에서 일반 GPR
`R0..Rn`의 read/write와 가장 가까운 선행 definition만 연결한다. Predicate,
uniform/special register, CFG-aware analysis, memory alias, cross-kernel
dependency는 지원하지 않는다.

2026-08-17 로컬 end-to-end 결과와 evidence hash는
[`operators/fma/VALIDATION.md`](operators/fma/VALIDATION.md)에 기록되어 있다.

## Operator knowledge

재사용 가능한 operator 지식은 기존 CUDA lifecycle과 독립된 파일 계층에 둔다.
Operator의 수학적/tensor/algebraic/numerical/fusion 의미는
`operators/<name>/operator.json`의 상위 필드에, 특정 kernel의 dtype/codegen/
hardware/memory/parallel/SASS/runtime 사실은 `implementations[]`에 둔다.
따라서 GEMM 같은 operator가 본질적으로 Tensor Core를 쓴다고 기록하지 않고,
실제로 해당 instruction을 생성한 variant에만 기록한다.

```text
operators/<name>/operator.json       # OperatorRecord와 variants
knowledge/schemas/operator.schema.json
knowledge/index.json                 # 검증 후 생성되는 registry
tools/knowledge/
├─ validate_operator_metadata.ps1
├─ extract_sass_features.ps1
├─ build_operator_index.ps1
└─ test_operator_knowledge.ps1
```

현재 검증된 첫 작업 단위는 `abs`, `neg`, `relu`다. 다음 명령은 metadata의
schema/semantic/evidence/SASS 일관성을 검사하고 index를 다시 만든다.

```powershell
.\tools\knowledge\validate_operator_metadata.ps1
.\tools\knowledge\build_operator_index.ps1
```

전체 모델, provenance 규칙, 새 operator/variant 등록 절차는
[`knowledge/README.md`](knowledge/README.md)에 있다. 기존 `build`, `run`,
`observe`, `measure` 명령은 metadata를 입력으로 요구하지 않으며 그대로 유지된다.

## Python model representation and frontend slice

`python/aicf_labs`는 기존 선언 계층을 유지하면서, 명시적인 operator wrapper 호출만
eager 실행·추적하는 작은 reference frontend를 제공한다. 사용자는 여전히 layer로
`Sequential`을 구성하고 semantic operator와 implementation evidence를 조회할 수
있다. 실행 가능한 모델은 `Model.forward()`에서 primitive operator를 호출하거나,
현재 지원되는 `Linear`와 `ReLU`를 `Sequential`로 연결한다.

```python
from aicf_labs import Sequential
from aicf_labs.layers import Flatten, Linear, ReLU

model = Sequential(
    Linear(in_features=128, out_features=64, bias=True),
    ReLU(),
    Flatten(),
)

print(model.summary())
```

`Linear`는 기존 구조-only 호출을 유지하면서, weight와 bias를 명시한 경우에만
`MatMulOperator`, `AddOperator`를 직접 실행한다. 별도 layer lowering은 없다.

```python
from aicf_labs import Model, execute_graph, trace_model
from aicf_labs.layers import Linear

class LinearModel(Model):
    def __init__(self):
        self.linear = Linear(
            2,
            2,
            weight=[[2, 1], [-1, 3]],
            bias_value=[1, -2],
        )

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
graph = trace_model(model, [[1, 2], [3, 4]], input_names=("x",))
assert graph.operator_sequence == ("matmul", "add")
assert execute_graph(graph, ([[1, 2], [3, 4]],)) == model([[1, 2], [3, 4]])
```

`Sequential`은 별도 lowering 없이 단일 value를 layer 선언 순서대로 전달한다.

```python
model = Sequential(
    Linear(
        2,
        2,
        weight=[[2, 1], [-1, 3]],
        bias_value=[-1, -10],
    ),
    ReLU(),
)
graph = trace_model(model, [[1, 2], [3, 4]], input_names=("x",))
assert graph.operator_sequence == ("matmul", "add", "relu")
```

현재 executable layer subset은 `Linear`, `ReLU`뿐이다. `Flatten`은 구조 metadata로
유지하며 `Sequential` 실행 중 만나면 명시적으로 지원하지 않는다고 보고한다.

두 Linear도 같은 순차 실행만으로 compose된다. 다음 모델은 별도 MLP/lowering
계층 없이 `matmul, add, relu, matmul, add`를 trace하며 네 parameter literal을 서로
다른 constant node로 보존한다.

```python
mlp = Sequential(
    Linear(2, 3, weight=[[1, -1, 2], [0, 3, -1]], bias_value=[1, 0, -2]),
    ReLU(),
    Linear(3, 1, weight=[[2], [-1], [3]], bias_value=[4]),
)
mlp_graph = trace_model(mlp, [[1, 2], [3, 4]], input_names=("x",))
assert mlp_graph.operator_sequence == ("matmul", "add", "relu", "matmul", "add")
```

`find_linear_relu_candidates()`는 fusion rule이 아니라 read-only structural detector다.
`ELEMENTWISE | PURE | SHAPE_PRESERVING` mask로 root 후보를 좁힌 후
`ReLU <- Add <- MatMul` input 관계를 확인한다. 10-node MLP trace에서 1개 node만
상세 검사해 `matmul.0, add.0, relu.0` 후보 하나를 찾으며 graph를 변경하지 않는다.
`check_linear_relu_legality()`는 detection과 분리된 두 조건만 검사한다.

```text
Add의 다른 입력이 [N] tensor constant이고 MatMul output이 [M, N]인가
MatMul consumer가 Add 하나이고 Add consumer가 ReLU 하나인가
```

실제 MLP에서는 structural/legal candidate가 각각 1개다. Dynamic Add input은
`not_bias_add`, MatMul/Add 분기는 각각 `matmul_has_extra_consumer`,
`add_has_extra_consumer`로 거부된다. Legality result에는 rewrite/apply가 없으며 cost,
backend kernel, dtype/layout legality는 포함하지 않는다.

`rewrite_linear_relu()`는 승인된 `LegalityResult` 하나만 받아 세 primitive node를
backend-unbound `linearRelu(x, weight, bias)` logical node 하나로 치환한다. Legality를
graph에 대해 다시 확인한 뒤 downstream input과 graph output을 새 node ID로 바꾸며
원본 graph는 유지한다.

```text
before: matmul -> add -> relu -> matmul -> add   (10 nodes)
after:  linearRelu         -> matmul -> add       (8 nodes)
```

`LinearReluOperator`의 mask는 현재 실제 검색에 필요한 `PURE` 하나뿐이고 backend
implementation은 비어 있다. Reference executor는 `relu(xW + b)` 합성 의미로 원본과
rewrite graph의 exact equality를 검증한다. 이는 logical rewrite일 뿐 CUDA/kernel
fusion이 실제로 발생했다는 증거가 아니다.

검증까지 통과한 rewrite는 `record_linear_relu_semantic_fusion()`으로 기존
`OptimizationDecision(kind=SEMANTIC_FUSION)`과 `TraceRecord`에 연결한다. 기존 trace
schema에서 decision output은 plan unit ID이므로 logical `PlannedExecutionUnit` 하나를
함께 만들지만, `expected_kernel_launches`, implementation binding, evidence는 기록하지
않는다. Decision reference가 `linearRelu.0` graph node와 verification tolerance를
가리키므로 semantic transformation과 backend 관측을 혼동하지 않는다.

첫 end-to-end 경로는 별도 Tensor/SSA/AST 계층 없이 다음처럼 사용한다.

```python
from aicf_labs import Model, compare_graph_executions, simplify_graph, trace_model
from aicf_labs.ops import add, matmul, mul, relu

class ToyModel(Model):
    def forward(self, x, w):
        y = matmul(x, w)
        y = add(y, 0)
        y = relu(y)
        return mul(y, 1)

graph = trace_model(
    ToyModel(),
    [[1.0, -2.0], [3.0, 4.0]],
    [[2.0, 1.0], [-1.0, 3.0]],
    input_names=("x", "w"),
)
simplification = simplify_graph(graph)
comparison = compare_graph_executions(
    graph,
    simplification.graph,
    ([[1.0, -2.0], [3.0, 4.0]], [[2.0, 1.0], [-1.0, 3.0]]),
)
assert comparison.equivalent
```

`Operator.__call__()`은 trace context 밖에서는 reference value를 계산하고, 안에서는
동시에 `ExecutionGraph` node를 생성한다. 각 node는 단일 출력만 가지므로 node ID를
value ID로 재사용한다. 별도 Value/SSA object 없이 `input_ids`가 producer를 가리키고
consumer는 역으로 계산된다. scalar와 rectangular tensor literal은 `constant` node로
보존된다.

```text
Before: MatMul -> Add(0) -> ReLU -> Mul(1)
After:  MatMul -> ReLU
```

Python의 `to_web_graph()`는 기존 scalar-constant rewrite slice에서 TypeScript
`Graph` schema와 같은 JSON을 만든다.
[`fixtures/toy_model_graph.json`](fixtures/toy_model_graph.json)을 Python serialization
test와 Web rewrite test가 함께 소비하므로 별도 cross-language IR이나 service는 없다.
Python end-to-end는 필요한 `add-zero`, `mul-one`만 적용하고, Web test는 같은 실제
trace에 기존 rewrite rule이 그대로 적용됨을 확인한다.

Linear의 tensor weight/bias는 이번 Python reference graph 안에서만 보존한다. 이를
위해 Web Constant 편집기와 document schema를 tensor 상수까지 확장하지 않았으며,
tensor constant가 있는 graph의 `to_web_graph()`는 명시적으로 지원하지 않는다.

### Feature masks

**Mask는 의미를 증명하는 체계가 아니라, 의미적으로 검토할 가치가 있는 계산
영역을 빠르게 찾기 위한 최소 인덱스다.**

현재 mask vocabulary는 실제 model/rewrite matching에 필요한 양의 feature만 둔다.

```text
ELEMENTWISE  REDUCTION  COMMUTATIVE  PURE
SHAPE_PRESERVING  PERMUTATION  BROADCAST
```

`LINEAR`, `ASSOCIATIVE`, `IDEMPOTENT`, `STATEFUL`, `CONTROL_FLOW`는 현재 활성
operator/rule에서 인덱스로 소비되지 않으므로 추가하지 않는다. Flag가 없다는 것은
반대 의미의 증명이 아니라 단지 그 feature로 인덱싱하지 않았다는 뜻이다.

Python의 `Operator.matches(required)`와 웹 graph의 `nodesMatchingMask()`가 먼저
후보를 좁힌다. 그 다음에만 rule의 `findMatches()`가 operator 종류, 상수 값, edge,
축 같은 상세 조건을 검사한다. Mask match 자체는 rewrite를 허용하지 않는다.
`summarize_masks()`/`summarizeMasks()`는 연속 operator의 공통 bit(AND)와 하나라도
존재하는 bit(OR)만 계산하며 별도 region IR을 만들지 않는다.

현재 웹 MVP가 자동 확인하는 rewrite 후 검증은 `validateGraph()`의 구조 일관성이다.
Python vertical slice는 원본과 단순화 graph를 작은 reference executor로 각각 실행해
명시적인 tolerance로 output을 비교한다.
dtype, NaN/Inf, signed zero 같은 의미 조건은 rule의 `conditions`에 남아 있으며 mask가
이를 통과했다고 간주하지 않는다. 해당 정책을 판정할 semantic verifier가 연결되기
전까지 이런 rule은 `conditionally-exact` candidate다.

`searchRewriteCandidates()`는 별도 profiler 없이 rule별 `nodesScanned`,
`maskAccepted`, `conditionAccepted`, `rewritesApplied`와 세 pass rate 및
`candidateReduction`을 반환한다. `{ useMask: false }`로 같은 matcher를 mask 없이
실행해 결과 동등성과 false negative를 검사할 수 있다. `rewritesApplied`는 UI에
commit된 횟수가 아니라 구조 검증과 중복 제거를 통과한 candidate graph 수다.

20-node representative graph에서 얻어 회귀 테스트로 고정한 결과는 다음과 같다.

```text
rule                scanned  mask-pass  condition-pass  rewrite  reduction
add-zero                 20          8               1        1      60.0%
mul-one                  20          8               1        1      60.0%
mul-zero                 20          8               1        1      60.0%
constant-fold            20          8               1        1      60.0%
add-commute              20          6               3        3      70.0%
mul-commute              20          6               3        3      70.0%
double-transpose         20          3               1        1      85.0%
```

현재 활성 region rule은 없으며 `MaskSummary`는 production rewrite에서 소비되지
않는다. 새 rule에 억지로 연결하지 않고 제거 후보로 유지한다.

```text
Operator / Graph
  -> feature mask screening
  -> rule-specific matching and conditions
  -> semantic verification / structural graph validation
  -> rewrite
  -> runtime/PTX/SASS evidence comparison
```

SASS instruction, shared memory, barrier와 같은 구현 관찰은 semantic mask에 넣지
않는다. Python implementation은 구체 `SassEvidence`를, trace 계층은 provenance가
있는 `ExecutionEvidence`를 사용한다.

```text
Linear(bias=True) -> MatMulOperator, AddOperator
ReLU              -> ReluOperator
Flatten           -> ReshapeOperator
```

현재 repository artifact와 정확히 연결한 implementation은 `ReluOperator`의
`fp32_scalar`뿐이다. `operators/relu/relu.cu`의 `relu_fp32`와
`operators/relu/artifacts/relu.sass`의 `FMNMX`를 evidence로 사용한다. 기존
elementwise `add_fp32`는 Linear bias broadcasting을 구현하지 않고, GEMM 실험도
일반 MatMul 계약과 동일하다고 단정하지 않았으므로 두 operator의 implementation은
비워 두었다.

외부 dependency는 없다. Repository root에서 다음과 같이 실행한다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\python).Path
python -m unittest discover -s tests -v
```

이 reference 경로는 현재 Add, Mul, MatMul, ReLU와 테스트에 필요한 소수 operator만
지원한다. 범용 Tensor runtime, autograd, CUDA dispatch, allocation/planner, symbolic
shape inference, training, code generation은 제공하지 않는다.

## Plan, lowering, and execution evidence

`python/aicf_labs`에는 frontend 계획과 backend 관측을 분리해 연결하는 immutable
trace value objects도 있다.

```text
logical operators -> decision -> plan unit -> implementation binding -> evidence
```

이 구조는 semantic fusion 계획을 실제 kernel/instruction fusion으로 간주하지
않으며, 기대 launch 수와 관측 launch 수를 별도로 비교한다. 상세 책임 경계와
예제는 [`python/EXECUTION_TRACE.md`](python/EXECUTION_TRACE.md)에 있다.
검증된 `LinearReLU` semantic rewrite는 backend 선택 전 상태를 명시적인
`ImplementationBinding(status=UNBOUND, backend=None)`으로 기록하며, 이때 kernel
launch 기대값과 runtime evidence는 계속 비워 둔다.
기존 `gemm`, `add`, `relu` CUDA source/build/probe 자산은 각각 독립 구현의
산출물이며 fused `LinearReLU` 구현으로 취급하지 않는다. 현재 직접 대응하는 fused
kernel과 metadata가 없으므로 semantic plan은 `UNBOUND`를 유지한다.

등록된 concrete implementation의 선택은 `select_implementation()`으로 명시적으로
수행한다. 첫 production 경로는 `ReluOperator.fp32_scalar`를 사용하며, 선택 결과는
`ImplementationBinding(status=SELECTED)`과 `selection_mode=explicit`만 기록한다.
이 단계는 kernel launch 기대값이나 execution evidence를 만들지 않는다.

기존 `add.exe`를 직접 실행하고 plan, lowering, process stdout, SASS/profiler
artifact, evidence, comparison을 단계별로 확인하는 opt-in CUDA 테스트도 제공한다.

```powershell
$env:PYTHONPATH = (Resolve-Path .\python).Path
$env:AICF_RUN_CUDA_E2E = "1"
python .\tests\test_add_cuda_trace_e2e.py -v
Remove-Item Env:AICF_RUN_CUDA_E2E
```

기본 test discovery에서는 GPU/executable 의존성을 피하기 위해 이 테스트를
skip한다.
