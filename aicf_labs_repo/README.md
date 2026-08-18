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

## Python model representation

`python/aicf_labs`는 실행 엔진이 아닌 최소 선언 계층이다. 사용자는 layer로
`Sequential`을 구성하고, 각 layer가 소유한 semantic operator와 선택 가능한
implementation evidence를 조회한다.

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

이 계층은 tensor 연산, forward/autograd, CUDA dispatch, allocation, shape inference,
optimization/fusion, code generation 또는 serialization을 제공하지 않는다.
