# AICF Labs

CUDA operator 구현을 하나씩 직접 작성하고 실행해 보는 최소 실험 저장소다.

현재는 FMA 하나만 다룬다. Frontend, backend registry, mask 또는 공통 분석
framework는 사용하지 않고, CUDA 도구가 만든 원본 artifact만 저장한다.

## 구조

```text
aicf_labs_repo/
├─ README.md
└─ operators/
   └─ fma/
      ├─ artifacts/
      ├─ build.ps1
      ├─ correlate.ps1
      ├─ fma.cu
      ├─ measure.ps1
      ├─ observe.ps1
      └─ run.ps1
```

새 operator가 필요해지면 `operators/<operator>/` 폴더를 하나 추가한다.

## FMA에서 비교하는 구현

분리 구현:

```text
temporary[i] = a[i] * b[i]
output[i] = temporary[i] + c[i]
```

최적화 구현:

```text
output[i] = fma(a[i], b[i], c[i])
```

최적화 커널은 FP16 값 두 개를 한 thread가 처리하도록 `half2`와
`__hfma2`를 사용한다. 분리 구현과 비교하면 다음 두 가지가 제거된다.

- FP16 intermediate tensor 하나
- Mul kernel과 Add kernel 중 한 번의 kernel launch

모든 배열은 연속 메모리이며 element 수는 짝수여야 한다. `cudaMalloc`으로
할당한 pointer는 `half2` 접근에 필요한 alignment를 만족한다.

이 코드는 elementwise CUDA FMA이며 Tensor Core MMA가 아니다.

## 환경과 실행

기준 환경:

- Windows
- CUDA Toolkit 13.3
- NVIDIA RTX 3060 (`sm_86`)
- Visual Studio 2022 C++ compiler

저장소 루트에서 실행한다.

```powershell
.\operators\fma\build.ps1
.\operators\fma\run.ps1
.\operators\fma\observe.ps1
```

Nsight Compute runtime report를 생성하려면:

```powershell
.\operators\fma\build.ps1
.\operators\fma\measure.ps1
.\operators\fma\observe.ps1
.\operators\fma\correlate.ps1
```

Primary observation은 CUDA source를 CUBIN line metadata를 통해 SASS offset에
연결하고, 같은 offset을 runtime PC와 detailed SourceCounters에 연결한다.
FMA correlation은 SASS operand의 일반 GPR read/write와 가장 가까운 선행
definition을 이용해 kernel 내부 producer-consumer edge도 기록한다. Predicate,
uniform/special register와 memory alias dependency는 이 최소 probe의 범위 밖이다.

```text
CUDA Source -> SASS -> SASS Dataflow -> Runtime Observation
```

PTX는 compiler lowering을 별도로 확인할 때만 생성하는 optional diagnostic
artifact이며 기본 observe/correlate/runtime workflow의 입력이 아니다.

작은 입력으로 빠르게 확인하려면:

```powershell
.\operators\fma\run.ps1 -Elements 1048576 -Iterations 20
```

CUDA 소스에서 canonical CUBIN과 SASS를 생성하려면:

```powershell
.\tools\cuda_artifacts\extract.ps1   -Source .\operators\fma\fma.cu
```

PTX diagnostic도 필요하면 `-IncludePtx`를 추가한다.

스크립트는 `fma.cu`를 executable 하나로 빌드한 뒤 분리 구현과 융합 구현의
평균 실행 시간을 출력한다. 생성 파일은 `operators/fma/build/` 아래에만
생기며 Git에 포함되지 않는다.

## Benchmark와 validation

Benchmark는 큰 입력에서 separated와 fused kernel 실행 시간만 측정한다.
입력 생성, host/device 복사, CPU reference 계산과 결과 비교는 측정 구간에
포함되지 않는다. 두 구현은 같은 non-zero 입력과 서로 다른 output buffer를
사용한다.

Validation은 작은 별도 입력에서 각 구현을 한 번씩 실행하고 FP16 결과를
비교한다. 직접 executable을 실행할 때의 인자는 다음과 같다.

```text
fma.exe [benchmark_elements] [iterations] [validation_elements] [seed]
```

기본값은 `16777216`, `100`, `4096`, `12345`다. PowerShell에서는 다음처럼
선택적으로 전달할 수 있다.

```powershell
.\operators\fma\run.ps1 `
  -Elements 1048576 `
  -Iterations 20 `
  -ValidationElements 1024 `
  -Seed 12345
```

Validation case는 `ordinary`, `cancellation`, `range_stress`, `special`이다.
Reference는 FP16으로 양자화된 입력을 double로 변환한 뒤 `a * b + c`로
계산한다.

Bitwise equality는 FP16의 16-bit 표현이 같은지를 뜻한다. Numerical
tolerance는 유한 결과의 오차가 허용 범위인지 판단하므로 서로 다른
개념이다. 기본 finite 정책은 다음과 같다.

```text
|separated - fused| <= 2^-9 + 2^-9 * |double reference|
relative denominator = max(|double reference|, 2^-14)
```

`2^-9`는 1 근처에서 약 두 FP16 ULP를 허용하기 위한 보수적인 시작값이다.
NaN과 Inf는 별도 classification으로 비교한다. 분리 Mul/Add는 곱셈과
덧셈에서 각각 반올림하지만 FMA는 마지막에 한 번 반올림하므로 bitwise
차이가 정상적으로 나타날 수 있다. 따라서 출력의 bitwise difference rate는
두 구현의 차이율이며 incorrect rate가 아니다.

프로그램은 관찰 결과와 다음 네 가지 policy verdict를 분리해 모두 출력한다.

- `strict_bitwise`: 모든 FP16 bit pattern이 같은지 판단한다.
- `finite_tolerance`: 둘 다 finite이고 reference를 비교할 수 있는 결과만
  tolerance로 판단한다. classification 차이는 이 정책의 범위 밖이다.
- `classification_preserving`: finite, Inf, NaN 분류가 같은지 판단한다.
- `accuracy_oriented_contraction`: reference-comparable 결과에서 fused가 더
  나쁜 사례가 없고 finite tolerance를 만족하는지 관찰한다.

기본 실행은 특정 정책 하나를 전체 정답으로 선택하지 않는다. CUDA 오류,
잘못된 입력, allocation/copy/kernel 오류는 exit code 1이지만, 비교가 정상적으로
끝난 뒤 어떤 정책이 `REJECTED`인 것은 실행 오류가 아니므로 exit code 0이다.

`range_stress`에서는 separated 구현의 FP16 intermediate multiplication이
Inf로 overflow한 뒤 덧셈되는 반면, fused FMA는 중간 FP16 반올림과 overflow
없이 finite 최종 결과를 만들 수 있다. 이때의 classification divergence는 두
구현의 FP16 의미 차이로 기록한다. `accuracy_oriented_contraction` verdict도
선택한 입력 case, seed와 validation 크기에서 얻은 관찰이며 모든 FP16 입력에
대한 증명이 아니다. 결과를 비교하거나 문서화할 때는 GPU, CUDA Toolkit 버전,
seed와 case별 validation element 수를 함께 기록해야 한다.
