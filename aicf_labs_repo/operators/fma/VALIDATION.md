# FMA local validation record

Status: **VERIFIED**  
Verified at: 2026-08-17 08:52 KST

이 문서는 FMA implementation/validation과 observation/measurement workflow를
로컬 machine에서 끝까지 실행한 기록이다. Runtime counter와 latency는 이 실행의
관찰값이며 다른 GPU, driver 또는 실행 시점에서 달라질 수 있다.

## Environment

- OS: Windows
- GPU: NVIDIA GeForce RTX 3060 (`sm_86`)
- CUDA compiler: 13.3, V13.3.73
- Nsight Compute CLI: 2026.2.1.0, build 38283040
- Source: `operators/fma/fma.cu`

## Reproduction

저장소 루트에서 다음 순서로 실행한다. `run.ps1`과 `measure.ps1`은 기존
`build/fma.exe`를 사용하므로 build가 먼저 성공해야 한다.

```powershell
Set-Location 'C:\Users\as042\OneDrive\Desktop\AI_compiler_framework_v2\aicf_labs_repo'

# Implementation / Validation
.\operators\fma\build.ps1
.\operators\fma\run.ps1

# Observation / Measurement
.\operators\fma\observe.ps1
.\operators\fma\measure.ps1
.\operators\fma\correlate.ps1
```

이 기록의 빠른 numerical validation은 다음 입력으로 재확인했다.

```powershell
.\operators\fma\run.ps1 `
  -Elements 1048576 `
  -Iterations 20 `
  -ValidationElements 1024 `
  -Seed 12345
```

PTX는 이 검증과 correlation에 사용하지 않았다. 별도 diagnostic이 필요할
때만 다음 명령을 사용한다.

```powershell
.\operators\fma\observe.ps1 -IncludePtx
```

## Results

### Implementation / Validation

- Separated MUL + ADD: 53.299 us
- Fused half2 FMA: 42.238 us
- Observed speedup: 1.262x
- Bitwise differences: 1,730 / 4,096 (42.24%)
- Finite tolerance violations: 0 / 3,072 (0.00%)
- Classification mismatches: 258 / 4,096 (6.30%)
- `finite_tolerance`: `ACCEPTED_FOR_JOINTLY_FINITE_RESULTS`
- `accuracy_oriented_contraction`: `FAVORABLE`

`strict_bitwise`와 `classification_preserving`의 rejection은 separated 연산과
fused 연산의 FP16 semantics 차이를 기록한 것이며 실행 실패가 아니다.
Latency는 단일 재확인 실행의 값이므로 성능 회귀 기준으로 고정하지 않는다.

### Static observation

- `artifacts/fma.cubin` 생성 확인
- `artifacts/fma.sass` 생성 확인
- 기본 observe에서 PTX가 필요하지 않음을 확인
- CUDA `fma.cu:70`이 SASS `0x00f0`의
  `HFMA2 R11, R2, R5, R6`에 연결됨을 확인

### Runtime measurement

- Basic report `runtime/fma.ncu-rep` 생성 및 details import 성공
- Basic text/CSV export 생성 확인
- Detailed report `runtime/fma_detailed.ncu-rep` 생성 및 Source/SASS import 성공
- Detailed import에서 1,935개 출력 행 확인

Report import 재확인 명령:

```powershell
ncu --import .\operators\fma\runtime\fma.ncu-rep --page details
ncu --import .\operators\fma\runtime\fma_detailed.ncu-rep `
  --page source --print-source sass
```

### Correlation

최종 `runtime/fma_correlation.txt`에서 다음 관계를 확인했다.

```text
CUDA line:          fma.cu:70
SASS offset:        0x00f0
Kernel base PC:     0x700eac400
Runtime PC:         0x700eac4f0
Instruction:        HFMA2 R11, R2, R5, R6
Stall samples:      1156
Not-issued samples: 1090
Instructions:       16384
Thread instructions: 524288
Predicated-on threads: 524288
```

검증된 GPR def-use edge:

```text
0x00a0 --R2--> 0x00f0
0x00c0 --R5--> 0x00f0
0x00d0 --R6--> 0x00f0
0x00f0 --R11--> 0x0100
```

## Artifact manifest

아래 hash는 2026-08-17 로컬 실행에서 생성된 evidence를 식별한다.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/fma.cubin` | 15,400 | `83F30FBC69A492A830A6A5248D4AB9A3BB00A2CC56F83FE22769929C59618154` |
| `artifacts/fma.sass` | 22,039 | `43D7977F829D579183B64E3430562E8DD025B87AEF207ADC14CF893B268E1FCB` |
| `runtime/fma.ncu-rep` | 2,444,967 | `0934AD75B816035EEBFA0042517DFD6D3332D3E1202984A63B9558B0A9416C89` |
| `runtime/fma.txt` | 244,182 | `1B6A07E30AA6EEC864EFFA43D41E45266AE22CF273BA4CC78895638561B56B2F` |
| `runtime/fma.csv` | 237,961 | `27187FEF5BAF09BD61EB1E9C040D7FC7497AB8008E8EE2233718D4981AE34841` |
| `runtime/fma_detailed.ncu-rep` | 5,248,647 | `1CEA9E09541BCDACC21D4A94FE5423202002DEC9BDBED70D7950B061C9C44960` |
| `runtime/fma_detailed_sass.txt` | 969,438 | `D8BEE4252F789C29DAE6CF31E557F8D5FF9597CDB83A6EB13611C1A9693FFDB0` |
| `runtime/fma_correlation.txt` | 1,747 | `6D5CB68D340ABDA3A80B327104AE4DBE254C06B0A0FE2784E4061960060D1305` |

`artifacts/`와 `runtime/`은 Git ignore 대상이다. `.ncu-rep`는 원본 evidence로
보존하며, 이 문서의 hash는 해당 로컬 파일의 동일성 확인 용도다.

## Workflow refactor regression

2026-08-17에 directory convention 기반 common workflow와 validation source
분리 후 다음을 추가로 확인했다.

- Common build가 `fma.cu`와 `validation.cu`를 함께 컴파일함
- Compatibility `build.ps1`과 `run.ps1` 성공
- Quick run에서 bitwise difference `1,730 / 4,096`, finite tolerance violation
  `0 / 3,072`, classification mismatch `258 / 4,096`으로 기존 결과 유지
- 해당 실행의 separated/fused latency는 45.005 us / 28.774 us, speedup은
  1.564x로 관찰됨
- 임시 output directory의 common observe 성공; PTX 없이 CUBIN/SASS 생성
- Refactored source도 `fma.cu:70 -> SASS 0x00f0 -> runtime PC 0x700eac4f0`
  mapping과 네 GPR edge를 유지함
- `correlate.ps1`을 임시 output으로 실행해 summary 생성을 확인함
- 기존 basic/detailed `.ncu-rep` import와 SourceCounters parsing 성공
- 위 artifact manifest의 canonical evidence hash가 모두 변경되지 않음

새 profiling은 Codex 실행 계정의 `ERR_NVGPUCTRPERM` 때문에 임시 directory에서
완료하지 못했다. 기존 로컬 end-to-end report는 정상이며 common measure는 같은
`tools/cuda_runtime` profiler/report helper를 호출한다. Canonical `.ncu-rep`는
regression 과정에서 재생성하거나 덮어쓰지 않았다.

## Intentional limitations

SASS dataflow는 같은 kernel의 straight-line instruction order에서 일반 GPR
`R0..Rn`과 가장 가까운 선행 definition만 다룬다. Predicate, uniform/special
register, CFG-aware path-sensitive analysis, memory alias, cross-kernel dependency는
이 검증 범위에 포함하지 않았다.
