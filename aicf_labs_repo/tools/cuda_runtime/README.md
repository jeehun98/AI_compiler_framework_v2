# CUDA runtime observation

`measure.ps1`은 `-lineinfo`로 빌드된 CUDA 실행 파일을 Nsight Compute의
`detailed` set으로 측정한다. SourceCounters와 CUDA source를 report에
포함하기 위해 `--import-source yes`와 `--source-folders`를 사용한다.

```powershell
.\tools\cuda_runtime\measure.ps1 `
  -Executable .\operators\fma\build\fma.exe `
  -OutputDirectory .\operators\fma\runtime `
  -Name fma_detailed `
  -Arguments @("1048576", "1", "120", "12345")
```

`export_instructions.ps1`은 report의 Source/SASS view를 text로 저장한다.
`detailed` report를 입력하면 SASS address와 instruction-level SourceCounters가
함께 출력된다.

```powershell
.\tools\cuda_runtime\export_instructions.ps1 `
  -Report .\operators\fma\runtime\fma_detailed.ncu-rep `
  -Output .\operators\fma\runtime\fma_detailed_sass.txt
```

Correlation의 경계는 다음과 같다.

```text
Static:  CUDA source <-> SASS offset
Runtime: SASS offset <-> runtime PC <-> SourceCounters
```

두 경로는 `kernel + SASS offset`으로 연결한다. 한 CUDA source line은 여러
SASS instruction에 대응할 수 있다. PTX는 optional diagnostic artifact이며
이 runtime correlation의 입력이 아니다.

FMA correlation summary는 operator 전용 스크립트로 재생성한다.

```powershell
.\operators\fma\correlate.ps1
```
