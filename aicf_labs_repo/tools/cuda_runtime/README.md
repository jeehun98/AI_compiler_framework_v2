# CUDA runtime measurement

`measure.ps1`은 기존 CUDA executable을 Nsight Compute의 `basic` 또는
`detailed` set으로 profiling한다. `-ExportSummary`를 사용하면 같은 report의
details text와 raw CSV도 함께 저장한다. metric set을 추가하거나 해석하지는
않는다.

```powershell
.\tools\cuda_runtime\measure.ps1 `
  -Executable .\operators\fma\build\fma.exe `
  -OutputDirectory .\operators\fma\runtime `
  -Name fma `
  -Set basic `
  -ExportSummary `
  -Arguments @("1048576", "1", "120", "12345")
```

`export_instructions.ps1`은 기존 detailed `.ncu-rep`의 Source/SASS view와
instruction-level SourceCounters를 text로 저장한다.

```powershell
.\tools\cuda_runtime\export_instructions.ps1 `
  -Report .\operators\fma\runtime\fma_detailed.ncu-rep `
  -Output .\operators\fma\runtime\fma_detailed_sass.txt
```

PTX는 이 runtime path의 입력이 아니다. FMA-specific runtime PC mapping과 GPR
edge 검증은 `operators/fma/correlate.ps1`에만 둔다.
