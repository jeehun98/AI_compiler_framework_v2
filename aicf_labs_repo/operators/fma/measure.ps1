$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$executablePath = Join-Path $PSScriptRoot "build/fma.exe"

& (Join-Path $PSScriptRoot "build.ps1")
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine("FMA build failed; runtime measurement was not started.")
    exit 1
}

& (Join-Path $repositoryRoot "tools/cuda_runtime/measure.ps1") `
    -Executable $executablePath `
    -OutputDirectory (Join-Path $PSScriptRoot "runtime") `
    -Name "fma_detailed" `
    -Arguments @("1048576", "1", "120", "12345")

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& (Join-Path $repositoryRoot "tools/cuda_runtime/export_instructions.ps1") `
    -Report (Join-Path $PSScriptRoot "runtime/fma_detailed.ncu-rep") `
    -Output (Join-Path $PSScriptRoot "runtime/fma_detailed_sass.txt")

exit $LASTEXITCODE
