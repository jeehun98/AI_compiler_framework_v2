$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$executablePath = Join-Path $PSScriptRoot "build/fma.exe"

if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
    & (Join-Path $PSScriptRoot "build.ps1")
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("FMA build failed; runtime measurement was not started.")
        exit 1
    }
}

& (Join-Path $repositoryRoot "tools/cuda_runtime/measure.ps1") `
    -Executable $executablePath `
    -OutputDirectory (Join-Path $PSScriptRoot "runtime") `
    -Name "fma" `
    -Arguments @("1048576", "1", "120", "12345")

exit $LASTEXITCODE
