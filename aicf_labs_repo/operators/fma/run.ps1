param(
    [int]$Elements = 16777216,
    [int]$Iterations = 100,
    [int]$ValidationElements = 4096,
    [uint32]$Seed = 12345
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

if ($Elements -le 0 -or ($Elements % 2) -ne 0) {
    throw "Elements must be positive and even for half2."
}
if ($Iterations -le 0) {
    throw "Iterations must be positive."
}
if ($ValidationElements -le 0 -or ($ValidationElements % 2) -ne 0) {
    throw "ValidationElements must be positive and even for half2."
}

& (Join-Path $repositoryRoot "tools/operator/run.ps1") `
    -Operator "fma" `
    -Arguments @("$Elements", "$Iterations", "$ValidationElements", "$Seed")
exit $LASTEXITCODE
