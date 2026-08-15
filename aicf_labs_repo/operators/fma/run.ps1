param(
    [int]$Elements = 16777216,
    [int]$Iterations = 100,
    [int]$ValidationElements = 4096,
    [uint32]$Seed = 12345
)

$ErrorActionPreference = "Stop"
$operatorDirectory = $PSScriptRoot
$buildDirectory = Join-Path $operatorDirectory "build"
$executablePath = Join-Path $buildDirectory "fma.exe"

if ($Elements -le 0 -or ($Elements % 2) -ne 0) {
    throw "Elements must be positive and even for half2."
}
if ($Iterations -le 0) {
    throw "Iterations must be positive."
}
if ($ValidationElements -le 0 -or ($ValidationElements % 2) -ne 0) {
    throw "ValidationElements must be positive and even for half2."
}

& (Join-Path $operatorDirectory "build.ps1")

Write-Host "Running FMA experiment..."
& $executablePath $Elements $Iterations $ValidationElements $Seed
if ($LASTEXITCODE -ne 0) {
    throw "FMA experiment failed because of a CUDA or input error."
}
