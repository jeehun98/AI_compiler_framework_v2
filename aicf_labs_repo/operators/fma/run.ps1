param(
    [int]$Elements = 16777216,
    [int]$Iterations = 100
)

$ErrorActionPreference = "Stop"
$operatorDirectory = $PSScriptRoot
$sourcePath = Join-Path $operatorDirectory "fma.cu"
$buildDirectory = Join-Path $operatorDirectory "build"
$executablePath = Join-Path $buildDirectory "fma.exe"

if ($Elements -le 0 -or ($Elements % 2) -ne 0) {
    throw "Elements must be positive and even for half2."
}
if ($Iterations -le 0) {
    throw "Iterations must be positive."
}

$nvcc = (Get-Command nvcc -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

Write-Host "Building fma.cu for sm_86..."
& $nvcc -arch=sm_86 --std=c++17 -Xcompiler=/wd4819 $sourcePath -o $executablePath
if ($LASTEXITCODE -ne 0) {
    throw "CUDA build failed."
}

Write-Host "Running FMA experiment..."
& $executablePath $Elements $Iterations
if ($LASTEXITCODE -ne 0) {
    throw "FMA experiment failed."
}
