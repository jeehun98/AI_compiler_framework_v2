$ErrorActionPreference = "Stop"
$operatorDirectory = $PSScriptRoot
$sourcePath = Join-Path $operatorDirectory "fma.cu"
$buildDirectory = Join-Path $operatorDirectory "build"
$executablePath = Join-Path $buildDirectory "fma.exe"
$nvcc = (Get-Command nvcc -ErrorAction Stop).Source

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

$nvccArguments = @(
    "-O3",
    "-arch=sm_86",
    "--std=c++17",
    "-lineinfo",
    "-Xcompiler=/wd4819",
    $sourcePath,
    "-o",
    $executablePath
)
$displayCommand = '"' + $nvcc + '" ' + (($nvccArguments | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
}) -join ' ')

Write-Host $displayCommand
& $nvcc @nvccArguments
if ($LASTEXITCODE -ne 0) {
    throw "CUDA build failed with exit code $LASTEXITCODE.`nCommand: $displayCommand"
}
