param(
    [switch]$IncludePtx
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

& (Join-Path $repositoryRoot "tools/operator/observe.ps1") `
    -Operator "fma" `
    -Architecture "sm_86" `
    -IncludePtx:$IncludePtx
exit $LASTEXITCODE
