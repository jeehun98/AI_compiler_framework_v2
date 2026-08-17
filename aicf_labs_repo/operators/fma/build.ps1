$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

& (Join-Path $repositoryRoot "tools/operator/build.ps1") `
    -Operator "fma" `
    -Architecture "sm_86"
exit $LASTEXITCODE
