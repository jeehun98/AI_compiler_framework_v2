$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$profileElements = 1048576
$profileIterations = 1
$validationElements = 120
$seed = 12345
$profileArguments = @(
    "$profileElements",
    "$profileIterations",
    "$validationElements",
    "$seed"
)

& (Join-Path $repositoryRoot "tools/operator/measure.ps1") `
    -Operator "fma" `
    -Arguments $profileArguments
exit $LASTEXITCODE
