$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$extractScript = Join-Path $repositoryRoot "tools/cuda_artifacts/extract.ps1"

& $extractScript `
    -Source (Join-Path $repositoryRoot "operators/fma/fma.cu") `
    -OutputDirectory (Join-Path $repositoryRoot "operators/fma/artifacts") `
    -Name "fma" `
    -Architecture "sm_86"

exit $LASTEXITCODE
