param(
    [Parameter(Mandatory = $true)]
    [string]$Operator,
    [string]$Architecture = "sm_86",
    [string]$OutputDirectory,
    [switch]$IncludePtx
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

try {
    $layout = Get-OperatorLayout -Operator $Operator
    $artifactDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $layout.ArtifactDirectory
    } else {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    }
    $extractScript = Join-Path $layout.RepositoryRoot "tools/cuda_artifacts/extract.ps1"

    & $extractScript `
        -Source $layout.Source `
        -OutputDirectory $artifactDirectory `
        -Name $Operator `
        -Architecture $Architecture `
        -IncludePtx:$IncludePtx
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("Operator observation failed: $($_.Exception.Message)")
    exit 1
}
