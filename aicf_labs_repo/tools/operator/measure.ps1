param(
    [Parameter(Mandatory = $true)]
    [string]$Operator,
    [string]$OutputDirectory,
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

try {
    $layout = Get-OperatorLayout -Operator $Operator
    if (-not (Test-Path -LiteralPath $layout.Executable -PathType Leaf)) {
        throw "Operator executable does not exist: $($layout.Executable). Build it first."
    }

    $runtimeDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $layout.RuntimeDirectory
    } else {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    }
    $measureScript = Join-Path $layout.RepositoryRoot "tools/cuda_runtime/measure.ps1"
    $exportInstructionsScript = Join-Path $layout.RepositoryRoot "tools/cuda_runtime/export_instructions.ps1"

    & $measureScript `
        -Executable $layout.Executable `
        -OutputDirectory $runtimeDirectory `
        -Name $Operator `
        -Set "basic" `
        -ExportSummary `
        -Arguments $Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $measureScript `
        -Executable $layout.Executable `
        -OutputDirectory $runtimeDirectory `
        -Name "${Operator}_detailed" `
        -Set "detailed" `
        -Arguments $Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $exportInstructionsScript `
        -Report (Join-Path $runtimeDirectory "${Operator}_detailed.ncu-rep") `
        -Output (Join-Path $runtimeDirectory "${Operator}_detailed_sass.txt")
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("Operator measurement failed: $($_.Exception.Message)")
    exit 1
}
