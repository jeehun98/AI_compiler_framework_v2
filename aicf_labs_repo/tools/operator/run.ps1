param(
    [Parameter(Mandatory = $true)]
    [string]$Operator,
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

try {
    $layout = Get-OperatorLayout -Operator $Operator
    if (-not (Test-Path -LiteralPath $layout.Executable -PathType Leaf)) {
        throw "Operator executable does not exist: $($layout.Executable). Build it first."
    }

    Write-Host "Running $($Operator.ToUpperInvariant()) experiment..."
    & $layout.Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Operator experiment failed with exit code $LASTEXITCODE."
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("Operator run failed: $($_.Exception.Message)")
    exit 1
}
