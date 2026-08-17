param(
    [Parameter(Mandatory = $true)]
    [string]$Report,

    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

try {
    if (-not (Test-Path -LiteralPath $Report -PathType Leaf)) {
        throw "Nsight Compute report does not exist: $Report"
    }

    $reportPath = (Resolve-Path -LiteralPath $Report).Path
    $outputPath = [System.IO.Path]::GetFullPath($Output)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null

    $ncu = Get-NsightComputeCli

    $arguments = @("--import", $reportPath, "--page", "source", "--print-source", "sass")
    $displayCommand = Format-NativeCommand $ncu $arguments
    Write-Host $displayCommand
    $sourceOutput = & $ncu @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Compute instruction export failed with exit code $LASTEXITCODE.`nCommand: $displayCommand"
    }

    $sourceOutput | Set-Content -LiteralPath $outputPath -Encoding utf8
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf) -or
        (Get-Item -LiteralPath $outputPath).Length -eq 0) {
        throw "Instruction export did not create a non-empty output file: $outputPath"
    }

    Write-Host "Nsight Compute instruction export: $outputPath"
    exit 0
} catch {
    [Console]::Error.WriteLine("Instruction export failed: $($_.Exception.Message)")
    exit 1
}
