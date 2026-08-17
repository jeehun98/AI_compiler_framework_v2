param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [string]$OutputDirectory,
    [string]$Name,
    [ValidateSet("basic", "detailed")]
    [string]$Set = "detailed",
    [switch]$ExportSummary,
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

function Export-NcuPage {
    param(
        [string]$Ncu,
        [string]$Report,
        [string]$Page,
        [string]$Output,
        [switch]$Csv
    )

    $exportArguments = @("--import", $Report, "--page", $Page)
    if ($Csv) {
        $exportArguments += "--csv"
    }
    $displayCommand = Format-NativeCommand $Ncu $exportArguments
    Write-Host $displayCommand
    $pageOutput = @(& $Ncu @exportArguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Compute $Page export failed with exit code $LASTEXITCODE.`nCommand: $displayCommand"
    }
    $pageOutput | Set-Content -LiteralPath $Output -Encoding utf8
    if (-not (Test-Path -LiteralPath $Output -PathType Leaf) -or
        (Get-Item -LiteralPath $Output).Length -eq 0) {
        throw "Nsight Compute $Page export did not create a non-empty file: $Output"
    }
}

try {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "CUDA executable does not exist: $Executable. Build it before measuring."
    }

    $executablePath = (Resolve-Path -LiteralPath $Executable).Path
    $executableDirectory = Split-Path -Parent $executablePath
    $operatorDirectory = if ((Split-Path -Leaf $executableDirectory) -eq "build") {
        Split-Path -Parent $executableDirectory
    } else {
        $executableDirectory
    }
    if ([string]::IsNullOrWhiteSpace($Name)) {
        $Name = [System.IO.Path]::GetFileNameWithoutExtension($executablePath)
    }
    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $OutputDirectory = Join-Path $operatorDirectory "runtime"
    }

    $outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
    $exportPath = Join-Path $outputPath $Name
    $reportPath = "$exportPath.ncu-rep"

    $ncu = Get-NsightComputeCli

    $ncuArguments = @(
        "--set", $Set,
        "--force-overwrite",
        "--import-source", "yes",
        "--source-folders", $operatorDirectory,
        "--export", $exportPath,
        $executablePath
    ) + $Arguments
    $displayCommand = Format-NativeCommand $ncu $ncuArguments
    Write-Host $displayCommand
    & $ncu @ncuArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Compute measurement failed with exit code $LASTEXITCODE.`nCommand: $displayCommand"
    }
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "Nsight Compute exited successfully but did not create the expected report: $reportPath"
    }

    Write-Host "Nsight Compute report: $reportPath"
    if ($ExportSummary) {
        $textPath = Join-Path $outputPath "$Name.txt"
        $csvPath = Join-Path $outputPath "$Name.csv"
        Export-NcuPage -Ncu $ncu -Report $reportPath -Page "details" -Output $textPath
        Export-NcuPage -Ncu $ncu -Report $reportPath -Page "raw" -Output $csvPath -Csv
        Write-Host "Nsight Compute details: $textPath"
        Write-Host "Nsight Compute raw CSV: $csvPath"
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("CUDA runtime measurement failed: $($_.Exception.Message)")
    exit 1
}
