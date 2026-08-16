param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [string]$OutputDirectory,
    [string]$Name,
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"

function Format-NativeCommand {
    param([string]$Command, [string[]]$CommandArguments)

    $items = @($Command) + $CommandArguments
    return (($items | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' ')
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

    $ncuCommand = Get-Command ncu -ErrorAction SilentlyContinue
    if ($ncuCommand) {
        $ncu = $ncuCommand.Source
    } else {
        $ncu = Get-ChildItem `
            -Path "C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.exe" `
            -File `
            -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $ncu) {
        throw "Nsight Compute CLI (ncu) was not found on PATH or under C:\Program Files\NVIDIA Corporation\Nsight Compute *."
    }

    $ncuArguments = @(
        "--set", "detailed",
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
    exit 0
} catch {
    [Console]::Error.WriteLine("CUDA runtime measurement failed: $($_.Exception.Message)")
    exit 1
}
