param(
    [Parameter(Mandatory = $true)]
    [string]$Operator,
    [string]$Architecture = "sm_86",
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

try {
    $layout = Get-OperatorLayout -Operator $Operator
    $buildDirectory = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $layout.BuildDirectory
    } else {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    }
    $executablePath = Join-Path $buildDirectory "$Operator.exe"
    New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

    $nvcc = (Get-Command nvcc -ErrorAction Stop).Source
    $nvccArguments = @(
        "-O3",
        "-arch=$Architecture",
        "--std=c++17",
        "-lineinfo",
        "-Xcompiler=/wd4819"
    ) + @($layout.Sources) + @("-o", $executablePath)
    $displayCommand = Format-OperatorCommand $nvcc $nvccArguments

    Write-Host $displayCommand
    & $nvcc @nvccArguments
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA build failed with exit code $LASTEXITCODE.`nCommand: $displayCommand"
    }

    Write-Host "Operator executable: $executablePath"
    exit 0
} catch {
    [Console]::Error.WriteLine("Operator build failed: $($_.Exception.Message)")
    exit 1
}
