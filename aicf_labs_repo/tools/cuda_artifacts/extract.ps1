param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [string]$OutputDirectory,
    [string]$Name,
    [string]$Architecture = "sm_86",
    [switch]$IncludePtx
)

$ErrorActionPreference = "Stop"

function Format-NativeCommand {
    param([string]$Executable, [string[]]$Arguments)
    $items = @($Executable) + $Arguments
    return (($items | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' ')
}

function Invoke-CheckedNativeCommand {
    param([string]$Description, [string]$Executable, [string[]]$Arguments)
    $displayCommand = Format-NativeCommand $Executable $Arguments
    Write-Host $displayCommand
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE.`nCommand: $displayCommand"
    }
}

try {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "CUDA source file does not exist: $Source"
    }

    $sourcePath = (Resolve-Path -LiteralPath $Source).Path
    $sourceDirectory = Split-Path -Parent $sourcePath
    if ([string]::IsNullOrWhiteSpace($Name)) {
        $Name = [System.IO.Path]::GetFileNameWithoutExtension($sourcePath)
    }
    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $OutputDirectory = Join-Path $sourceDirectory "artifacts"
    }

    $outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
    $cubinPath = Join-Path $outputPath "$Name.cubin"
    $sassPath = Join-Path $outputPath "$Name.sass"
    $ptxPath = Join-Path $outputPath "$Name.ptx"

    $nvcc = (Get-Command nvcc -ErrorAction Stop).Source
    $cuobjdump = (Get-Command cuobjdump -ErrorAction Stop).Source
    $commonArguments = @("-O3", "-arch=$Architecture", "--std=c++17", "-lineinfo", "-Xcompiler=/wd4819")

    Invoke-CheckedNativeCommand "CUBIN generation" $nvcc `
        ($commonArguments + @("-cubin", $sourcePath, "-o", $cubinPath))

    $sassArguments = @("--dump-sass", $cubinPath)
    $sassCommand = Format-NativeCommand $cuobjdump $sassArguments
    Write-Host $sassCommand
    $sassOutput = & $cuobjdump @sassArguments
    if ($LASTEXITCODE -ne 0) {
        throw "SASS extraction failed with exit code $LASTEXITCODE.`nCommand: $sassCommand"
    }
    $sassOutput | Set-Content -LiteralPath $sassPath -Encoding utf8

    if ($IncludePtx) {
        Invoke-CheckedNativeCommand "PTX generation" $nvcc `
            ($commonArguments + @("-src-in-ptx", "-ptx", $sourcePath, "-o", $ptxPath))
    }

    Write-Host "CUDA artifact extraction completed:"
    Write-Host "CUBIN: $cubinPath"
    Write-Host "SASS:  $sassPath"
    if ($IncludePtx) {
        Write-Host "PTX (optional): $ptxPath"
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("CUDA artifact extraction failed: $($_.Exception.Message)")
    exit 1
}
