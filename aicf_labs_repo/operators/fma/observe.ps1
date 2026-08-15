$ErrorActionPreference = "Stop"

function Format-NativeCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $items = @($Executable) + $Arguments
    return (($items | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' ')
}

function Invoke-NativeCaptured {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        [void]$startInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start native command: $Executable"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $exitCode = $process.ExitCode
    $process.Dispose()

    return [pscustomobject]@{
        ExitCode = $exitCode
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Invoke-NativeStdoutFile {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$StdoutPath
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        [void]$startInfo.ArgumentList.Add($argument)
    }

    $outputStream = [System.IO.File]::Create($StdoutPath)
    try {
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) {
            throw "Failed to start native command: $Executable"
        }
        $stdoutTask = $process.StandardOutput.BaseStream.CopyToAsync($outputStream)
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
        $process.Dispose()
    } finally {
        $outputStream.Dispose()
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Stdout = "[stdout written directly to $StdoutPath]"
        Stderr = $stderr
    }
}

function New-CommandRecord {
    param(
        [string]$Command,
        $Result
    )

    return @"
command: $Command
exit code: $($Result.ExitCode)
stdout:
$($Result.Stdout)
stderr:
$($Result.Stderr)
"@
}

function Write-TextFile {
    param(
        [string]$Path,
        [string]$Text
    )

    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $utf8WithoutBom)
}

function Stop-RequiredCommand {
    param(
        [string]$Name,
        [string]$Command,
        $Result
    )

    $details = ($Result.Stderr + $Result.Stdout).Trim()
    throw "$Name failed with exit code $($Result.ExitCode).`nCommand: $Command`n$details"
}

$operatorDirectory = $PSScriptRoot
$sourcePath = Join-Path $operatorDirectory "fma.cu"
$artifactDirectory = Join-Path $operatorDirectory "artifacts"
$astDirectory = Join-Path $artifactDirectory "ast"
$ptxDirectory = Join-Path $artifactDirectory "ptx"
$cubinDirectory = Join-Path $artifactDirectory "cubin"
$sassDirectory = Join-Path $artifactDirectory "sass"
$logsDirectory = Join-Path $artifactDirectory "logs"

foreach ($directory in @(
    $astDirectory,
    $ptxDirectory,
    $cubinDirectory,
    $sassDirectory,
    $logsDirectory
)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$astPath = Join-Path $astDirectory "fma.ast.txt"
$ptxPath = Join-Path $ptxDirectory "fma.sm_86.ptx"
$cubinPath = Join-Path $cubinDirectory "fma.sm_86.cubin"
$sassPath = Join-Path $sassDirectory "fma.sm_86.sass"
$nvccVersionPath = Join-Path $logsDirectory "nvcc_version.txt"
$ptxasVerbosePath = Join-Path $logsDirectory "ptxas_verbose.txt"
$buildCommandPath = Join-Path $logsDirectory "build_command.txt"
$summaryPath = Join-Path $logsDirectory "observation_summary.txt"

$nvcc = (Get-Command nvcc -ErrorAction Stop).Source
$commandRecords = [System.Collections.Generic.List[string]]::new()

$versionArguments = @("--version")
$versionCommand = Format-NativeCommand $nvcc $versionArguments
Write-Host $versionCommand
$versionResult = Invoke-NativeCaptured $nvcc $versionArguments $operatorDirectory
Write-TextFile $nvccVersionPath ($versionResult.Stdout + $versionResult.Stderr)
if ($versionResult.ExitCode -ne 0) {
    Stop-RequiredCommand "nvcc version query" $versionCommand $versionResult
}

$commonNvccArguments = @(
    "-O3",
    "-arch=sm_86",
    "--std=c++17",
    "-Xcompiler=/wd4819"
)

$ptxArguments = $commonNvccArguments + @("-ptx", $sourcePath, "-o", $ptxPath)
$ptxCommand = Format-NativeCommand $nvcc $ptxArguments
Write-Host $ptxCommand
$ptxResult = Invoke-NativeCaptured $nvcc $ptxArguments $operatorDirectory
$commandRecords.Add((New-CommandRecord $ptxCommand $ptxResult))
Write-TextFile $buildCommandPath ($commandRecords -join "`r`n")
if ($ptxResult.ExitCode -ne 0) {
    Stop-RequiredCommand "PTX extraction" $ptxCommand $ptxResult
}

$cubinArguments = $commonNvccArguments + @(
    "-Xptxas=-v",
    "-cubin",
    $sourcePath,
    "-o",
    $cubinPath
)
$cubinCommand = Format-NativeCommand $nvcc $cubinArguments
Write-Host $cubinCommand
$cubinResult = Invoke-NativeCaptured $nvcc $cubinArguments $operatorDirectory
$commandRecords.Add((New-CommandRecord $cubinCommand $cubinResult))
Write-TextFile $ptxasVerbosePath ($cubinResult.Stdout + $cubinResult.Stderr)
Write-TextFile $buildCommandPath ($commandRecords -join "`r`n")
if ($cubinResult.ExitCode -ne 0) {
    Stop-RequiredCommand "CUBIN extraction" $cubinCommand $cubinResult
}

$sassStatus = "failed"
$sassTool = $null
$cuobjdumpCommandInfo = Get-Command cuobjdump -ErrorAction SilentlyContinue
if ($cuobjdumpCommandInfo) {
    $sassTool = $cuobjdumpCommandInfo.Source
    $sassArguments = @("--dump-sass", $cubinPath)
    $sassCommand = Format-NativeCommand $sassTool $sassArguments
    Write-Host $sassCommand
    $sassResult = Invoke-NativeCaptured $sassTool $sassArguments $operatorDirectory
    $commandRecords.Add((New-CommandRecord $sassCommand $sassResult))
} else {
    $sassResult = [pscustomobject]@{ ExitCode = -1; Stdout = ""; Stderr = "cuobjdump not found" }
    $sassCommand = "cuobjdump --dump-sass $cubinPath"
}

if ($sassResult.ExitCode -ne 0) {
    $nvdisasmCommandInfo = Get-Command nvdisasm -ErrorAction SilentlyContinue
    if (-not $nvdisasmCommandInfo) {
        Write-TextFile $buildCommandPath ($commandRecords -join "`r`n")
        Stop-RequiredCommand "SASS extraction" $sassCommand $sassResult
    }
    $sassTool = $nvdisasmCommandInfo.Source
    $sassArguments = @($cubinPath)
    $sassCommand = Format-NativeCommand $sassTool $sassArguments
    Write-Host $sassCommand
    $sassResult = Invoke-NativeCaptured $sassTool $sassArguments $operatorDirectory
    $commandRecords.Add((New-CommandRecord $sassCommand $sassResult))
}

Write-TextFile $buildCommandPath ($commandRecords -join "`r`n")
if ($sassResult.ExitCode -ne 0) {
    Stop-RequiredCommand "SASS extraction" $sassCommand $sassResult
}
Write-TextFile $sassPath $sassResult.Stdout
$sassStatus = "success ($([System.IO.Path]::GetFileName($sassTool)))"

$astStatus = "unavailable"
$astCommand = "clang++ not found"
$astExitCode = -1
$astError = "clang++ was not found on PATH"
$clangCommandInfo = Get-Command clang++ -ErrorAction SilentlyContinue
if ($clangCommandInfo) {
    $clang = $clangCommandInfo.Source
    $cudaRoot = Split-Path (Split-Path $nvcc -Parent) -Parent
    $astArguments = @(
        "-x",
        "cuda",
        "--cuda-gpu-arch=sm_86",
        "--cuda-path=$cudaRoot",
        "-Xclang",
        "-ast-dump",
        "-fsyntax-only",
        $sourcePath
    )
    $astCommand = Format-NativeCommand $clang $astArguments
    Write-Host $astCommand
    $astResult = Invoke-NativeStdoutFile $clang $astArguments $operatorDirectory $astPath
    $astExitCode = $astResult.ExitCode
    $astError = $astResult.Stderr.Trim()
    $commandRecords.Add((New-CommandRecord $astCommand $astResult))
    if ($astResult.ExitCode -eq 0) {
        $astStatus = "success"
    } else {
        Write-TextFile $astPath "AST unavailable`r`n"
        $astStatus = "unavailable (exit code $astExitCode)"
    }
} else {
    Write-TextFile $astPath "AST unavailable`r`n"
}
Write-TextFile $buildCommandPath ($commandRecords -join "`r`n")

$ptxText = [System.IO.File]::ReadAllText($ptxPath)
$sassText = [System.IO.File]::ReadAllText($sassPath)
$ptxEntries = [regex]::Matches($ptxText, '(?m)\.entry\s+([^\s(]+)') |
    ForEach-Object { $_.Groups[1].Value } |
    Select-Object -Unique
$sassFunctions = [regex]::Matches($sassText, '(?m)^\s*Function\s*:\s*(\S+)') |
    ForEach-Object { $_.Groups[1].Value } |
    Select-Object -Unique

$toolkitMatch = [regex]::Match(
    ($versionResult.Stdout + $versionResult.Stderr),
    'release\s+([^,]+),\s+V([^\s]+)'
)
$toolkitVersion = if ($toolkitMatch.Success) {
    "release $($toolkitMatch.Groups[1].Value), V$($toolkitMatch.Groups[2].Value)"
} else {
    "see logs/nvcc_version.txt"
}

$relativeSource = "operators/fma/fma.cu"
$relativeArtifacts = @(
    "operators/fma/artifacts/ast/fma.ast.txt",
    "operators/fma/artifacts/ptx/fma.sm_86.ptx",
    "operators/fma/artifacts/cubin/fma.sm_86.cubin",
    "operators/fma/artifacts/sass/fma.sm_86.sass",
    "operators/fma/artifacts/logs/nvcc_version.txt",
    "operators/fma/artifacts/logs/ptxas_verbose.txt",
    "operators/fma/artifacts/logs/build_command.txt",
    "operators/fma/artifacts/logs/observation_summary.txt"
)

$summaryLines = [System.Collections.Generic.List[string]]::new()
$summaryLines.Add("source file: $relativeSource")
$summaryLines.Add("CUDA Toolkit version: $toolkitVersion")
$summaryLines.Add("target architecture: sm_86")
$summaryLines.Add("compile flags: -O3 -arch=sm_86 --std=c++17 -Xcompiler=/wd4819; CUBIN adds -Xptxas=-v")
$summaryLines.Add("AST extraction status: $astStatus")
if ($astStatus -ne "success") {
    $errorLines = $astError -split "`r?`n" | Select-Object -First 8
    $summaryLines.Add("AST command: $astCommand")
    $summaryLines.Add("AST failure exit code: $astExitCode")
    $summaryLines.Add("AST error: $($errorLines -join ' | ')")
}
$summaryLines.Add("PTX extraction status: success")
$summaryLines.Add("CUBIN extraction status: success")
$summaryLines.Add("SASS extraction status: $sassStatus")
$summaryLines.Add("artifact paths:")
foreach ($path in $relativeArtifacts) {
    $summaryLines.Add("- $path")
}
$summaryLines.Add("PTX kernel entries: $($ptxEntries -join ', ')")
$summaryLines.Add("SASS kernel functions: $($sassFunctions -join ', ')")
$summaryLines.Add("HFMA2 found: $([bool]($sassText -match '(?i)HFMA2'))")
$summaryLines.Add("HMUL2 or corresponding multiply found: $([bool]($sassText -match '(?i)HMUL2|MUL.*F16'))")
$summaryLines.Add("HADD2 or corresponding add found: $([bool]($sassText -match '(?i)HADD2|ADD.*F16'))")
Write-TextFile $summaryPath (($summaryLines -join "`r`n") + "`r`n")

Write-Host "Observation complete: $artifactDirectory"
