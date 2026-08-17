param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

function Get-FmaDataflow {
    param(
        [string]$CubinPath,
        [string]$KernelName
    )

    $dataflow = Get-SassGprDataflow -Cubin $CubinPath -KernelName $KernelName
    $focusOffsets = @(0x00a0, 0x00c0, 0x00d0, 0x00f0, 0x0100)
    $focusNodes = @($dataflow.Nodes | Where-Object { $_.Offset -in $focusOffsets })
    if ($focusNodes.Count -ne $focusOffsets.Count) {
        throw "Expected FMA SASS dataflow nodes were not all found."
    }

    $expectedEdges = @(
        @{ From = 0x00a0; Register = "R2"; To = 0x00f0 },
        @{ From = 0x00c0; Register = "R5"; To = 0x00f0 },
        @{ From = 0x00d0; Register = "R6"; To = 0x00f0 },
        @{ From = 0x00f0; Register = "R11"; To = 0x0100 }
    )
    $focusEdges = [System.Collections.Generic.List[object]]::new()
    foreach ($expected in $expectedEdges) {
        $edge = $dataflow.Edges | Where-Object {
            $_.From -eq $expected.From -and
            $_.Register -eq $expected.Register -and
            $_.To -eq $expected.To
        } | Select-Object -First 1
        if (-not $edge) {
            throw ("Expected GPR def-use edge was not found: 0x{0:x4} --{1}--> 0x{2:x4}" -f
                $expected.From, $expected.Register, $expected.To)
        }
        $focusEdges.Add($edge)
    }

    return [pscustomobject]@{
        Nodes = $focusNodes
        Edges = @($focusEdges)
    }
}

function Write-CorrelationSummary {
    param(
        [string]$OutputPath,
        [string]$KernelName,
        $SassMapping,
        $Dataflow,
        $RuntimeCounters
    )

    $dataflowNodeLines = @($Dataflow.Nodes | ForEach-Object {
        $readsText = if ($_.Reads.Count) { $_.Reads -join "," } else { "-" }
        $writesText = if ($_.Writes.Count) { $_.Writes -join "," } else { "-" }
        $sourceText = if ($_.SourceLocations.Count) {
            ($_.SourceLocations | ForEach-Object {
                ($_ -replace '^.*[\\/]fma\.cu', 'fma.cu')
            } | Select-Object -Unique) -join " | "
        } else {
            "unavailable"
        }
        "$($_.OffsetText) $($_.Instruction) | reads: $readsText | writes: $writesText | live GPRs: $($_.LiveRegisters) | source: $sourceText"
    })
    $dataflowEdgeLines = @($Dataflow.Edges | ForEach-Object {
        "$($_.FromText) --$($_.Register)--> $($_.ToText)"
    })

    $lines = @(
        "Kernel: $KernelName",
        "",
        "[CUDA]",
        "file: operators/fma/fma.cu",
        "line: $($SassMapping.SourceLine)",
        "source: $($SassMapping.SourceText)",
        "",
        "[SASS]",
        "source: operators/fma/artifacts/fma.sass",
        "function-relative offset: 0x$($SassMapping.SassOffsetHex)",
        "instruction: $($SassMapping.SassInstruction)",
        "",
        "[SASS DATAFLOW NODES]",
        $dataflowNodeLines,
        "",
        "[DATAFLOW]",
        $dataflowEdgeLines,
        "",
        "[Runtime]",
        "source: operators/fma/runtime/fma_detailed_sass.txt",
        ("kernel base PC: 0x{0:x}" -f $RuntimeCounters.KernelBasePc),
        "runtime PC: $($RuntimeCounters.RuntimePcText)",
        "instruction: $($RuntimeCounters.Instruction)",
        "warp stall sampling (all samples): $($RuntimeCounters.WarpStallSamples)",
        "warp stall sampling (not-issued samples): $($RuntimeCounters.WarpStallNotIssuedSamples)",
        "samples: $($RuntimeCounters.Samples)",
        "instructions executed: $($RuntimeCounters.InstructionsExecuted)",
        "thread instructions executed: $($RuntimeCounters.ThreadInstructionsExecuted)",
        "predicated-on threads executed: $($RuntimeCounters.PredicatedOnThreadsExecuted)",
        "",
        "[Relations]",
        "CUDA -> SASS: CUBIN line metadata; one source line may map to multiple instructions",
        "SASS -> Runtime PC: kernel base PC + function-relative offset; exact instruction matched",
        "Runtime PC -> observations: detailed SourceCounters row at the matched PC",
        "GPR dataflow: nearest preceding definition in straight-line order within the same kernel",
        "",
        "[Known limitation]",
        "Dataflow covers ordinary GPR R0..Rn only. Predicates, uniform/special registers,",
        "path-sensitive CFG analysis, memory aliases, and cross-kernel dependencies are excluded."
    )
    $lines | Set-Content -LiteralPath $OutputPath -Encoding utf8
}

try {
    $repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repositoryRoot "tools/cuda_artifacts/sass_dataflow.ps1")
    . (Join-Path $repositoryRoot "tools/cuda_runtime/correlation.ps1")

    $kernelName = "fma_half2"
    $sourcePath = Join-Path $PSScriptRoot "fma.cu"
    $cubinPath = Join-Path $PSScriptRoot "artifacts/fma.cubin"
    $sassPath = Join-Path $PSScriptRoot "artifacts/fma.sass"
    $runtimePath = Join-Path $PSScriptRoot "runtime/fma_detailed_sass.txt"
    $outputPath = if ([string]::IsNullOrWhiteSpace($OutputPath)) {
        Join-Path $PSScriptRoot "runtime/fma_correlation.txt"
    } else {
        [System.IO.Path]::GetFullPath($OutputPath)
    }

    foreach ($path in @($sourcePath, $cubinPath, $sassPath, $runtimePath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required correlation input does not exist: $path"
        }
    }

    $dataflow = Get-FmaDataflow -CubinPath $cubinPath -KernelName $kernelName
    $sassMapping = Get-SassSourceMapping `
        -SourcePath $sourcePath `
        -SassPath $sassPath `
        -Dataflow $dataflow `
        -Opcode "HFMA2"
    $runtimeCounters = Get-RuntimeCounters `
        -RuntimePath $runtimePath `
        -SassMapping $sassMapping `
        -Opcode "HFMA2"
    Write-CorrelationSummary `
        -OutputPath $outputPath `
        -KernelName $kernelName `
        -SassMapping $sassMapping `
        -Dataflow $dataflow `
        -RuntimeCounters $runtimeCounters

    Write-Host "FMA correlation: $outputPath"
    exit 0
} catch {
    [Console]::Error.WriteLine("FMA correlation failed: $($_.Exception.Message)")
    exit 1
}
