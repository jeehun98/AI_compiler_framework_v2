$ErrorActionPreference = "Stop"

function Find-PreviousMatch {
    param(
        [string[]]$Lines,
        [int]$StartIndex,
        [string]$Pattern
    )

    for ($index = $StartIndex; $index -ge 0; $index--) {
        $match = [regex]::Match($Lines[$index], $Pattern)
        if ($match.Success) {
            return $match
        }
    }
    return $null
}

try {
    $repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    . (Join-Path $repositoryRoot "tools/cuda_artifacts/sass_dataflow.ps1")

    $sourcePath = Join-Path $PSScriptRoot "fma.cu"
    $ptxPath = Join-Path $PSScriptRoot "artifacts/fma.ptx"
    $cubinPath = Join-Path $PSScriptRoot "artifacts/fma.cubin"
    $runtimePath = Join-Path $PSScriptRoot "runtime/fma_detailed_sass.txt"
    $outputPath = Join-Path $PSScriptRoot "runtime/fma_correlation.txt"

    foreach ($path in @($sourcePath, $ptxPath, $cubinPath, $runtimePath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required correlation input does not exist: $path"
        }
    }

    $nvdisasm = (Get-Command nvdisasm -ErrorAction Stop).Source
    $cudaSass = @(& $nvdisasm --print-line-info-inline $cubinPath 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvdisasm --print-line-info-inline failed with exit code $LASTEXITCODE."
    }
    $ptxSass = @(& $nvdisasm --print-line-info-ptx $cubinPath 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvdisasm --print-line-info-ptx failed with exit code $LASTEXITCODE."
    }

    $sassMatchInfo = $cudaSass |
        Select-String -Pattern '^\s*/\*([0-9a-fA-F]+)\*/\s+HFMA2\s+(.+?)\s*;' |
        Select-Object -First 1
    if (-not $sassMatchInfo) {
        throw "HFMA2 was not found in the line-info CUBIN."
    }
    $sassMatch = $sassMatchInfo.Matches[0]
    $sassOffsetHex = $sassMatch.Groups[1].Value.ToLowerInvariant()
    $sassOffset = [Convert]::ToInt64($sassOffsetHex, 16)
    $sassInstruction = "HFMA2 $($sassMatch.Groups[2].Value.Trim())"

    $cudaLineMatch = Find-PreviousMatch `
        $cudaSass `
        ($sassMatchInfo.LineNumber - 2) `
        'File ".*fma\.cu", line (\d+)'
    if (-not $cudaLineMatch) {
        throw "CUDA line metadata for HFMA2 was not found. Run observe.ps1 after a -lineinfo build."
    }
    $cudaLine = [int]$cudaLineMatch.Groups[1].Value
    $cudaSource = (Get-Content -LiteralPath $sourcePath)[$cudaLine - 1].Trim()

    $ptxSassMatchInfo = $ptxSass |
        Select-String -Pattern '^\s*/\*([0-9a-fA-F]+)\*/\s+HFMA2\s+(.+?)\s*;' |
        Select-Object -First 1
    $embeddedPtxMatch = Find-PreviousMatch `
        $ptxSass `
        ($ptxSassMatchInfo.LineNumber - 2) `
        'File "\.nv_debug_ptx_txt", line (\d+)'
    if (-not $embeddedPtxMatch) {
        throw "PTX line metadata for HFMA2 was not found. Run observe.ps1 after a -lineinfo build."
    }

    $ptxMatchInfo = Select-String -LiteralPath $ptxPath -SimpleMatch "fma.rn.f16x2" |
        Select-Object -First 1
    if (-not $ptxMatchInfo) {
        throw "fma.rn.f16x2 was not found in the PTX artifact."
    }
    $ptxLines = @(Get-Content -LiteralPath $ptxPath)
    $ptxLocMatch = Find-PreviousMatch `
        $ptxLines `
        ($ptxMatchInfo.LineNumber - 2) `
        '^\s*\.loc\s+(.+)$'

    $runtimeMatchInfo = Select-String `
        -LiteralPath $runtimePath `
        -Pattern '^\s*(0x[0-9a-fA-F]+)\s+(HFMA2\s+\S+,\s+\S+,\s+\S+,\s+\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)' |
        Select-Object -First 1
    if (-not $runtimeMatchInfo) {
        throw "HFMA2 with detailed SourceCounters was not found in $runtimePath."
    }
    $runtimeMatch = $runtimeMatchInfo.Matches[0]
    $runtimePcText = $runtimeMatch.Groups[1].Value.ToLowerInvariant()
    $runtimePc = [Convert]::ToInt64($runtimePcText.Substring(2), 16)
    $runtimeInstruction = $runtimeMatch.Groups[2].Value.Trim()
    $kernelBasePc = $runtimePc - $sassOffset

    if ($runtimeInstruction -ne $sassInstruction) {
        throw "Static and runtime SASS differ: '$sassInstruction' vs '$runtimeInstruction'."
    }

    $dataflow = Get-SassGprDataflow -Cubin $cubinPath -KernelName "fma_half2"
    $focusOffsets = @(0x00a0, 0x00c0, 0x00d0, 0x00f0, 0x0100)
    $focusNodes = @($dataflow.Nodes | Where-Object { $_.Offset -in $focusOffsets })
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

    $dataflowNodeLines = @($focusNodes | ForEach-Object {
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
    $dataflowEdgeLines = @($focusEdges | ForEach-Object {
        "$($_.FromText) --$($_.Register)--> $($_.ToText)"
    })

    $lines = @(
        "Kernel: fma_half2",
        "",
        "[CUDA]",
        "file: operators/fma/fma.cu",
        "line: $cudaLine",
        "source: $cudaSource",
        "",
        "[PTX]",
        "standalone line: $($ptxMatchInfo.LineNumber)",
        "loc: $($ptxLocMatch.Groups[1].Value.Trim())",
        "instruction: $($ptxMatchInfo.Line.Trim())",
        "",
        "[SASS]",
        "embedded PTX line: $($embeddedPtxMatch.Groups[1].Value)",
        "function-relative offset: 0x$sassOffsetHex",
        "instruction: $sassInstruction",
        "",
        "[SASS DATAFLOW NODES]",
        $dataflowNodeLines,
        "",
        "[DATAFLOW]",
        $dataflowEdgeLines,
        "",
        "[Runtime]",
        "source: operators/fma/runtime/fma_detailed_sass.txt",
        ("kernel base PC: 0x{0:x}" -f $kernelBasePc),
        "runtime PC: $runtimePcText",
        "instruction: $runtimeInstruction",
        "warp stall sampling (all samples): $($runtimeMatch.Groups[3].Value)",
        "warp stall sampling (not-issued samples): $($runtimeMatch.Groups[4].Value)",
        "samples: $($runtimeMatch.Groups[5].Value)",
        "instructions executed: $($runtimeMatch.Groups[6].Value)",
        "thread instructions executed: $($runtimeMatch.Groups[7].Value)",
        "predicated-on threads executed: $($runtimeMatch.Groups[8].Value)",
        "",
        "[Relations]",
        "CUDA -> PTX: line metadata (.loc/inlined_at); one-to-many or many-to-one allowed",
        "PTX -> SASS: .nv_debug_line_sass metadata; standalone and embedded PTX line numbers differ",
        "SASS -> Runtime PC: kernel base PC + function-relative offset; exact instruction matched",
        "Runtime PC -> observations: detailed SourceCounters row at the matched PC",
        "GPR dataflow: nearest preceding definition within the same kernel; register overwrite respected",
        "",
        "[Known limitation]",
        "Nsight Compute reports 'PTX source is not available'; direct report-internal PTX -> PC is unavailable.",
        "Dataflow currently covers ordinary GPR R0..Rn only; predicates, uniform/special registers,",
        "64-bit register-pair expansion, path-sensitive CFG merges, and memory alias dependencies are excluded."
    )
    $lines | Set-Content -LiteralPath $outputPath -Encoding utf8
    Write-Host "FMA correlation: $outputPath"
    exit 0
} catch {
    [Console]::Error.WriteLine("FMA correlation failed: $($_.Exception.Message)")
    exit 1
}
