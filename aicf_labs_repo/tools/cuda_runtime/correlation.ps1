function Get-SassSourceMapping {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$SassPath,
        [Parameter(Mandatory = $true)]
        $Dataflow,
        [Parameter(Mandatory = $true)]
        [string]$Opcode
    )

    $escapedOpcode = [regex]::Escape($Opcode)
    $sassMatchInfo = Select-String `
        -LiteralPath $SassPath `
        -Pattern "^\s*/\*([0-9a-fA-F]+)\*/\s+($escapedOpcode(?:\s+.+?)?)\s*;" |
        Select-Object -First 1
    if (-not $sassMatchInfo) {
        throw "$Opcode was not found in the canonical SASS artifact: $SassPath"
    }

    $sassMatch = $sassMatchInfo.Matches[0]
    $sassOffsetHex = $sassMatch.Groups[1].Value.ToLowerInvariant()
    $sassOffset = [Convert]::ToInt64($sassOffsetHex, 16)
    $sassInstruction = $sassMatch.Groups[2].Value.Trim()
    $sassNode = $Dataflow.Nodes |
        Where-Object { $_.Offset -eq $sassOffset } |
        Select-Object -First 1
    if (-not $sassNode) {
        throw "CUBIN line metadata does not contain canonical SASS offset 0x$sassOffsetHex."
    }
    if ($sassNode.Instruction -ne $sassInstruction) {
        throw "Canonical SASS and CUBIN line-info SASS differ: '$sassInstruction' vs '$($sassNode.Instruction)'."
    }

    $sourceFileName = [System.IO.Path]::GetFileName($SourcePath)
    $sourcePattern = '[\\/]' + [regex]::Escape($sourceFileName) + ':(\d+)'
    $sourceLines = @($sassNode.SourceLocations | ForEach-Object {
        $match = [regex]::Match($_, $sourcePattern)
        if ($match.Success) {
            [int]$match.Groups[1].Value
        }
    } | Select-Object -Unique)
    if ($sourceLines.Count -ne 1) {
        throw "Expected one $sourceFileName source line for $Opcode, found: $($sourceLines -join ', ')."
    }

    $sourceLine = $sourceLines[0]
    $sourceText = @(Get-Content -LiteralPath $SourcePath)
    if ($sourceLine -lt 1 -or $sourceLine -gt $sourceText.Count) {
        throw "CUBIN line metadata points outside the CUDA source: ${sourceFileName}:$sourceLine."
    }

    return [pscustomobject]@{
        SourceLine = $sourceLine
        SourceText = $sourceText[$sourceLine - 1].Trim()
        SassOffset = $sassOffset
        SassOffsetHex = $sassOffsetHex
        SassInstruction = $sassInstruction
    }
}

function Get-RuntimeCounters {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RuntimePath,
        [Parameter(Mandatory = $true)]
        $SassMapping,
        [Parameter(Mandatory = $true)]
        [string]$Opcode
    )

    $escapedOpcode = [regex]::Escape($Opcode)
    $runtimeMatchInfo = Select-String `
        -LiteralPath $RuntimePath `
        -Pattern "^\s*(0x[0-9a-fA-F]+)\s+($escapedOpcode(?:\s+.*?)?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)" |
        Select-Object -First 1
    if (-not $runtimeMatchInfo) {
        throw "$Opcode with detailed SourceCounters was not found in $RuntimePath."
    }

    $runtimeMatch = $runtimeMatchInfo.Matches[0]
    $runtimePcText = $runtimeMatch.Groups[1].Value.ToLowerInvariant()
    $runtimePc = [Convert]::ToInt64($runtimePcText.Substring(2), 16)
    $runtimeInstruction = $runtimeMatch.Groups[2].Value.Trim()
    if ($runtimeInstruction -ne $SassMapping.SassInstruction) {
        throw "Static and runtime SASS differ: '$($SassMapping.SassInstruction)' vs '$runtimeInstruction'."
    }

    return [pscustomobject]@{
        KernelBasePc = $runtimePc - $SassMapping.SassOffset
        RuntimePcText = $runtimePcText
        Instruction = $runtimeInstruction
        WarpStallSamples = $runtimeMatch.Groups[3].Value
        WarpStallNotIssuedSamples = $runtimeMatch.Groups[4].Value
        Samples = $runtimeMatch.Groups[5].Value
        InstructionsExecuted = $runtimeMatch.Groups[6].Value
        ThreadInstructionsExecuted = $runtimeMatch.Groups[7].Value
        PredicatedOnThreadsExecuted = $runtimeMatch.Groups[8].Value
    }
}
