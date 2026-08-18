$script:KnowledgeRepositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

function Get-KnowledgeRepositoryRoot {
    return $script:KnowledgeRepositoryRoot
}

function Resolve-KnowledgePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    if ([System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "Knowledge references must be repository-relative: $RelativePath"
    }

    $root = [System.IO.Path]::GetFullPath($script:KnowledgeRepositoryRoot)
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    $rootPrefix = $root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Knowledge reference escapes the repository: $RelativePath"
    }
    return $candidate
}

function Get-SassFeatureRecord {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Sass
    )

    if (-not (Test-Path -LiteralPath $Sass -PathType Leaf)) {
        throw "SASS artifact does not exist: $Sass"
    }

    $instructionNodes = [System.Collections.Generic.List[object]]::new()
    foreach ($line in Get-Content -LiteralPath $Sass) {
        $match = [regex]::Match(
            $line,
            '^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@(!?P\d+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;'
        )
        if (-not $match.Success) {
            continue
        }

        $instructionNodes.Add([pscustomobject][ordered]@{
            offset = "0x$($match.Groups[1].Value.ToLowerInvariant())"
            predicate = if ($match.Groups[2].Success) { $match.Groups[2].Value } else { $null }
            opcode = $match.Groups[3].Value
            operands = $match.Groups[4].Value.Trim()
            instruction = ($match.Groups[3].Value + " " + $match.Groups[4].Value.Trim()).Trim()
        })
    }

    if ($instructionNodes.Count -eq 0) {
        throw "No cuobjdump-style SASS instructions were found: $Sass"
    }

    function Get-OpcodeCount {
        param([string]$Pattern)
        return @($instructionNodes | Where-Object { $_.opcode -match $Pattern }).Count
    }

    $ffmaCount = Get-OpcodeCount '^FFMA(?:\.|$)'
    $hfma2Count = Get-OpcodeCount '^HFMA2(?:\.|$)'
    $hmmaCount = Get-OpcodeCount '^HMMA(?:\.|$)'
    $minmaxCount = Get-OpcodeCount '^[FIU]MNMX(?:\.|$)'
    $barrierCount = Get-OpcodeCount '^BAR(?:\.|$)'
    $atomicCount = Get-OpcodeCount '^(ATOM|RED)(?:\.|$)'
    $globalLoadCount = Get-OpcodeCount '^LDG(?:\.|$)'
    $globalStoreCount = Get-OpcodeCount '^STG(?:\.|$)'
    $sharedLoadCount = Get-OpcodeCount '^LDS(?:\.|$)'
    $sharedStoreCount = Get-OpcodeCount '^STS(?:\.|$)'
    $branchCount = Get-OpcodeCount '^(BRA|BRX|JMP)(?:\.|$)'
    $predicateCount = @($instructionNodes | Where-Object { $null -ne $_.predicate }).Count
    $predicatedStoreCount = @(
        $instructionNodes |
            Where-Object { $null -ne $_.predicate -and $_.opcode -match '^ST' }
    ).Count

    $hasAbsoluteModifier = @(
        $instructionNodes | Where-Object { $_.operands -match '\|(?:U?R\d+|RZ)\|' }
    ).Count -gt 0
    $hasNegatedRegister = @(
        $instructionNodes | Where-Object { $_.operands -match '(?:^|,\s*)-U?R\d+(?:\W|$)' }
    ).Count -gt 0

    $motifs = [System.Collections.Generic.List[object]]::new()
    foreach ($node in $instructionNodes) {
        if ($node.operands -match '\|(?:U?R\d+|RZ)\|') {
            $motifs.Add([pscustomobject][ordered]@{
                name = 'absolute_value_operand_modifier'
                offsets = @($node.offset)
                instructions = @($node.instruction)
            })
        }
        if ($node.opcode -match '^[FIU]MNMX(?:\.|$)' -and $node.operands -match '(?:^|,\s*)RZ(?:,|$)') {
            $motifs.Add([pscustomobject][ordered]@{
                name = 'minmax_against_zero'
                offsets = @($node.offset)
                instructions = @($node.instruction)
            })
        }
        if ($node.opcode -match '^FADD(?:\.|$)' -and $node.operands -match '(?:^|,\s*)-R\d+(?:,|$)') {
            $motifs.Add([pscustomobject][ordered]@{
                name = 'negated_register_arithmetic'
                offsets = @($node.offset)
                instructions = @($node.instruction)
            })
        }
    }

    $opcodes = @(
        $instructionNodes |
            Select-Object -ExpandProperty opcode -Unique
    )

    return [pscustomobject][ordered]@{
        instructions = $opcodes
        features = [pscustomobject][ordered]@{
            has_ffma = $ffmaCount -gt 0
            has_hfma2 = $hfma2Count -gt 0
            has_hmma = $hmmaCount -gt 0
            has_mufu_exp2 = (Get-OpcodeCount '^MUFU\.EX2(?:\.|$)') -gt 0
            has_mufu_rcp = (Get-OpcodeCount '^MUFU\.RCP(?:\.|$)') -gt 0
            has_mufu_rsqrt = (Get-OpcodeCount '^MUFU\.(RSQ|RSQRT)(?:\.|$)') -gt 0
            has_minmax = $minmaxCount -gt 0
            has_shuffle = (Get-OpcodeCount '^SHFL(?:\.|$)') -gt 0
            has_barrier = $barrierCount -gt 0
            has_atomic = $atomicCount -gt 0
            has_shared_load = $sharedLoadCount -gt 0
            has_shared_store = $sharedStoreCount -gt 0
            has_global_load = $globalLoadCount -gt 0
            has_global_store = $globalStoreCount -gt 0
            has_predicate = $predicateCount -gt 0
            has_predicated_store = $predicatedStoreCount -gt 0
            has_async_copy = (Get-OpcodeCount '^CP\.ASYNC(?:\.|$)') -gt 0
            has_absolute_operand_modifier = $hasAbsoluteModifier
            has_negated_register_operand = $hasNegatedRegister
        }
        counts = [pscustomobject][ordered]@{
            instruction_count = $instructionNodes.Count
            register_count = $null
            global_load_count = $globalLoadCount
            global_store_count = $globalStoreCount
            shared_load_count = $sharedLoadCount
            shared_store_count = $sharedStoreCount
            ffma_count = $ffmaCount
            hfma2_count = $hfma2Count
            hmma_count = $hmmaCount
            minmax_count = $minmaxCount
            barrier_count = $barrierCount
            atomic_count = $atomicCount
            branch_count = $branchCount
            predicate_count = $predicateCount
        }
        motifs = @($motifs)
        nodes = @($instructionNodes)
    }
}
