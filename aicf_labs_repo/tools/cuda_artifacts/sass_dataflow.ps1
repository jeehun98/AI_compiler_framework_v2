function Get-SassGprDataflow {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Cubin,

        [Parameter(Mandatory = $true)]
        [string]$KernelName
    )

    if (-not (Test-Path -LiteralPath $Cubin -PathType Leaf)) {
        throw "CUBIN does not exist: $Cubin"
    }

    $nvdisasm = (Get-Command nvdisasm -ErrorAction Stop).Source
    $lineInfoOutput = @(& $nvdisasm --print-line-info-inline $Cubin 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvdisasm --print-line-info-inline failed with exit code $LASTEXITCODE."
    }
    $lifeRangeOutput = @(& $nvdisasm --life-range-mode count $Cubin 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvdisasm --life-range-mode count failed with exit code $LASTEXITCODE."
    }

    $liveRegistersByOffset = @{}
    $inLifeRangeKernel = $false
    foreach ($line in $lifeRangeOutput) {
        if ($line -match '^//-+\s+\.text\.(\S+)') {
            $inLifeRangeKernel = $Matches[1] -match [regex]::Escape($KernelName)
            continue
        }
        if ($inLifeRangeKernel -and
            $line -match '^\s*/\*([0-9a-fA-F]+)\*/.*?;\s*//\s*\|\s*(\d+)\s*\|') {
            $liveRegistersByOffset[$Matches[1].ToLowerInvariant()] = [int]$Matches[2]
        }
    }

    $nodes = [System.Collections.Generic.List[object]]::new()
    $sourceLocations = [System.Collections.Generic.List[string]]::new()
    $resetSourceLocationsOnMetadata = $false
    $inKernel = $false
    foreach ($line in $lineInfoOutput) {
        if ($line -match '^//-+\s+\.text\.(\S+)') {
            if ($inKernel) {
                break
            }
            $inKernel = $Matches[1] -match [regex]::Escape($KernelName)
            $sourceLocations.Clear()
            $resetSourceLocationsOnMetadata = $false
            continue
        }
        if (-not $inKernel) {
            continue
        }

        if ($line -match 'File "([^"]+)", line (\d+)(?: inlined at "([^"]+)", line (\d+))?') {
            if ($resetSourceLocationsOnMetadata) {
                $sourceLocations.Clear()
                $resetSourceLocationsOnMetadata = $false
            }
            $location = "$($Matches[1]):$($Matches[2])"
            if ($Matches[3]) {
                $location += " inlined at $($Matches[3]):$($Matches[4])"
            }
            if (-not $sourceLocations.Contains($location)) {
                $sourceLocations.Add($location)
            }
            continue
        }

        $instructionMatch = [regex]::Match(
            $line,
            '^\s*/\*([0-9a-fA-F]+)\*/\s+(.+?)\s*;'
        )
        if (-not $instructionMatch.Success) {
            continue
        }

        $offset = $instructionMatch.Groups[1].Value.ToLowerInvariant()
        $instruction = $instructionMatch.Groups[2].Value.Trim()
        $withoutPredicate = $instruction -replace '^@!?P\d+\s+', ''
        $parts = $withoutPredicate -split '\s+', 2
        $opcode = $parts[0]
        $operandText = if ($parts.Count -gt 1) { $parts[1] } else { "" }
        $operands = @($operandText -split ',' | ForEach-Object { $_.Trim() })

        $registerPattern = '(?<![A-Z0-9_])R\d+'
        $allRegisters = @(
            [regex]::Matches($operandText, $registerPattern) |
                ForEach-Object { $_.Value } |
                Select-Object -Unique
        )
        $reads = @()
        $writes = @()

        if ($opcode -match '^ST') {
            $reads = $allRegisters
        } elseif ($operands.Count -gt 0 -and
                  $operands[0] -match '^R\d+(?:\.[A-Za-z0-9]+)?$') {
            $destination = [regex]::Match($operands[0], '^R\d+').Value
            $writes = @($destination)
            $sourceOperandText = if ($operands.Count -gt 1) {
                ($operands[1..($operands.Count - 1)] -join ', ')
            } else {
                ""
            }
            $reads = @(
                [regex]::Matches($sourceOperandText, $registerPattern) |
                    ForEach-Object { $_.Value } |
                    Select-Object -Unique
            )

            if ($opcode -match '^LD' -and $operandText -match '^R\d+(?:\.[A-Za-z0-9]+)?\s*,\s*\[') {
                $addressText = $operandText.Substring($operandText.IndexOf('['))
                $reads = @(
                    [regex]::Matches($addressText, $registerPattern) |
                        ForEach-Object { $_.Value } |
                        Select-Object -Unique
                )
            }
        } else {
            $reads = $allRegisters
        }

        $nodes.Add([pscustomobject]@{
            Offset = [Convert]::ToInt64($offset, 16)
            OffsetText = "0x$offset"
            Opcode = $opcode
            Instruction = $instruction
            Reads = @($reads)
            Writes = @($writes)
            SourceLocations = @($sourceLocations)
            LiveRegisters = $liveRegistersByOffset[$offset]
        })
        $resetSourceLocationsOnMetadata = $true
    }

    if ($nodes.Count -eq 0) {
        throw "No SASS instructions were found for kernel '$KernelName'."
    }

    $edges = [System.Collections.Generic.List[object]]::new()
    $lastDefinition = @{}
    foreach ($node in $nodes) {
        foreach ($register in $node.Reads) {
            if ($lastDefinition.ContainsKey($register)) {
                $producer = $lastDefinition[$register]
                $edges.Add([pscustomobject]@{
                    Register = $register
                    From = $producer.Offset
                    FromText = $producer.OffsetText
                    To = $node.Offset
                    ToText = $node.OffsetText
                })
            }
        }
        foreach ($register in $node.Writes) {
            $lastDefinition[$register] = $node
        }
    }

    return [pscustomobject]@{
        Kernel = $KernelName
        Nodes = @($nodes)
        Edges = @($edges)
    }
}
