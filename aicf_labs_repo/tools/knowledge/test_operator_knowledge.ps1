$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$validator = Join-Path $PSScriptRoot 'validate_operator_metadata.ps1'
$indexBuilder = Join-Path $PSScriptRoot 'build_operator_index.ps1'
$schema = Join-Path $repositoryRoot 'knowledge/schemas/operator.schema.json'
$baseline = Join-Path $repositoryRoot 'operators/abs/operator.json'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("aicf-knowledge-test-" + [guid]::NewGuid().ToString('N'))

function Assert-ExitCode {
    param([int]$Actual, [int]$Expected, [string]$Description)
    if ($Actual -ne $Expected) {
        throw "$Description expected exit code $Expected but got $Actual."
    }
    Write-Host "PASS: $Description"
}

function Invoke-InvalidMetadataCase {
    param(
        [string]$Name,
        [scriptblock]$Mutate
    )

    $caseDirectory = Join-Path $testRoot ("$Name/abs")
    New-Item -ItemType Directory -Force -Path $caseDirectory | Out-Null
    $document = Get-Content -Raw -LiteralPath $baseline | ConvertFrom-Json
    & $Mutate $document
    $casePath = Join-Path $caseDirectory 'operator.json'
    $json = $document | ConvertTo-Json -Depth 100
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($casePath, $json + [Environment]::NewLine, $utf8)

    & $validator -Path $casePath *> $null
    Assert-ExitCode -Actual $LASTEXITCODE -Expected 1 -Description $Name
}

try {
    New-Item -ItemType Directory -Force -Path $testRoot | Out-Null

    & $validator *> $null
    Assert-ExitCode -Actual $LASTEXITCODE -Expected 0 -Description 'valid metadata set'

    Invoke-InvalidMetadataCase 'unary commutative false is rejected' {
        param($document)
        $document.algebra.commutative = $false
    }
    Invoke-InvalidMetadataCase 'operator-level hardware flag is rejected' {
        param($document)
        $document | Add-Member -NotePropertyName uses_tensor_core -NotePropertyValue $false
    }
    Invoke-InvalidMetadataCase 'observed properties require evidence' {
        param($document)
        $document.implementations[0].hardware.observed.evidence = @()
    }
    Invoke-InvalidMetadataCase 'SASS disagreement is rejected' {
        param($document)
        $document.implementations[0].sass.observed.features.has_hmma = $true
    }
    Invoke-InvalidMetadataCase 'missing referenced artifact is rejected' {
        param($document)
        $document.implementations[0].source.file = 'operators/abs/missing.cu'
    }
    Invoke-InvalidMetadataCase 'runtime values require runtime artifacts' {
        param($document)
        foreach ($property in $document.implementations[0].measurements.artifacts.PSObject.Properties) {
            $property.Value = $null
        }
    }

    $testJsonCommand = Get-Command Test-Json -ErrorAction SilentlyContinue
    if ($null -ne $testJsonCommand) {
        foreach ($metadata in Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'operators') -Filter operator.json -File -Recurse) {
            $valid = Get-Content -Raw -LiteralPath $metadata.FullName | Test-Json -SchemaFile $schema
            if (-not $valid) {
                throw "Test-Json rejected $($metadata.FullName)."
            }
        }
        Write-Host 'PASS: platform JSON Schema validation'
    } else {
        Write-Host 'SKIP: Test-Json is unavailable in this Windows PowerShell version.'
    }

    $scripts = @(
        'tools/operator/common.ps1',
        'tools/operator/build.ps1',
        'tools/operator/run.ps1',
        'tools/operator/measure.ps1',
        'tools/operator/observe.ps1',
        'tools/knowledge/common.ps1',
        'tools/knowledge/extract_sass_features.ps1',
        'tools/knowledge/validate_operator_metadata.ps1',
        'tools/knowledge/build_operator_index.ps1'
    )
    foreach ($relativeScript in $scripts) {
        $tokens = $null
        $errors = $null
        $scriptPath = Join-Path $repositoryRoot $relativeScript
        [System.Management.Automation.Language.Parser]::ParseFile(
            $scriptPath,
            [ref]$tokens,
            [ref]$errors
        ) | Out-Null
        if ($errors.Count -ne 0) {
            throw "PowerShell parser rejected $relativeScript`: $($errors[0].Message)"
        }
    }
    Write-Host 'PASS: existing operator workflow and knowledge scripts parse'

    & $indexBuilder *> $null
    Assert-ExitCode -Actual $LASTEXITCODE -Expected 0 -Description 'operator index regeneration'

    Write-Host 'Operator knowledge tests passed.'
    exit 0
} catch {
    [Console]::Error.WriteLine("Operator knowledge tests failed: $($_.Exception.Message)")
    exit 1
} finally {
    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/')
    $resolvedTestRoot = [System.IO.Path]::GetFullPath($testRoot)
    if ((Test-Path -LiteralPath $resolvedTestRoot -PathType Container) -and
        $resolvedTestRoot.StartsWith($tempRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
