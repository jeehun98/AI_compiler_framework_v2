param(
    [string[]]$Path,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

try {
    $repositoryRoot = Get-KnowledgeRepositoryRoot
    $metadataPaths = @()
    if ($null -eq $Path -or $Path.Count -eq 0) {
        $metadataPaths = @(
            Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'operators') `
                -Filter 'operator.json' -File -Recurse |
                Sort-Object FullName |
                Select-Object -ExpandProperty FullName
        )
    } else {
        foreach ($candidate in $Path) {
            $metadataPaths += (Resolve-Path -LiteralPath $candidate).Path
        }
        $metadataPaths = @($metadataPaths | Sort-Object)
    }

    if ($metadataPaths.Count -eq 0) {
        throw 'No operator metadata files were found.'
    }

    $validator = Join-Path $PSScriptRoot 'validate_operator_metadata.ps1'
    & $validator -Path $metadataPaths
    if ($LASTEXITCODE -ne 0) {
        throw "Metadata validation failed with exit code $LASTEXITCODE."
    }

    $entries = [System.Collections.Generic.List[object]]::new()
    foreach ($metadataPath in $metadataPaths) {
        $document = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
        $implementationNames = @(
            $document.implementations | Select-Object -ExpandProperty name
        )
        $relativeMetadataPath = "operators/$($document.identity.name)/operator.json"
        $entries.Add([pscustomobject][ordered]@{
            name = $document.identity.name
            category = $document.identity.category
            metadata = $relativeMetadataPath
            implementations = $implementationNames
        })
    }

    $index = [pscustomobject][ordered]@{
        schema_version = 1
        schema = 'knowledge/schemas/operator.schema.json'
        operators = @($entries | Sort-Object name)
    }
    $json = $index | ConvertTo-Json -Depth 8
    $outputPath = if ([string]::IsNullOrWhiteSpace($Output)) {
        Join-Path $repositoryRoot 'knowledge/index.json'
    } else {
        [System.IO.Path]::GetFullPath($Output)
    }
    $outputDirectory = Split-Path -Parent $outputPath
    if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    }
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($outputPath, $json + [Environment]::NewLine, $utf8)
    Write-Host "Operator index: $outputPath"
    exit 0
} catch {
    [Console]::Error.WriteLine("Operator index generation failed: $($_.Exception.Message)")
    exit 1
}
