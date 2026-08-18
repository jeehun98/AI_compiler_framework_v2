param(
    [Parameter(Mandatory = $true)]
    [string]$Sass,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

try {
    $sassPath = (Resolve-Path -LiteralPath $Sass).Path
    $record = Get-SassFeatureRecord -Sass $sassPath
    $json = $record | ConvertTo-Json -Depth 12

    if ([string]::IsNullOrWhiteSpace($Output)) {
        Write-Output $json
    } else {
        $outputPath = [System.IO.Path]::GetFullPath($Output)
        $outputDirectory = Split-Path -Parent $outputPath
        if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
            New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
        }
        $utf8 = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($outputPath, $json + [Environment]::NewLine, $utf8)
        Write-Host "SASS features: $outputPath"
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("SASS feature extraction failed: $($_.Exception.Message)")
    exit 1
}
