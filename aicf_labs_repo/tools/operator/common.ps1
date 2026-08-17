$script:OperatorRepositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

function Get-OperatorLayout {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Operator
    )

    if ($Operator -notmatch '^[A-Za-z0-9_-]+$') {
        throw "Operator must contain only letters, numbers, underscores, or hyphens: $Operator"
    }

    $operatorDirectory = Join-Path $script:OperatorRepositoryRoot "operators/$Operator"
    $sourcePath = Join-Path $operatorDirectory "$Operator.cu"
    if (-not (Test-Path -LiteralPath $operatorDirectory -PathType Container)) {
        throw "Operator directory does not exist: $operatorDirectory"
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Primary CUDA source does not exist: $sourcePath"
    }

    $sourceFiles = @($sourcePath) + @(
        Get-ChildItem -LiteralPath $operatorDirectory -Filter "*.cu" -File |
            Where-Object { $_.FullName -ne $sourcePath } |
            Sort-Object Name |
            Select-Object -ExpandProperty FullName
    )

    return [pscustomobject]@{
        Name = $Operator
        RepositoryRoot = $script:OperatorRepositoryRoot
        Directory = $operatorDirectory
        Source = $sourcePath
        Sources = $sourceFiles
        BuildDirectory = Join-Path $operatorDirectory "build"
        Executable = Join-Path $operatorDirectory "build/$Operator.exe"
        ArtifactDirectory = Join-Path $operatorDirectory "artifacts"
        RuntimeDirectory = Join-Path $operatorDirectory "runtime"
    }
}

function Format-OperatorCommand {
    param([string]$Executable, [string[]]$Arguments)

    $items = @($Executable) + $Arguments
    return (($items | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' ')
}
