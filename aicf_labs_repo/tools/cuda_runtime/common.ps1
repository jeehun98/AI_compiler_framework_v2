function Format-NativeCommand {
    param([string]$Command, [string[]]$CommandArguments)

    $items = @($Command) + $CommandArguments
    return (($items | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' ')
}

function Get-NsightComputeCli {
    $command = Get-Command ncu -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $installedCli = Get-ChildItem `
        -Path "C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.exe" `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $installedCli) {
        throw "Nsight Compute CLI (ncu) was not found on PATH or under C:\Program Files\NVIDIA Corporation\Nsight Compute *."
    }
    return $installedCli
}
