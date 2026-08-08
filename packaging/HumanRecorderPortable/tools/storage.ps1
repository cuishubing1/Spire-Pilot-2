function Get-HumanRecorderRoots {
    $portable = Join-Path (Split-Path -Parent $PSScriptRoot) "记录数据"
    $fallback = Join-Path $env:LOCALAPPDATA "SlayTheSpire2\HumanRecorder"
    @($portable, $fallback) | Select-Object -Unique
}

function Get-HumanRecorderActiveRoot {
    foreach ($root in Get-HumanRecorderRoots) {
        $health = Join-Path $root "health.json"
        if (Test-Path -LiteralPath $health -PathType Leaf) {
            try {
                $value = Get-Content -LiteralPath $health -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($value.storage_root -and (Test-Path -LiteralPath $value.storage_root -PathType Container)) {
                    return [IO.Path]::GetFullPath($value.storage_root)
                }
            }
            catch { }
            return [IO.Path]::GetFullPath($root)
        }
    }
    return [IO.Path]::GetFullPath((Get-HumanRecorderRoots)[0])
}
