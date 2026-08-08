[CmdletBinding(SupportsShouldProcess)]
param([Parameter(Mandatory)][string]$GameRoot)

$ErrorActionPreference = "Stop"
$expected = "A1F9E653F1E28E4076558FEE1E60D218619CB7E057B887C6417F62C62C6D7A52"
$root = [IO.Path]::GetFullPath($GameRoot)
$gameAssembly = Join-Path $root "data_sts2_windows_x86_64\sts2.dll"
$source = Join-Path $PSScriptRoot "mod"
$destination = Join-Path $root "mods\HumanRecorder"
$requestedWhatIf = $WhatIfPreference
$WhatIfPreference = $false
try {
    if (-not (Test-Path -LiteralPath $gameAssembly -PathType Leaf)) { throw "Not a Slay the Spire 2 Windows game root: $root" }
    $actual = (Get-FileHash -LiteralPath $gameAssembly -Algorithm SHA256).Hash
    if ($actual -ne $expected) { throw "HumanRecorder requires STS2 v0.107.1/build 23811903. sts2.dll was $actual" }
    foreach ($name in @("HumanRecorder.dll", "HumanRecorder.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $source $name) -PathType Leaf)) { throw "Package is incomplete: $name" }
    }
}
finally {
    $WhatIfPreference = $requestedWhatIf
}
if ($PSCmdlet.ShouldProcess($destination, "Install HumanRecorder 0.5.3-internal")) {
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $source "HumanRecorder.dll") -Destination $destination -Force
    Copy-Item -LiteralPath (Join-Path $source "HumanRecorder.json") -Destination $destination -Force
}
Get-ChildItem -LiteralPath $destination -File -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime
