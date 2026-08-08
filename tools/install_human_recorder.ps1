[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$GameRoot
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $projectRoot "dist\HumanRecorder"
$resolvedGameRoot = [IO.Path]::GetFullPath($GameRoot)
$assembly = Join-Path $resolvedGameRoot "data_sts2_windows_x86_64\sts2.dll"
$expected = "A1F9E653F1E28E4076558FEE1E60D218619CB7E057B887C6417F62C62C6D7A52"
$requestedWhatIf = $WhatIfPreference
$WhatIfPreference = $false
try {
    if (-not (Test-Path -LiteralPath $assembly -PathType Leaf)) { throw "Not an STS2 Windows game root: $resolvedGameRoot" }
    $actual = (Get-FileHash -LiteralPath $assembly -Algorithm SHA256).Hash
    if ($actual -ne $expected) { throw "Installation refused: game is not locked v0.107.1 ($actual)" }
    foreach ($name in @("HumanRecorder.dll", "HumanRecorder.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $source $name) -PathType Leaf)) { throw "Build artifact missing: $name" }
    }
}
finally {
    $WhatIfPreference = $requestedWhatIf
}
$destination = Join-Path $resolvedGameRoot "mods\HumanRecorder"
if ($PSCmdlet.ShouldProcess($destination, "Install HumanRecorder v0.5.3-internal")) {
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $source "HumanRecorder.dll") -Destination $destination -Force
    Copy-Item -LiteralPath (Join-Path $source "HumanRecorder.json") -Destination $destination -Force
    $pdb = Join-Path $source "HumanRecorder.pdb"
    if (Test-Path -LiteralPath $pdb -PathType Leaf) { Copy-Item -LiteralPath $pdb -Destination $destination -Force }
}
if (Test-Path -LiteralPath $destination -PathType Container) {
    Get-ChildItem -LiteralPath $destination -File | Select-Object Name, Length, LastWriteTime
}
else {
    [pscustomobject]@{ Destination = $destination; Installed = $false }
}
