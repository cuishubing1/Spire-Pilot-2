[CmdletBinding(SupportsShouldProcess)]
param([Parameter(Mandatory)][string]$GameRoot)

$ErrorActionPreference = "Stop"
$destination = Join-Path ([IO.Path]::GetFullPath($GameRoot)) "mods\HumanRecorder"
if (-not (Test-Path -LiteralPath $destination -PathType Container)) { Write-Host "HumanRecorder is not installed: $destination"; return }
if ($PSCmdlet.ShouldProcess($destination, "Uninstall HumanRecorder (recorded datasets are preserved)")) {
    foreach ($name in @("HumanRecorder.dll", "HumanRecorder.json", "HumanRecorder.pdb")) {
        $path = Join-Path $destination $name
        if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force }
    }
    if (-not (Get-ChildItem -LiteralPath $destination -Force)) { Remove-Item -LiteralPath $destination }
}
Write-Host "Recorded data was not removed. It remains under %LOCALAPPDATA%\SlayTheSpire2\HumanRecorder."
