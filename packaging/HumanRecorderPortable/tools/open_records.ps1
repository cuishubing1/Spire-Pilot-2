$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "storage.ps1")

$root = Get-HumanRecorderActiveRoot
$inbox = Join-Path $root "已完成记录"
if (-not (Test-Path -LiteralPath $inbox -PathType Container)) {
    $fallbackInbox = Join-Path $root "inbox"
    if (Test-Path -LiteralPath $fallbackInbox -PathType Container) { $inbox = $fallbackInbox }
    else { New-Item -ItemType Directory -Path $inbox -Force | Out-Null }
}
Start-Process explorer.exe -ArgumentList @($inbox)
