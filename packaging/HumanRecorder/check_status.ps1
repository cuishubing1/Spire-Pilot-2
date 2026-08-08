[CmdletBinding()]
param([string]$GameRoot = "")

$ErrorActionPreference = "Stop"
$health = Join-Path $env:LOCALAPPDATA "SlayTheSpire2\HumanRecorder\health.json"
$result = [ordered]@{
    HealthFile = $health
    HealthExists = Test-Path -LiteralPath $health -PathType Leaf
    Installed = $null
    GameCompatible = $null
}
if ($result.HealthExists) {
    $status = Get-Content -LiteralPath $health -Raw | ConvertFrom-Json
    $result.Status = $status.status
    $result.RecorderVersion = $status.recorder_version
    $result.RunId = $status.run_id
    $result.LastDecision = $status.last_decision_sequence
    $result.LastError = $status.last_error
    $result.WriterMetrics = $status.writer_metrics
}
if (-not [string]::IsNullOrWhiteSpace($GameRoot)) {
    $root = [IO.Path]::GetFullPath($GameRoot)
    $result.Installed = Test-Path -LiteralPath (Join-Path $root "mods\HumanRecorder\HumanRecorder.dll") -PathType Leaf
    $assembly = Join-Path $root "data_sts2_windows_x86_64\sts2.dll"
    $result.GameCompatible = (Test-Path -LiteralPath $assembly -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $assembly -Algorithm SHA256).Hash -eq "A1F9E653F1E28E4076558FEE1E60D218619CB7E057B887C6417F62C62C6D7A52")
}
[pscustomobject]$result
