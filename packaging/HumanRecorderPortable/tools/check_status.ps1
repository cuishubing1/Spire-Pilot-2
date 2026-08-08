$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "storage.ps1")

$root = Get-HumanRecorderActiveRoot
$health = Join-Path $root "health.json"
Write-Host ""
Write-Host "HumanRecorder 状态检查" -ForegroundColor Cyan
Write-Host "记录位置：$root"
if (-not (Test-Path -LiteralPath $health -PathType Leaf)) {
    Write-Host "状态：未找到状态文件" -ForegroundColor Yellow
    Write-Host "请启动游戏、新开一局并完成一个选择，然后再检查。"
    exit 1
}

$value = Get-Content -LiteralPath $health -Raw -Encoding UTF8 | ConvertFrom-Json
$statusText = switch ($value.status) {
    "recording" { "正在记录" }
    "ready" { "等待新游戏" }
    "disabled" { "记录器已停止" }
    default { [string]$value.status }
}
$color = if ($value.status -eq "disabled") { "Red" } else { "Green" }
Write-Host "状态：$statusText" -ForegroundColor $color
Write-Host "版本：$($value.recorder_version)"
if ($value.last_decision_sequence -ne $null) { Write-Host "最近记录编号：$($value.last_decision_sequence)" }
if ($value.last_error) { Write-Host "错误：$($value.last_error)" -ForegroundColor Red }
else { Write-Host "错误：无" -ForegroundColor Green }
if ($value.storage_mode -eq "local_app_data_fallback") {
    Write-Host "提示：游戏目录无法写入，数据已自动保存到备用位置。" -ForegroundColor Yellow
}
