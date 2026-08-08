$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "storage.ps1")

$files = foreach ($root in Get-HumanRecorderRoots) {
    foreach ($name in @("已完成记录", "inbox")) {
        $inbox = Join-Path $root $name
        if (Test-Path -LiteralPath $inbox -PathType Container) {
            Get-ChildItem -LiteralPath $inbox -Filter "human-*.jsonl" -File
        }
    }
}
$files = @($files | Group-Object Name | ForEach-Object { $_.Group | Sort-Object LastWriteTime -Descending | Select-Object -First 1 })
if ($files.Count -eq 0) {
    Write-Host "没有找到已经完成的游戏记录。" -ForegroundColor Yellow
    Write-Host "请先完成、失败或放弃至少一局游戏，再重新打包。"
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
if ([string]::IsNullOrWhiteSpace($desktop)) { $desktop = (Get-Location).Path }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$zip = Join-Path $desktop "HumanRecorder玩家记录-$stamp.zip"
$stage = Join-Path ([IO.Path]::GetTempPath()) ("HumanRecorder-export-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    foreach ($file in $files) { Copy-Item -LiteralPath $file.FullName -Destination $stage }
    $hashes = Get-ChildItem -LiteralPath $stage -Filter "*.jsonl" -File | Get-FileHash -Algorithm SHA256 |
        Select-Object Hash,@{n="File";e={Split-Path -Leaf $_.Path}}
    $hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage "checksums.json") -Encoding UTF8
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -CompressionLevel Optimal
}
finally {
    if (Test-Path -LiteralPath $stage) { [IO.Directory]::Delete($stage, $true) }
}

Write-Host ""
Write-Host "打包完成！" -ForegroundColor Green
Write-Host "共打包 $($files.Count) 局。"
Write-Host "文件已经放到桌面：$zip"
Start-Process explorer.exe -ArgumentList @("/select,`"$zip`"")
