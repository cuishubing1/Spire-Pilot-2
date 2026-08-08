[CmdletBinding()]
param([string]$Configuration = "Release")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "build_human_recorder.ps1") -Configuration $Configuration
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$version = "0.5.3-internal"
$distRoot = [IO.Path]::GetFullPath((Join-Path $root "dist"))
$stage = [IO.Path]::GetFullPath((Join-Path $distRoot "HumanRecorder-$version-portable"))
if (-not $stage.StartsWith($distRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe package staging path: $stage"
}
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
$modStage = Join-Path $stage "HumanRecorder"
New-Item -ItemType Directory -Path $modStage -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $root "dist\HumanRecorder\HumanRecorder.dll") -Destination $modStage
Copy-Item -LiteralPath (Join-Path $root "dist\HumanRecorder\HumanRecorder.json") -Destination $modStage
Copy-Item -Path (Join-Path $root "packaging\HumanRecorderPortable\*") -Destination $modStage -Recurse

$dllHash = (Get-FileHash -LiteralPath (Join-Path $modStage "HumanRecorder.dll") -Algorithm SHA256).Hash
@(
    "HumanRecorder $version",
    "支持游戏版本：v0.107.1 / Steam build 23811903",
    "HumanRecorder.dll SHA-256：$dllHash",
    "不包含游戏本体文件，不依赖其他 Mod。"
) | Set-Content -LiteralPath (Join-Path $modStage "版本信息.txt") -Encoding UTF8

$zip = Join-Path $distRoot "HumanRecorder-$version-portable.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $modStage -DestinationPath $zip -CompressionLevel Optimal
[pscustomobject]@{
    Package = $zip
    Bytes = (Get-Item -LiteralPath $zip).Length
    SHA256 = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
    DllSHA256 = $dllHash
}
