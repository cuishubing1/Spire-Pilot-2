[CmdletBinding()]
param([string]$Configuration = "Release")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "build_human_recorder.ps1") -Configuration $Configuration
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$version = "0.5.3-internal"
$distRoot = [IO.Path]::GetFullPath((Join-Path $root "dist"))
$stage = [IO.Path]::GetFullPath((Join-Path $distRoot "HumanRecorder-$version"))
if (-not $stage.StartsWith($distRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe package staging path: $stage"
}
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $stage "mod") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $root "dist\HumanRecorder\HumanRecorder.dll") -Destination (Join-Path $stage "mod")
Copy-Item -LiteralPath (Join-Path $root "dist\HumanRecorder\HumanRecorder.json") -Destination (Join-Path $stage "mod")
Get-ChildItem -LiteralPath (Join-Path $root "packaging\HumanRecorder") -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $stage }
$release = [ordered]@{
    package_version = $version
    recorder_schema = "human-live-0.4.1"
    game_version = "0.107.1"
    steam_build = "23811903"
    sts2_sha256 = "A1F9E653F1E28E4076558FEE1E60D218619CB7E057B887C6417F62C62C6D7A52"
    runtime_dependencies = @("sts2.dll", "0Harmony.dll", ".NET 9 game runtime")
    external_mod_dependencies = @()
    generated_at = (Get-Date).ToUniversalTime().ToString("O")
}
$release | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stage "release-manifest.json") -Encoding UTF8
$hashes = Get-ChildItem -LiteralPath $stage -Recurse -File | Get-FileHash -Algorithm SHA256 |
    Select-Object Hash,@{n="File";e={$_.Path.Substring($stage.Length + 1)}}
$hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage "checksums.json") -Encoding UTF8
$zip = Join-Path $root "dist\HumanRecorder-$version.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
[pscustomobject]@{Package=$zip;Bytes=(Get-Item $zip).Length;SHA256=(Get-FileHash $zip -Algorithm SHA256).Hash}
