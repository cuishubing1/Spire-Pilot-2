[CmdletBinding()]
param(
    [string]$GameDataDir = "",
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$dotnet = Join-Path $projectRoot ".dotnet\dotnet.exe"
if (-not (Test-Path -LiteralPath $dotnet -PathType Leaf)) { throw "Pinned dotnet not found: $dotnet" }
if ([string]::IsNullOrWhiteSpace($GameDataDir)) {
    $GameDataDir = Join-Path $projectRoot "archives\sts2-v0.107.1-build-23811903\game\data_sts2_windows_x86_64"
}
$GameDataDir = [IO.Path]::GetFullPath($GameDataDir)
$expected = "A1F9E653F1E28E4076558FEE1E60D218619CB7E057B887C6417F62C62C6D7A52"
$assembly = Join-Path $GameDataDir "sts2.dll"
if (-not (Test-Path -LiteralPath $assembly -PathType Leaf)) { throw "sts2.dll not found: $assembly" }
$actual = (Get-FileHash -LiteralPath $assembly -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw "Wrong sts2.dll. Expected $expected, got $actual" }

& $dotnet build (Join-Path $projectRoot "mods\HumanRecorder\HumanRecorder.csproj") -c $Configuration "/p:GameDataDir=$GameDataDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $dotnet run --project (Join-Path $projectRoot "tools\Sts2RecorderVerify") -- $GameDataDir (Join-Path $projectRoot "dist\HumanRecorder\HumanRecorder.dll")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-ChildItem -LiteralPath (Join-Path $projectRoot "dist\HumanRecorder") -File | Get-FileHash -Algorithm SHA256
