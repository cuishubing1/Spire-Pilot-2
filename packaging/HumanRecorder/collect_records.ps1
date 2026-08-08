[CmdletBinding()]
param(
    [string]$Inbox = "$env:LOCALAPPDATA\SlayTheSpire2\HumanRecorder\inbox",
    [string]$Destination = "."
)

$ErrorActionPreference = "Stop"
$source = [IO.Path]::GetFullPath($Inbox)
$targetRoot = [IO.Path]::GetFullPath($Destination)
$files = Get-ChildItem -LiteralPath $source -Filter "human-*.jsonl" -File | Sort-Object LastWriteTime
if (-not $files) { throw "No sealed HumanRecorder .jsonl files found in $source" }
New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stage = Join-Path $targetRoot "HumanRecorder-$stamp"
New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($file in $files) { Copy-Item -LiteralPath $file.FullName -Destination $stage }
$hashes = Get-ChildItem -LiteralPath $stage -File | Get-FileHash -Algorithm SHA256 |
    Select-Object Hash,@{n="File";e={Split-Path -Leaf $_.Path}}
$hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage "checksums.json") -Encoding UTF8
$zip = Join-Path $targetRoot "HumanRecorder-$stamp.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
[pscustomobject]@{Archive=$zip;Runs=$files.Count;Bytes=(Get-Item $zip).Length;SourcePreserved=$true}
