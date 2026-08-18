param(
    [string]$GameDir = 'D:\steam\steamapps\common\Slay the Spire 2',
    [int]$Steps = 20,
    [int]$BranchProbes = 0,
    [int]$ReplaySteps = 20,
    [int]$ReplayRepeats = 5,
    [double]$TimeoutSeconds = 10.0
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dotnet = Join-Path $repoRoot '.dotnet\dotnet.exe'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$cliRoot = Join-Path $repoRoot 'third_party\sts2-cli'
$project = Join-Path $cliRoot 'src\Sts2Headless\Sts2Headless.csproj'
$engineDll = Join-Path $cliRoot 'src\Sts2Headless\bin\Debug\net9.0\Sts2Headless.dll'
$libDir = Join-Path $cliRoot 'lib'
$gameDataDir = Join-Path $GameDir 'data_sts2_windows_x86_64'
if (Test-Path -LiteralPath (Join-Path $GameDir 'sts2.dll')) {
    $gameDataDir = $GameDir
}

foreach ($required in @($dotnet, $python, $project, $libDir, $gameDataDir)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

$env:DOTNET_EXE = $dotnet
$env:DOTNET_ROOT = Split-Path -Parent $dotnet
$env:STS2_GAME_DIR = $gameDataDir
$env:STS2_LIB = $libDir

Write-Host '[1/4] Building the v0.107.1-compatible headless engine'
& $dotnet build $project --no-restore
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed with exit code $LASTEXITCODE" }

Write-Host '[2/4] Running the stable combat and save/load regression gate'
$tests = @(
    "$cliRoot\tests\test_combat.py::TestCombatStructure",
    "$cliRoot\tests\test_combat.py::TestPlayCards",
    "$cliRoot\tests\test_combat.py::TestTurnFlow",
    "$cliRoot\tests\test_combat.py::TestCombatEnd::test_player_powers_after_enemy_debuff",
    "$cliRoot\tests\test_combat.py::TestCombatEdgeCases::test_exhaust_all_and_end_turn",
    "$cliRoot\tests\test_combat.py::TestCombatEdgeCases::test_many_cards_per_turn",
    "$cliRoot\tests\test_combat.py::TestCombatEdgeCases::test_low_hp_death",
    "$cliRoot\tests\test_save_load.py",
    "$cliRoot\tests\test_protocol.py"
)
& $python -m pytest @tests -q
if ($LASTEXITCODE -ne 0) { throw "sts2-cli regression gate failed with exit code $LASTEXITCODE" }

Write-Host '[3/4] Measuring sequential combat command latency'
$output = Join-Path $repoRoot 'artifacts\sts2_cli_benchmark.json'
& $python (Join-Path $repoRoot 'tools\benchmark_sts2_cli.py') `
    --game-dir $GameDir `
    --engine-dll $engineDll `
    --steps $Steps `
    --branch-probes $BranchProbes `
    --timeout $TimeoutSeconds `
    --output $output
if ($LASTEXITCODE -ne 0) { throw "sts2-cli benchmark failed with exit code $LASTEXITCODE" }

Write-Host '[4/4] Verifying exact combat action-prefix replay'
$replayOutput = Join-Path $repoRoot 'artifacts\sts2_cli_combat_replay.json'
& $python (Join-Path $repoRoot 'tools\benchmark_combat_replay.py') `
    --game-dir $GameDir `
    --engine-dll $engineDll `
    --prefix-steps $ReplaySteps `
    --repeats $ReplayRepeats `
    --timeout $TimeoutSeconds `
    --output $replayOutput
if ($LASTEXITCODE -ne 0) { throw "combat replay gate failed with exit code $LASTEXITCODE" }

Write-Host "Simulator gate passed. Benchmarks: $output and $replayOutput"
