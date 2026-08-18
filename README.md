# Spire Pilot 2

面向《杀戮尖塔 2》的非 LLM、结构化状态决策项目。整体方向由两部分组成：

- **战斗 Agent**：从玩家可见的结构化战斗状态出发，学习合法动作策略、战斗价值与风险。
- **局外 World Model / Planner**：建模抓牌、路线、商店、事件和资源选择的长期影响，并据此规划。

当前最成熟的部分是数据基础设施：`HumanRecorder` 人工对局采集 Mod、数据
Schema、版本门禁、审计与导出工具。战斗 Agent 和局外 World Model 尚处于设计与
基线建设阶段，不能将规划中的模型能力表述为已经实现。

> 本仓库只保存项目原创源码、测试、Schema 与构建脚本，不包含《杀戮尖塔 2》
> 游戏本体、反编译产物、玩家对局数据或预编译 Mod。使用前需要合法拥有游戏，
> 并在本机配置游戏目录。普通玩家安装说明见
> [`mods/HumanRecorder/README.md`](mods/HumanRecorder/README.md)。

## Repository layout

```text
Spire-Pilot-2/
├─ agents/combat/       # 战斗 Agent：策略、价值、风险与搜索（建设中）
├─ world_model/         # 局外 World Model 与长期规划（设计中）
├─ mods/HumanRecorder/  # 人工对局数据采集 Mod
├─ src/sts2_dataset/    # 数据导入、规范化、审计与导出
├─ schemas/             # 原始记录与健康状态 Schema
├─ config/              # 版本锁定与数据集配置
├─ tools/               # 构建、验证和诊断工具
├─ tests/               # 自动化测试
└─ docs/                # 项目设计文档
```

详细项目边界与阶段目标见
[`PROJECT_HANDOFF.zh-CN.md`](PROJECT_HANDOFF.zh-CN.md)。玩家原始对局数据属于
私有、不可变证据，不纳入公开仓库。

## Data foundation

Reproducible data preparation for **Slay the Spire 2 v0.107.1** using the real
game engine through a pinned `sts2-cli` adapter.

The pipeline deliberately separates two views:

- `agent_observation`: a strict allow-list of player-visible information.
- `audit`: compressed engine checkpoints and deterministic replay prefixes.

## Bootstrap

```powershell
& .\.dotnet\dotnet.exe --version
& C:\ProgramData\miniconda3\python.exe -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\sts2-data.exe setup-engine
& .\.venv\Scripts\sts2-data.exe lock-environment
& .\.venv\Scripts\sts2-data.exe smoke
```

`smoke` is a hard gate. Collection refuses to start if the game version, DLL
hash, engine protocol, checkpoint, or deterministic restore checks fail.

## sts2-cli simulator feasibility gate

The pinned v0.107.1 headless adapter can now be built, regression-tested and
benchmarked with one command:

```powershell
.\tools\test_sts2_cli.cmd
```

This currently validates real-engine sequential combat actions and room-level
checkpoint recovery. It does **not** claim action-level state cloning or
MCTS-ready branching. See
[`docs/sts2-cli-simulator-gate.zh-CN.md`](docs/sts2-cli-simulator-gate.zh-CN.md)
for the measured throughput and known timeout paths.

## Collect and export

For a formal build, use the unified gated pipeline:

```powershell
& .\.venv\Scripts\sts2-data.exe pipeline --runs 20
```

The equivalent individual stages are:

```powershell
& .\.venv\Scripts\sts2-data.exe collect --runs 20
& .\.venv\Scripts\sts2-data.exe fixtures
& .\.venv\Scripts\sts2-data.exe export
& .\.venv\Scripts\sts2-data.exe validate --acceptance
```

Raw JSONL is the source of truth. Parquet files are derived only from sealed
raw runs. Fixture records are isolated and never enter the default training
partition.

## Private version archive

Create and verify a complete private copy of the locked game build:

```powershell
& .\.venv\Scripts\sts2-data.exe archive-game
& .\.venv\Scripts\sts2-data.exe verify-archive archives\sts2-v0.107.1-build-23811903
& .\.venv\Scripts\sts2-data.exe test-archive-replay archives\sts2-v0.107.1-build-23811903
```

After the Steam installation updates, use the pinned archive directly:

```powershell
& .\.venv\Scripts\sts2-data.exe --config config\dataset_v1_archive.json smoke
```

The archive contains a sanitized Steam build/depot fingerprint and a SHA-256
entry for every game file. It is for private personal debugging only and is
excluded from source control.

## Important limitations

The upstream engine checkpoint rolls an in-progress room back to its entrance,
and loading a v0.107.1 map save can perturb hidden RNG used to resolve an
`Unknown` node. Every audit checkpoint therefore stores the complete
deterministic action prefix. Exact decision restoration replays from the locked
seed; the native save remains an internal-state audit artifact, not an exact RNG
restore mechanism.

## HumanRecorder: record normal play

`mods/HumanRecorder` is a no-assets, no-BaseLib observation mod locked to
v0.107.1/build 23811903. Version 0.5.3 hooks semantic game actions rather than
mouse or keyboard events and writes one crash-safe JSONL stream per logical
climb. It does not read Steam identity and generates a local anonymous actor id.

The recorder action surface covers map choice, card play/end turn, potion use and
discard, events, rest sites, shop purchases/removal/leave, card rewards and
multi-card selection, starter bundle choice, room rewards, and treasure choices.
Dedicated relic-choice screens (including boss rewards) are also covered. Only
committed semantic choices are recorded; hover and inspection are excluded.

Save/load no longer starts a new raw file. Every reload increments `attempt_id`
and emits either a resolved `rollback` or a quarantined `resume_unmatched` event.
The matcher degrades through exact state, semantic state and room-entry anchors;
location-only or unmatched boundaries are never exported as ordinary dynamics.

Build and run the static v0.107.1 hook gate:

```powershell
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .\tools\build_human_recorder.ps1
```

The installable files are staged in `dist\HumanRecorder`. Installation is an
explicit operation; neither the build nor tests change the game or the private
archive:

```powershell
# Preview first.
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .\tools\install_human_recorder.ps1 -GameRoot "D:\path\to\Slay the Spire 2" -WhatIf

# Install after checking the resolved destination.
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .\tools\install_human_recorder.ps1 -GameRoot "D:\path\to\Slay the Spire 2"
```

The installer refuses any `sts2.dll` other than the locked v0.107.1 hash. Start
the game normally and enable `Human Dataset Recorder` in the mod list. A sealed
run is written beside the installed Mod by default:

```text
<GameRoot>\mods\HumanRecorder\记录数据\human-*.jsonl
```

Set `STS2_HUMAN_RECORDER_DIR` before starting the game to use another inbox.
While a climb is active the suffix is `.jsonl.partial`; every line is flushed to
disk and protected by a SHA-256 hash chain. Quitting the process leaves it active
and the next `Continue` reopens and verifies it before appending. Only game
end/abandon atomically renames it to `.jsonl`. For an unrecoverable partial file,
keep the original evidence and create a sealed copy:

```powershell
& .\.venv\Scripts\sts2-data.exe recover-human "C:\path\run.jsonl.partial"
```

Import sealed recordings and validate the derived human partition:

```powershell
# Audit a new sealed or still-active recording without changing it.
& .\.venv\Scripts\sts2-data.exe audit-human "C:\path\human-run.jsonl.partial"

& .\.venv\Scripts\sts2-data.exe import-human "$env:LOCALAPPDATA\SlayTheSpire2\HumanRecorder\inbox"
& .\.venv\Scripts\sts2-data.exe validate-human
& .\.venv\Scripts\sts2-data.exe build-combat-dataset
& .\.venv\Scripts\sts2-data.exe validate-combat-dataset
& .\.venv\Scripts\sts2-data.exe build-combat-examples
& .\.venv\Scripts\sts2-data.exe validate-combat-examples
& .\.venv\Scripts\sts2-data.exe build-combat-vocab
```

`audit-human` reports action and phase counts plus any missing stable action
arguments or legal-action mismatches. A nonzero exit code means the recording
must remain quarantined until the relevant hook is repaired.

The importer verifies the hash chain, game fingerprint and seal before writing
`data\human\raw\*.jsonl.zst` plus `data\human\dataset\episodes.parquet` and
`transitions.parquet` plus `rollbacks.parquet`. Transitions expose
`attempt_id`, `is_canonical`, `sl_contaminated`, `termination` and
`boundary_status`; rollback boundaries always terminate a transition rather
than linking back to the restored state. Version 0.2.1 also records loaded
content mods and entity assembly provenance. Use `is_training_eligible=true`
to keep complete canonical base-content transitions, or the stricter
`strict_vanilla_eligible=true` to require a verified no-content-Mod process.
Partial and detected Mod-content decisions remain auditable in Parquet but are
isolated from both training views. Pass `--reject-partial` when an import should
fail instead of isolating incomplete decisions.

`import-human` is append-only and idempotent. It indexes every sealed run by
`run_id` and source SHA-256, fully parses only new recordings, preserves the
immutable compressed raw copy, and rejects a reused `run_id` whose bytes differ.
Existing aggregate Parquet files are atomically replaced after new rows are
appended.

`build-combat-dataset` derives the Ironclad combat-policy view under
`data/human/combat_v1/`. A combat is identified by its run and room locator, so
all actions from one combat—including canonical actions after a reload—remain in
one split. Test is an explicit full-run holdout: every combat from a configured
test `run_id` stays in test. After those runs are removed, train and validation
are formed at combat level, separately within Act 1, Act 2 and Act 3. New
non-test combats fill the per-Act train/validation deficits without moving
existing assignments. Use `--rebuild` only when the split configuration or
derivation contract intentionally changes.

`build-combat-examples` projects the audited combat rows into the frozen
Combat Observation V0 and dynamic-candidate Combat Action V0 contracts. It
removes engine/source/localization metadata, retains opaque instance references
only for binding actions to hand cards, potions and enemies, and exports the
unique human legal-candidate index as the behavior-cloning label. The output is
incremental and lives under `data/human/combat_v1/model_v0/`.

`build-combat-vocab` scans only the training split and creates an append-stable
entity vocabulary. Card identity is shared across hand/draw/discard/exhaust
zones, while zone type remains a separate feature. The NumPy tensorizer emits
variable-length entity sets, dynamic legal-action references and masks; it does
not require PyTorch and can be wrapped by the training DataLoader.

`CombatPolicyTransformer` is the first executable policy baseline: a shared
4-layer, 128-dimensional entity Transformer followed by a scorer over the
current legal candidates. The scorer binds each candidate to its source and
target entity and masks padded candidates before behavior-cloning loss. PyTorch
is kept as the optional `train` dependency so data preparation remains light.
The model uses only fields projected by Combat Observation V0; its first
full-data offline baseline and limitations are documented in
[`docs/combat-policy-v0.zh-CN.md`](docs/combat-policy-v0.zh-CN.md).

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[train]"
& .\.venv\Scripts\sts2-train.exe
```

Recorder 0.5.3-internal stores audit-only run provenance under `run_start.run_context`:
the original and numeric seed, character IDs, ascension, game mode, act IDs,
modifiers, badges, save mode and daily timestamp. Human dataset schema 0.3.0
preserves these as episode columns while keeping seed out of transition
observations and other player-visible training inputs.

Recorder 0.5.3-internal also follows the native model/save layout. Enchantments and
afflictions are nested under their card instance; energy/star costs preserve native
modifier lifetimes; and relics separate visible counters, engine `SavedProperties`,
and versioned turn/combat runtime state. Every decision has a sibling `audit_state`
containing native serialized model state and ordered combat piles. The v0.4 importer
requires that audit view to be complete while guaranteeing it is not copied into
`observation_json`. `relic_trigger_observed` engine events record reliable native
`RelicModel.Flash` signals without claiming that every flash produced a particular
effect.
