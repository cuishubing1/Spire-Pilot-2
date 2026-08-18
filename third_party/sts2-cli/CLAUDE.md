# CLAUDE.md

## Testing Requirements

Any code change MUST pass a full regression test before claiming completion:

```bash
# Run 5 games per character, ALL must complete (0 crashes/stuck)
for char in Ironclad Silent Defect Regent Necrobinder; do
    STS2_GAME_DIR="$HOME/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/data_sts2_macos_arm64" python3 python/play_full_run.py 5 "$char" 2>&1 | grep -E "Wins|Completed"
do
```

Expected: `Completed: 5/5` for every character.

## Localization

- Always use the game's official Chinese translations (from `localization_zhs/`)
- Never invent translations — look them up
- All user-facing strings must go through `t(en, zh)` for bilingual support
- Template variables like `{Damage}`, `{Block}`, `{MaxHp}` must be resolved to actual values before display

## Build

```bash
~/.dotnet-arm64/dotnet build src/Sts2Headless/Sts2Headless.csproj
```

## Key Architecture

- `src/Sts2Headless/RunSimulator.cs` — game lifecycle, decision point detection, state serialization
- `src/Sts2Headless/Program.cs` — JSON command router
- `src/GodotStubs/` — replacement GodotSharp.dll (no-op Godot types)
- `python/play.py` — interactive terminal player
- `python/play_full_run.py` — batch testing tool
- `lib/` — game DLLs (not in repo, copied by setup.sh)
- `localization_eng/`, `localization_zhs/` — bilingual loc data

## Conventions

- **Event completion**: trust `localEvent.IsFinished`. Never gate on "option count unchanged after a choice" — events legitimately loop on the same page (Slippery Bridge Hold On) and post-selection continuations (heal/enchant/add-card) often run on the same options page; force-finishing kills them silently.
- **Async selection continuations**: any path whose effect can open a card_select / card_reward / bundle (event option, shop relic pickup, etc.) must run on `Task.Run(...)` and yield as soon as `_cardSelector.HasPending` / `HasPendingReward` / `_pendingBundles != null` appears. The Task completes naturally once the external `select_cards` feeds the selector's TCS. Reference shape: `DoChooseOption`, `DoBuyRelic`.
- **DynamicVar preview during serialization**: `UpdateDynamicVarPreview` mutates the live card. Bracket reads with `ClearPreview` **before and after** — leaving the card in preview state corrupts subsequent play actions (Momentum Strike `PlayCardAction` failure).

## Protocol notes

- `card_select` decision uses key `cards` (not `options`) and action `select_cards` with comma-separated `indices`.
- `AnyEnemy` cards/potions require `target_index` when ≥2 enemies are alive; with a single alive enemy the adapter auto-targets.
