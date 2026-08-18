"""Audit v0.107.1 shared + Ironclad potion execution through sts2-cli.

The catalog is a design contract; this tool checks the real game assembly.
It injects one potion at a time into a controlled Ironclad combat, records
whether the potion can be created and used, and resolves the first exposed
card-selection boundary.  It does not assign potion values yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    EngineProcess,
    _game_data_dir,
    _skip_neow,
)
from sts2_dataset.combat_potions import (  # noqa: E402
    GAME_VERSION,
    POTION_CATALOG_VERSION,
    POTION_SPECS,
    STEAM_BUILD,
    PotionSpec,
    validate_potion_catalog,
)
from sts2_dataset.util import utc_now, write_json_atomic  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_potion_cli_audit_v01071.json"
DEFAULT_DECK = [
    "STRIKE_IRONCLAD",
    "STRIKE_IRONCLAD",
    "STRIKE_IRONCLAD",
    "BASH",
    "ANGER",
    "DEFEND_IRONCLAD",
    "DEFEND_IRONCLAD",
    "DEFEND_IRONCLAD",
    "DEFEND_IRONCLAD",
    "ARMAMENTS",
    "HEADBUTT",
    "FEED",
]
SELECTION_MODES = {
    "random_offer_one",
    "single_hand",
    "single_draw",
    "single_discard",
    "hand_subset",
}


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "type": state.get("type"),
        "decision": state.get("decision"),
        "round": state.get("round"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "block": player.get("block"),
        "energy": state.get("energy"),
        "hand_ids": [card.get("id") for card in state.get("hand") or []],
        "selection_ids": [card.get("id") for card in state.get("cards") or []],
        "min_select": state.get("min_select"),
        "max_select": state.get("max_select"),
        "potion_ids": [potion.get("id") for potion in player.get("potions") or []],
        "enemy_hp": [enemy.get("hp") for enemy in state.get("enemies") or []],
    }


def _start_case(engine: EngineProcess, spec: PotionSpec, case_index: int) -> dict[str, Any]:
    state, _ = engine.send({
        "cmd": "start_run",
        "character": "Ironclad",
        "ascension": 0,
        # A known-simple Neow seed keeps this audit focused on potion behavior.
        # Cases run in isolated processes, so sharing the seed cannot leak state.
        "seed": "card-select-attack-potion",
        "lang": "en",
    })
    state, _ = _skip_neow(engine, state)
    engine.send({
        "cmd": "set_player",
        "hp": 40,
        "max_hp": 80,
        "deck": DEFAULT_DECK,
        "potions": [spec.potion_id.removeprefix("POTION.")],
    })
    state, _ = engine.send({
        "cmd": "enter_room",
        "type": "combat",
        "encounter": "FUZZY_WURM_CRAWLER_WEAK",
    })
    return state


def _prepare_state(engine: EngineProcess, state: dict[str, Any], spec: PotionSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    if spec.setup not in {"discard_card", "block_in_hand"}:
        return state, steps
    required_cards = 2 if spec.setup == "discard_card" else 1
    for _ in range(required_cards):
        card = next(
            (
                row for row in state.get("hand") or []
                if str(row.get("id") or "").endswith("DEFEND_IRONCLAD") and row.get("can_play", True)
            ),
            None,
        )
        if card is None:
            raise EngineError(f"{spec.potion_id} setup could not find enough playable Defends")
        state, elapsed_ms = engine.send({
            "cmd": "action",
            "action": "play_card",
            "args": {"card_index": int(card["index"])},
        })
        steps.append({
            "action": "play_card_setup",
            "elapsed_ms": round(elapsed_ms, 3),
            "state": _compact_state(state),
        })
    return state, steps


def _resolve_first_selection(engine: EngineProcess, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if state.get("decision") != "card_select":
        return state, None
    cards = list(state.get("cards") or [])
    if not cards:
        if int(state.get("min_select") or 0) > 0:
            raise EngineError("card_select requires a card but exposes no candidates")
        command = {"cmd": "action", "action": "skip_select"}
    else:
        selected_index = int(cards[0].get("index") or 0)
        command = {
            "cmd": "action",
            "action": "select_cards",
            "args": {"indices": str(selected_index)},
        }
    resolved, elapsed_ms = engine.send(command)
    return resolved, {
        "command": command,
        "elapsed_ms": round(elapsed_ms, 3),
        "state": _compact_state(resolved),
    }


def _audit_one(engine: EngineProcess, spec: PotionSpec, case_index: int) -> dict[str, Any]:
    stderr_start = len(engine._stderr)
    initial = _start_case(engine, spec, case_index)
    potion_rows = [
        row for row in (initial.get("player") or {}).get("potions") or []
        if str(row.get("id")) == spec.potion_id
    ]
    row: dict[str, Any] = {
        "spec": spec.to_dict(),
        "created": bool(potion_rows),
        "initial": _compact_state(initial),
        "target_type": potion_rows[0].get("target_type") if potion_rows else None,
        "status": "pending",
        "steps": [],
    }
    if not potion_rows:
        row["status"] = "create_failed"
        return row
    if spec.evaluator == "passive_reserve":
        row["status"] = "passive_created"
        return row

    state, setup_steps = _prepare_state(engine, initial, spec)
    row["steps"].extend(setup_steps)
    current_potions = (state.get("player") or {}).get("potions") or []
    potion = next((p for p in current_potions if str(p.get("id")) == spec.potion_id), None)
    if potion is None:
        row["status"] = "missing_after_setup"
        return row
    command: dict[str, Any] = {
        "cmd": "action",
        "action": "use_potion",
        "args": {"potion_index": int(potion.get("index") or 0)},
    }
    if str(potion.get("target_type") or "").lower() == "anyenemy":
        command["args"]["target_index"] = 0
    after_use, elapsed_ms = engine.send(command, timeout_s=20.0)
    exposed_selection = after_use.get("decision") == "card_select"
    row["steps"].append({
        "action": "use_potion",
        "command": command,
        "elapsed_ms": round(elapsed_ms, 3),
        "state": _compact_state(after_use),
    })
    final, selection_step = _resolve_first_selection(engine, after_use)
    if selection_step is not None:
        row["steps"].append({"action": "resolve_selection", **selection_step})
    new_stderr = list(engine._stderr)[stderr_start:]
    engine_failures = [line for line in new_stderr if "Use potion failed" in line or "[ERROR]" in line]
    row.update({
        "exposed_selection": exposed_selection,
        "selection_expected": spec.choice_mode in SELECTION_MODES,
        "selection_contract_match": exposed_selection == (spec.choice_mode in SELECTION_MODES),
        "engine_failure_logs": engine_failures,
        "final": _compact_state(final),
        "status": "engine_warning" if engine_failures else "ok",
    })
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog_counts = validate_potion_catalog()
    requested = set(args.potion_id or [])
    specs = [spec for spec in POTION_SPECS if not requested or spec.potion_id in requested]
    unknown = requested - {spec.potion_id for spec in POTION_SPECS}
    if unknown:
        raise ValueError(f"unknown catalog potion ids: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    game_data_dir = _game_data_dir(args.game_dir)
    for index, spec in enumerate(specs):
        row: dict[str, Any] | None = None
        try:
            with EngineProcess(
                dotnet=args.dotnet,
                engine_dll=args.engine_dll,
                game_data_dir=game_data_dir,
                sts2_lib=args.sts2_lib,
                timeout_s=args.timeout,
            ) as engine:
                try:
                    row = _audit_one(engine, spec, index)
                except Exception as exc:  # retain process diagnostics before close
                    row = {
                        "spec": spec.to_dict(),
                        "status": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "stderr_tail": list(engine._stderr)[-30:],
                    }
        except Exception as exc:  # startup/teardown failure for this isolated case
            if row is None:
                row = {
                    "spec": spec.to_dict(),
                    "status": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        assert row is not None
        rows.append(row)
        print(f"[{index + 1:02d}/{len(specs):02d}] {spec.potion_id}: {row['status']}", flush=True)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "combat-potion-cli-audit-0.1.0",
        "generated_at": utc_now(),
        "catalog_version": POTION_CATALOG_VERSION,
        "game_version": GAME_VERSION,
        "steam_build": STEAM_BUILD,
        "engine_dll": str(args.engine_dll.resolve()),
        "catalog_counts": catalog_counts,
        "audited_count": len(rows),
        "status_counts": status_counts,
        "selection_contract_failures": [
            row["spec"]["potion_id"]
            for row in rows
            if row.get("selection_contract_match") is False
        ],
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--potion-id", action="append", help="Full POTION.* id; repeatable")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0 if not result["status_counts"].get("exception") else 1


if __name__ == "__main__":
    raise SystemExit(main())
