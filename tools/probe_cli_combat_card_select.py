"""Probe sts2-cli combat card-selection behavior with controlled encounters.

This diagnostic distinguishes generic combat card selection from selections
opened during enemy turn processing.  It does not modify the game assembly or
the simulator and writes a compact JSON evidence report.
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
    EngineProcess,
    _game_data_dir,
    _skip_neow,
)
from sts2_dataset.util import utc_now, write_json_atomic  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "cli_combat_card_select_probe.json"


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "type": state.get("type"),
        "decision": state.get("decision"),
        "round": state.get("round"),
        "player_hp": player.get("hp"),
        "victory": state.get("victory"),
        "cards": [
            {
                "index": card.get("index"),
                "id": card.get("id"),
                "name": card.get("name"),
            }
            for card in state.get("cards") or []
        ],
        "hand": [card.get("id") for card in state.get("hand") or []],
        "enemies": [
            {
                "id": enemy.get("id"),
                "hp": enemy.get("hp"),
                "intents": enemy.get("intents"),
            }
            for enemy in state.get("enemies") or []
        ],
    }


def _start_combat(
    engine: EngineProcess,
    *,
    seed: str,
    encounter: str,
    deck: list[str],
    potions: list[str] | None = None,
) -> dict[str, Any]:
    state, _ = engine.send({
        "cmd": "start_run",
        "character": "Ironclad",
        "ascension": 0,
        "seed": seed,
        "lang": "en",
    })
    state, _ = _skip_neow(engine, state)
    set_player: dict[str, Any] = {
        "cmd": "set_player",
        "hp": 999,
        "max_hp": 999,
        "deck": deck,
    }
    if potions is not None:
        set_player["potions"] = potions
    engine.send(set_player)
    state, _ = engine.send({
        "cmd": "enter_room",
        "type": "combat",
        "encounter": encounter,
    })
    return state


def _new_engine(args: argparse.Namespace) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet,
        engine_dll=args.engine_dll,
        game_data_dir=_game_data_dir(args.game_dir),
        sts2_lib=args.sts2_lib,
        timeout_s=args.timeout,
    )


def _knowledge_demon_case(args: argparse.Namespace, seed: str) -> dict[str, Any]:
    with _new_engine(args) as engine:
        initial = _start_combat(
            engine,
            seed=seed,
            encounter="KNOWLEDGE_DEMON_BOSS",
            deck=["DEFEND_IRONCLAD"] * 5,
        )
        response, elapsed_ms = engine.send(
            {"cmd": "action", "action": "end_turn"},
            timeout_s=max(args.timeout, 20.0),
        )
        after_get_state, get_state_ms = engine.send({"cmd": "get_state"})
        after_selection = None
        if after_get_state.get("decision") == "card_select":
            selected_state, select_ms = engine.send({
                "cmd": "action",
                "action": "select_cards",
                "args": {"indices": "0"},
            })
            after_selection = {
                "elapsed_ms": round(select_ms, 3),
                "state": _summary(selected_state),
            }
        return {
            "case": "knowledge_demon_enemy_turn_selection",
            "seed": seed,
            "initial": _summary(initial),
            "end_turn_response": _summary(response),
            "end_turn_ms": round(elapsed_ms, 3),
            "get_state_response": _summary(after_get_state),
            "get_state_ms": round(get_state_ms, 3),
            "after_selection": after_selection,
            "stderr_tail": list(engine._stderr),
        }


def _headbutt_case(args: argparse.Namespace) -> dict[str, Any]:
    with _new_engine(args) as engine:
        state = _start_combat(
            engine,
            seed="card-select-headbutt",
            encounter="FUZZY_WURM_CRAWLER_WEAK",
            deck=[
                "STRIKE_IRONCLAD",
                "DEFEND_IRONCLAD",
                "HEADBUTT",
                "DEFEND_IRONCLAD",
                "DEFEND_IRONCLAD",
            ],
        )
        steps = [{"label": "initial", "state": _summary(state)}]
        # Put two distinct cards into the discard pile.  A single eligible card
        # may be auto-resolved by the game and is not evidence of a CLI bug.
        for identity, label in (("DEFEND_IRONCLAD", "after_defend"), ("STRIKE_IRONCLAD", "after_strike")):
            card = next(
                card for card in state["hand"]
                if str(card.get("id", "")).endswith(identity)
            )
            action_args = {"card_index": card["index"]}
            if card.get("target_type") == "AnyEnemy":
                action_args["target_index"] = 0
            state, elapsed_ms = engine.send({
                "cmd": "action",
                "action": "play_card",
                "args": action_args,
            })
            steps.append({
                "label": label,
                "elapsed_ms": round(elapsed_ms, 3),
                "state": _summary(state),
            })
        headbutt = next(card for card in state["hand"] if str(card.get("id", "")).endswith("HEADBUTT"))
        state, headbutt_ms = engine.send({
            "cmd": "action",
            "action": "play_card",
            "args": {"card_index": headbutt["index"], "target_index": 0},
        })
        steps.append({"label": "after_headbutt", "elapsed_ms": round(headbutt_ms, 3), "state": _summary(state)})
        if state.get("decision") == "card_select":
            state, select_ms = engine.send({
                "cmd": "action",
                "action": "select_cards",
                "args": {"indices": "0"},
            })
            steps.append({"label": "after_selection", "elapsed_ms": round(select_ms, 3), "state": _summary(state)})
        return {
            "case": "player_card_selection_headbutt",
            "steps": steps,
            "stderr_tail": list(engine._stderr),
        }


def _ordinary_end_turn_case(args: argparse.Namespace) -> dict[str, Any]:
    with _new_engine(args) as engine:
        initial = _start_combat(
            engine,
            seed="ordinary-end-turn",
            encounter="FUZZY_WURM_CRAWLER_WEAK",
            deck=["DEFEND_IRONCLAD"] * 5,
        )
        response, elapsed_ms = engine.send({"cmd": "action", "action": "end_turn"})
        return {
            "case": "ordinary_enemy_turn_without_selection",
            "initial": _summary(initial),
            "end_turn_response": _summary(response),
            "end_turn_ms": round(elapsed_ms, 3),
            "stderr_tail": list(engine._stderr),
        }


def _attack_potion_case(args: argparse.Namespace) -> dict[str, Any]:
    with _new_engine(args) as engine:
        state = _start_combat(
            engine,
            seed="card-select-attack-potion",
            encounter="FUZZY_WURM_CRAWLER_WEAK",
            deck=["DEFEND_IRONCLAD"] * 5,
            potions=["ATTACK_POTION"],
        )
        initial = _summary(state)
        response, elapsed_ms = engine.send({
            "cmd": "action",
            "action": "use_potion",
            "args": {"potion_index": 0},
        })
        resolved = None
        if response.get("decision") == "card_select":
            resolved_state, resolve_ms = engine.send({
                "cmd": "action",
                "action": "select_cards",
                "args": {"indices": "0"},
            })
            resolved = {
                "elapsed_ms": round(resolve_ms, 3),
                "state": _summary(resolved_state),
            }
        return {
            "case": "player_potion_selection_attack_potion",
            "initial": initial,
            "use_potion_response": _summary(response),
            "use_potion_ms": round(elapsed_ms, 3),
            "after_selection": resolved,
            "stderr_tail": list(engine._stderr),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = [
        _knowledge_demon_case(args, seed)
        for seed in ("knowledge-select-a", "knowledge-select-b", "knowledge-select-c")
    ]
    cases.append(_ordinary_end_turn_case(args))
    cases.append(_headbutt_case(args))
    cases.append(_attack_potion_case(args))
    return {
        "schema_version": "sts2-cli-combat-card-select-probe-0.1.0",
        "generated_at": utc_now(),
        "engine_dll": str(args.engine_dll.resolve()),
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
