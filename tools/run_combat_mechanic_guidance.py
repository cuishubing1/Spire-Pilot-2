"""Compare default and mechanism-aware Combat Directives on fixed engine fights."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch  # Keep Windows DLL load order stable before importing pyarrow.
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_policy_guided_mcts import (  # noqa: E402
    POST_COMBAT_DECISIONS,
    SEARCH_DECISIONS,
    _engine,
    _resolve_checkpoint,
)
from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    _game_data_dir,
)
from run_combat_mcts_act_sweep import _prepare_scenario_save  # noqa: E402
from run_combat_mcts_comparison import sha256_file_bytes  # noqa: E402
from run_combat_policy_online import (  # noqa: E402
    _advance_initial_event,
    _load_policy,
    _state_summary,
)
from sts2_dataset.combat_directive import CombatDirectiveV0  # noqa: E402
from sts2_dataset.combat_mechanics import (  # noqa: E402
    MECHANIC_GUIDANCE_VERSION,
    MechanicDirectiveControllerV0,
)
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
)
from sts2_dataset.combat_tool import CombatToolV0  # noqa: E402
from sts2_dataset.legal_actions import enumerate_legal_actions  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config" / "combat_mechanic_scenarios_v0.json"
DEFAULT_TRANSITIONS = REPO_ROOT / "data" / "human" / "combat_v1" / "transitions.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_mechanic_guidance_v0.json"


def _entry(model_id: str) -> str:
    return str(model_id).split(".", 1)[-1]


def _source_snapshots(path: Path, combat_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(
        path,
        columns=[
            "combat_id",
            "record_sequence",
            "act",
            "floor",
            "is_training_eligible",
            "observation_json",
        ],
    ).to_pylist()
    first: dict[str, dict[str, Any]] = {}
    for row in rows:
        combat_id = str(row["combat_id"])
        if combat_id not in combat_ids or not bool(row["is_training_eligible"]):
            continue
        previous = first.get(combat_id)
        if previous is not None and int(previous["record_sequence"]) <= int(row["record_sequence"]):
            continue
        first[combat_id] = row
    missing = combat_ids - set(first)
    if missing:
        raise EngineError(f"source combat snapshots not found: {sorted(missing)}")

    result: dict[str, dict[str, Any]] = {}
    for combat_id, row in first.items():
        observation = json.loads(row["observation_json"])
        player = observation.get("player") or {}
        run = observation.get("run") or {}
        deck = [
            {
                "id": _entry(card["id"]),
                "upgrade_level": int(card.get("upgrade_level") or 0),
            }
            for card in player.get("deck") or []
        ]
        result[combat_id] = {
            "source_combat_id": combat_id,
            "source_floor": int(row["floor"]),
            "ascension": int(run.get("ascension") or 0),
            "hp": int(player["hp"]),
            "max_hp": int(player["max_hp"]),
            "gold": int(player.get("gold") or 0),
            "deck": deck,
            "relics": [_entry(relic["id"]) for relic in player.get("relics") or []],
            "potions": [_entry(potion["id"]) for potion in player.get("potions") or []],
            "deck_complexity": {
                "deck_size": len(deck),
                "unique_card_ids": len({card["id"] for card in deck}),
                "upgraded_cards": sum(card["upgrade_level"] > 0 for card in deck),
            },
        }
    return result


def _create_base_save(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    ascension: int,
    seed: str,
    path: Path,
) -> dict[str, Any]:
    with _engine(args, game_data_dir) as engine:
        state, _ = engine.send({
            "cmd": "start_run",
            "character": "Ironclad",
            "ascension": ascension,
            "seed": seed,
            "lang": "en",
        })
        state, _ = _advance_initial_event(engine, state)
        saved, _ = engine.send({"cmd": "write_continue_save", "path": str(path)})
        if not saved.get("success"):
            raise EngineError(f"failed to create base save: {saved!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def _intent_damage(enemy: dict[str, Any]) -> float:
    total = 0.0
    for intent in enemy.get("intents") or []:
        if not isinstance(intent, dict) or not isinstance(intent.get("damage"), (int, float)):
            continue
        hits = intent.get("hits", intent.get("repeats", 1))
        total += float(intent["damage"]) * float(hits if isinstance(hits, (int, float)) else 1)
    return total


def _stunned(enemy: dict[str, Any]) -> bool:
    return any(
        str(intent.get("type") or "").lower() == "stun"
        for intent in enemy.get("intents") or []
        if isinstance(intent, dict)
    )


def _kill_events(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    if after.get("decision") != "combat_play" and after.get("decision") not in POST_COMBAT_DECISIONS:
        return []
    before_counts = Counter(
        str(row.get("id"))
        for row in before.get("enemies") or []
        if float(row.get("hp") or 0.0) > 0.0
    )
    after_counts = Counter(
        str(row.get("id"))
        for row in after.get("enemies") or []
        if float(row.get("hp") or 0.0) > 0.0
    )
    killed: list[str] = []
    for enemy_id, count in before_counts.items():
        killed.extend([enemy_id] * max(0, count - after_counts[enemy_id]))
    return killed


def _mechanic_metrics(
    scenario: dict[str, Any], steps: list[dict[str, Any]]
) -> dict[str, Any]:
    mechanic = scenario["mechanic"]
    kill_order = [enemy_id for row in steps for enemy_id in row.get("killed_enemies") or []]
    result: dict[str, Any] = {"kill_order": kill_order}
    if mechanic == "bowlbug_rock_full_block":
        stun_rounds = sorted({
            int(row["before"].get("round") or 0)
            for row in steps
            for enemy in row["before"].get("enemies") or []
            if enemy.get("id") == "MONSTER.BOWLBUG_ROCK" and _stunned(enemy)
        })
        result.update({
            "rock_stun_rounds": stun_rounds,
            "rock_was_stunned": bool(stun_rounds),
            "guidance_active_steps": sum(
                (row.get("guidance") or {}).get("phase") == "complete_visible_block"
                for row in steps
            ),
        })
    elif mechanic == "terror_eel_threshold_burst":
        threshold = next(
            (
                float((row.get("guidance") or {}).get("threshold_hp"))
                for row in steps
                if (row.get("guidance") or {}).get("threshold_hp") is not None
            ),
            next(
                (
                    float(power.get("amount"))
                    for row in steps
                    for enemy in row["before"].get("enemies") or []
                    if enemy.get("id") == "MONSTER.TERROR_EEL"
                    for power in enemy.get("powers") or []
                    if power.get("id") == "POWER.SHRIEK_POWER"
                    and isinstance(power.get("amount"), (int, float))
                ),
                None,
            ),
        )
        trigger_round = next(
            (
                int(row["before"].get("round") or 0)
                for row in steps
                for enemy in row["before"].get("enemies") or []
                if enemy.get("id") == "MONSTER.TERROR_EEL"
                and threshold is not None
                and float(enemy.get("hp") or 0.0) <= threshold
            ),
            None,
        )
        stun_rounds = sorted({
            int(row["before"].get("round") or 0)
            for row in steps
            for enemy in row["before"].get("enemies") or []
            if enemy.get("id") == "MONSTER.TERROR_EEL" and _stunned(enemy)
        })
        post_trigger_damage = [
            _intent_damage(enemy)
            for row in steps
            if trigger_round is not None and int(row["before"].get("round") or 0) >= trigger_round
            for enemy in row["before"].get("enemies") or []
            if enemy.get("id") == "MONSTER.TERROR_EEL"
        ]
        result.update({
            "threshold_hp": threshold,
            "trigger_round": trigger_round,
            "stun_rounds": stun_rounds,
            "rounds_from_trigger_to_kill": (
                max(0, max((int(row["before"].get("round") or 0) for row in steps), default=0) - trigger_round + 1)
                if trigger_round is not None and "MONSTER.TERROR_EEL" in kill_order else None
            ),
            "max_visible_attack_after_trigger": max(post_trigger_damage, default=0.0),
        })
    elif mechanic == "overgrowth_shrinker_priority":
        result.update({
            "first_kill": kill_order[0] if kill_order else None,
            "shrinker_killed_first": bool(kill_order and kill_order[0] == "MONSTER.SHRINKER_BEETLE"),
            "guidance_active_steps": sum(
                (row.get("guidance") or {}).get("phase") == "focus_shrinker"
                for row in steps
            ),
        })
    return result


def _run_profile(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    tool: CombatToolV0,
    guided: bool,
) -> dict[str, Any]:
    controller = (
        MechanicDirectiveControllerV0(
            scenario["mechanic"],
            terror_setup_rounds=int(scenario.get("terror_setup_rounds") or 0),
        )
        if guided else None
    )
    steps: list[dict[str, Any]] = []
    with _engine(args, game_data_dir) as engine:
        state, _ = engine.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
        state, _ = engine.send({
            "cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]
        })
        if state.get("decision") != "combat_play":
            raise EngineError(f"failed to enter {scenario['scenario_id']}: {state!r}")
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        root_signature = sha256_file_bytes(json.dumps(state, sort_keys=True).encode("utf-8"))
        for step_index in range(args.max_actions):
            decision_type = str(state.get("decision") or "")
            if decision_type not in SEARCH_DECISIONS:
                break
            before = _state_summary(state)
            guidance: dict[str, Any] | None = None
            if decision_type == "combat_play":
                sample = headless_state_to_model_sample(
                    state,
                    transition_id=f"mechanic:{scenario['scenario_id']}:{step_index}",
                    combat_id=scenario["scenario_id"],
                )
                if controller is None:
                    directive = CombatDirectiveV0.default()
                else:
                    directive, guidance = controller.directive_for(sample)
                response = tool.decide(sample, directive=directive, top_k=3)
                chosen = response["chosen"]
                candidate = chosen["candidate"]
                command = candidate_to_headless_command(candidate)
                step_detail = {
                    "chosen_candidate": candidate,
                    "probability": chosen["probability"],
                    "top_k": [
                        {
                            "candidate": row["candidate"],
                            "probability": row["probability"],
                            "score_breakdown": row["score_breakdown"],
                        }
                        for row in response["top_k"]
                    ],
                    "request_replan": response["request_replan"],
                    "inference_ms": response["inference_ms"],
                }
            else:
                actions = enumerate_legal_actions(state)
                if not actions:
                    break
                action = actions[0]
                command = {"cmd": "action", "action": action.action, "args": action.args}
                step_detail = {
                    "chosen_candidate": {
                        "candidate_id": action.action_id,
                        "action_type": action.action,
                        "source_type": "card_selection_fallback",
                    },
                    "probability": 1.0,
                    "top_k": [],
                    "request_replan": None,
                    "inference_ms": 0.0,
                }
            state, engine_ms = engine.send(command)
            after = _state_summary(state)
            steps.append({
                "step": step_index,
                "before": before,
                "guidance": guidance,
                **step_detail,
                "engine_ms": round(engine_ms, 3),
                "after": after,
                "killed_enemies": _kill_events(before, after),
            })

    final_decision = str(state.get("decision") or "")
    final_hp = float((state.get("player") or {}).get("hp") or 0.0)
    if final_decision == "game_over":
        status = "victory" if state.get("victory") else "death"
    elif final_decision in POST_COMBAT_DECISIONS:
        status = "combat_won"
    elif final_decision in SEARCH_DECISIONS:
        status = "step_limit"
    else:
        status = f"unsupported_subdecision:{final_decision or 'unknown'}"
    return {
        "profile": "mechanic_guidance_v0" if guided else "default",
        "status": status,
        "root_signature": root_signature,
        "initial_hp": initial_hp,
        "final_hp": final_hp,
        "hp_loss": max(0.0, initial_hp - final_hp),
        "rounds": max((int(row["before"].get("round") or 0) for row in steps), default=0),
        "decision_count": len(steps),
        "replan_request_count": sum(
            bool((row.get("request_replan") or {}).get("required")) for row in steps
        ),
        "mechanic_metrics": _mechanic_metrics(scenario, steps),
        "steps": steps,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if model.resource_value_head is None:
        raise EngineError("mechanic guidance requires a policy/value checkpoint")
    tool = CombatToolV0(model, tensorizer, device=device)
    raw_config = load_json(args.config.resolve())
    scenarios = list(raw_config.get("scenarios") or [])
    if args.scenario_ids:
        requested = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row.get("scenario_id") in requested]
        missing = requested - {row.get("scenario_id") for row in scenarios}
        if missing:
            raise EngineError(f"unknown scenario ids: {sorted(missing)}")
    snapshots = _source_snapshots(
        args.transitions.resolve(), {str(row["source_combat_id"]) for row in scenarios}
    )
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sts2_mechanic_guidance_") as temp_dir:
        temp = Path(temp_dir)
        for index, raw_scenario in enumerate(scenarios):
            scenario = dict(raw_scenario)
            scenario["player"] = snapshots[str(scenario["source_combat_id"])]
            base_path = temp / f"base-{index}.save"
            entrance_path = temp / f"entrance-{index}.save"
            base = _create_base_save(
                args,
                game_data_dir=game_data_dir,
                ascension=int(scenario["player"]["ascension"]),
                seed=f"{args.seed}-{scenario['scenario_id']}",
                path=base_path,
            )
            root = _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base,
                scenario=scenario,
                path=entrance_path,
            )
            default = _run_profile(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_path,
                scenario=scenario,
                tool=tool,
                guided=False,
            )
            guided = _run_profile(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_path,
                scenario=scenario,
                tool=tool,
                guided=True,
            )
            if default["root_signature"] != guided["root_signature"]:
                raise EngineError(f"root mismatch in {scenario['scenario_id']}")
            comparison = {
                "guided_minus_default_hp_loss": round(
                    float(guided["hp_loss"]) - float(default["hp_loss"]), 3
                ),
                "guided_minus_default_rounds": int(guided["rounds"]) - int(default["rounds"]),
                "same_outcome": guided["status"] == default["status"],
            }
            results.append({
                **{key: value for key, value in scenario.items() if key != "player"},
                "source_snapshot": scenario["player"],
                "root": root,
                "profiles": [default, guided],
                "comparison": comparison,
            })
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "default": {
                    "status": default["status"],
                    "hp_loss": default["hp_loss"],
                    "mechanic": default["mechanic_metrics"],
                },
                "guided": {
                    "status": guided["status"],
                    "hp_loss": guided["hp_loss"],
                    "mechanic": guided["mechanic_metrics"],
                },
            }, ensure_ascii=False), flush=True)
    return {
        "schema_version": "combat-mechanic-guidance-run-0.1.0",
        "guidance_version": MECHANIC_GUIDANCE_VERSION,
        "generated_at": utc_now(),
        "status": "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(args.config.resolve()),
        "transitions": str(args.transitions.resolve()),
        "device": device,
        "seed": args.seed,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "scenarios": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--scenario-ids", nargs="+")
    parser.add_argument("--seed", default="mechanic-guidance-v0")
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
        "scenarios": [
            {
                "scenario_id": row["scenario_id"],
                "comparison": row["comparison"],
            }
            for row in report["scenarios"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
