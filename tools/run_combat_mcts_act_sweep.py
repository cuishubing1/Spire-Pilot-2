"""Run reproducible policy/MCTS combat comparisons across Acts and deck snapshots.

This is an engineering diagnostic, not a gameplay-strength benchmark.  It uses
real A0 Ironclad deck/resource snapshots from the frozen combat dataset, but
starts controlled encounters through sts2-cli so policy-only and every search
budget see exactly the same combat entrance state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch  # Keep the Windows DLL load order stable; import before pyarrow.
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from benchmark_policy_guided_mcts import (  # noqa: E402
    DEFAULT_CONFIG,
    POST_COMBAT_DECISIONS,
    SEARCH_DECISIONS,
    _candidate_command,
    _engine,
    _resolve_checkpoint,
    search_current_root,
)
from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    _game_data_dir,
)
from run_combat_mcts_comparison import _result, sha256_file_bytes  # noqa: E402
from run_combat_policy_online import (  # noqa: E402
    _advance_initial_event,
    _load_policy,
    _rank_actions,
    _state_summary,
)
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    first_card_select_candidate,
    headless_state_to_model_sample,
)
from sts2_dataset.combat_search import SEARCH_VERSION, normalized_policy_entropy  # noqa: E402
from sts2_dataset.legal_actions import enumerate_legal_actions  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_TRANSITIONS = REPO_ROOT / "data" / "human" / "combat_v1" / "transitions.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_mcts_act_sweep.json"

REPRESENTATIVE_ENCOUNTERS = {
    1: "FUZZY_WURM_CRAWLER_WEAK",
    2: "BOWLBUGS_NORMAL",
    3: "SCROLLS_OF_BITING_NORMAL",
}


def _entry(model_id: str) -> str:
    return str(model_id).split(".", 1)[-1]


def _first_a0_ironclad_snapshots(path: Path) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(
        path,
        columns=[
            "combat_id",
            "record_sequence",
            "act",
            "floor",
            "room_type",
            "is_training_eligible",
            "observation_json",
        ],
    ).to_pylist()
    first_by_combat: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["is_training_eligible"]:
            continue
        observation = json.loads(row["observation_json"])
        if observation["player"].get("character_id") != "CHARACTER.IRONCLAD":
            continue
        if int(observation["run"].get("ascension") or 0) != 0:
            continue
        combat_id = str(row["combat_id"])
        previous = first_by_combat.get(combat_id)
        if previous is None or int(row["record_sequence"]) < int(previous["record_sequence"]):
            first_by_combat[combat_id] = {**row, "observation": observation}

    result: dict[int, dict[str, Any]] = {}
    for act in (1, 2, 3):
        candidates = [row for row in first_by_combat.values() if int(row["act"]) == act]
        if not candidates:
            raise EngineError(f"no eligible A0 Ironclad combat snapshot for Act {act}")
        row = min(candidates, key=lambda value: (int(value["floor"]), int(value["record_sequence"])))
        observation = row["observation"]
        player = observation["player"]
        deck = [
            {
                "id": _entry(card["id"]),
                "upgrade_level": int(card.get("upgrade_level") or 0),
            }
            for card in player.get("deck") or []
        ]
        result[act] = {
            "source_combat_id": row["combat_id"],
            "source_floor": int(row["floor"]),
            "source_room_type": row["room_type"],
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
                "relic_count": len(player.get("relics") or []),
                "potion_count": len(player.get("potions") or []),
            },
        }
    return result


def _scenario_specs(snapshots: dict[int, dict[str, Any]], include_controls: bool) -> list[dict[str, Any]]:
    rows = [
        {
            "scenario_id": f"representative_act{act}",
            "experiment": "representative_progression",
            "act": act,
            "deck_snapshot_act": act,
            "encounter": REPRESENTATIVE_ENCOUNTERS[act],
        }
        for act in (1, 2, 3)
    ]
    if include_controls:
        # Hold the middle-game deck fixed while encounter/Act changes.
        rows.extend([
            {
                "scenario_id": "monster_control_act1_with_act2_deck",
                "experiment": "monster_difficulty_control",
                "act": 1,
                "deck_snapshot_act": 2,
                "encounter": REPRESENTATIVE_ENCOUNTERS[1],
            },
            {
                "scenario_id": "monster_control_act3_with_act2_deck",
                "experiment": "monster_difficulty_control",
                "act": 3,
                "deck_snapshot_act": 2,
                "encounter": REPRESENTATIVE_ENCOUNTERS[3],
            },
            # Hold the Act-2 encounter fixed while deck complexity changes.
            {
                "scenario_id": "deck_control_act2_with_act1_deck",
                "experiment": "deck_complexity_control",
                "act": 2,
                "deck_snapshot_act": 1,
                "encounter": REPRESENTATIVE_ENCOUNTERS[2],
            },
            {
                "scenario_id": "deck_control_act2_with_act3_deck",
                "experiment": "deck_complexity_control",
                "act": 2,
                "deck_snapshot_act": 3,
                "encounter": REPRESENTATIVE_ENCOUNTERS[2],
            },
        ])
    for row in rows:
        row["player"] = snapshots[int(row["deck_snapshot_act"])]
    return rows


def _create_base_save(
    args: argparse.Namespace,
    game_data_dir: Path,
    path: Path,
    *,
    ascension: int = 0,
) -> dict[str, Any]:
    with _engine(args, game_data_dir) as engine:
        state, _ = engine.send({
            "cmd": "start_run",
            "character": "Ironclad",
            "ascension": int(ascension),
            "seed": args.seed,
            "lang": "en",
        })
        state, _ = _advance_initial_event(engine, state)
        result, _ = engine.send({"cmd": "write_continue_save", "path": str(path)})
        if not result.get("success"):
            raise EngineError(f"failed to write base save: {result!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_optional_precombat_selects(
    engine: Any, state: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve optional start-of-combat selectors with a neutral skip.

    Relics such as Gambling Chip expose ``card_select`` before the first
    ``combat_play`` state.  These choices are not yet controlled by the combat
    policy, so the controlled reconstruction uses the same deterministic
    ``skip_select`` baseline for every compared policy and search worker.
    Required selections remain unsupported instead of being silently guessed.
    """
    prefix: list[dict[str, Any]] = []
    for _ in range(8):
        if state.get("decision") != "card_select":
            return state, prefix
        if int(state.get("min_select") or 0) != 0:
            raise EngineError(f"required pre-combat card selection is unsupported: {state!r}")
        command = {"cmd": "action", "action": "skip_select", "args": {}}
        state, _ = engine.send(command)
        prefix.append(command)
    raise EngineError("too many pre-combat card selections")


def _prepare_scenario_save(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    base_save: dict[str, Any],
    scenario: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    patched = dict(base_save)
    patched["current_act_index"] = int(scenario["act"]) - 1
    player = scenario["player"]
    with _engine(args, game_data_dir) as engine:
        state, _ = engine.send({"cmd": "load_save", "json": json.dumps(patched), "lang": "en"})
        if state.get("decision") != "map_select":
            raise EngineError(f"scenario save did not load at map: {state!r}")
        set_result, _ = engine.send({
            "cmd": "set_player",
            "hp": player["hp"],
            "max_hp": player["max_hp"],
            "gold": player["gold"],
            "deck": player["deck"],
            "relics": player["relics"],
            "potions": player["potions"],
        })
        if set_result.get("type") != "ok":
            raise EngineError(f"set_player failed: {set_result!r}")
        saved, _ = engine.send({"cmd": "write_continue_save", "path": str(path)})
        if not saved.get("success"):
            raise EngineError(f"failed to write scenario save: {saved!r}")
        combat, _ = engine.send({
            "cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]
        })
        combat, _ = _resolve_optional_precombat_selects(engine, combat)
        if combat.get("decision") != "combat_play":
            raise EngineError(f"scenario did not enter combat: {combat!r}")
        sample = headless_state_to_model_sample(
            combat, transition_id=f"scenario:{scenario['scenario_id']}", combat_id=scenario["scenario_id"]
        )
        return {
            "root_signature": sha256_file_bytes(json.dumps(combat, sort_keys=True).encode("utf-8")),
            "enemies": [
                {"id": enemy.get("id"), "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp")}
                for enemy in combat.get("enemies") or []
            ],
            "enemy_count": len(combat.get("enemies") or []),
            "enemy_total_max_hp": sum(float(enemy.get("max_hp") or 0) for enemy in combat.get("enemies") or []),
            "root_legal_action_count": len(sample["candidates"]),
        }


def _enter_command(scenario: dict[str, Any]) -> dict[str, Any]:
    return {"cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]}


def _run_policy(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    encounter_signature: str | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    with _engine(args, game_data_dir) as engine:
        state, _ = engine.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
        state, _ = engine.send(_enter_command(scenario))
        state, _ = _resolve_optional_precombat_selects(engine, state)
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        root_signature = sha256_file_bytes(json.dumps(state, sort_keys=True).encode("utf-8"))
        for step in range(args.max_actions):
            decision = str(state.get("decision") or "")
            if decision not in SEARCH_DECISIONS:
                break
            if decision == "combat_play":
                sample = headless_state_to_model_sample(
                    state,
                    transition_id=f"policy:{scenario['scenario_id']}:{step}",
                    combat_id=scenario["scenario_id"],
                    encounter_signature=encounter_signature,
                )
                ranked, inference_ms = _rank_actions(model, tensorizer, sample, device=device, objective=None)
                chosen = max(ranked, key=lambda row: float(row["policy_probability"]))
                command = candidate_to_headless_command(chosen["candidate"])
                policy_probability = float(chosen["policy_probability"])
                policy_entropy = normalized_policy_entropy(
                    [float(row["policy_probability"]) for row in ranked]
                )
                candidate = chosen["candidate"]
            else:
                actions = enumerate_legal_actions(state)
                candidate = first_card_select_candidate(state)
                command = candidate_to_headless_command(candidate)
                ranked = actions
                inference_ms = 0.0
                policy_probability = 1.0
                policy_entropy = 0.0
            before = _state_summary(state)
            state, engine_ms = engine.send(command)
            steps.append({
                "step": step,
                "before": before,
                "chosen_candidate": candidate,
                "policy_probability": policy_probability,
                "policy_entropy": policy_entropy,
                "legal_action_count": len(ranked),
                "inference_ms": round(inference_ms, 3),
                "engine_ms": round(engine_ms, 3),
                "after": _state_summary(state),
            })
        result = _result(state, initial_hp=initial_hp, steps=steps)
        result["root_signature"] = root_signature
        result["mean_policy_entropy"] = statistics.fmean(
            float(row["policy_entropy"]) for row in steps
        ) if steps else None
        result["mean_legal_action_count"] = statistics.fmean(
            int(row["legal_action_count"]) for row in steps
        ) if steps else None
        return result


def _run_mcts(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    config: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    with _engine(args, game_data_dir) as real_engine:
        state, _ = real_engine.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
        state, _ = real_engine.send(_enter_command(scenario))
        state, precombat_prefix = _resolve_optional_precombat_selects(real_engine, state)
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        root_signature = sha256_file_bytes(json.dumps(state, sort_keys=True).encode("utf-8"))
        prefix.extend(precombat_prefix)
        with _engine(args, game_data_dir) as worker:
            warm, _ = worker.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
            if warm.get("decision") != "map_select":
                raise EngineError(f"worker warmup failed: {warm!r}")
            for step in range(args.max_actions):
                if state.get("decision") not in SEARCH_DECISIONS:
                    break
                before = _state_summary(state)
                search = search_current_root(
                    worker=worker,
                    entrance_save=entrance_save,
                    enter_command=_enter_command(scenario),
                    root_prefix=prefix,
                    root_state=state,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    objective=objective,
                    config=config,
                    budget=budget,
                    max_depth=int(args.max_depth or config["puct"]["maximum_player_decision_depth"]),
                    search_seed=args.search_seed + step * 100003,
                )
                command = _candidate_command(search["chosen_candidate"])
                state, engine_ms = real_engine.send(command)
                prefix.append(command)
                steps.append({
                    "step": step,
                    "before": before,
                    "chosen_candidate": search["chosen_candidate"],
                    "engine_ms": round(engine_ms, 3),
                    "after": _state_summary(state),
                    "search": search,
                })
    result = _result(state, initial_hp=initial_hp, steps=steps)
    result["root_signature"] = root_signature
    result["budget"] = budget
    result["total_search_ms"] = round(sum(float(row["search"]["search_wall_ms"]) for row in steps), 3)
    result["total_simulations"] = sum(int(row["search"]["effective_budget"]) for row in steps)
    result["mean_legal_action_count"] = statistics.fmean(
        len(row["search"]["actions"]) for row in steps
    ) if steps else None
    result["mean_root_policy_entropy"] = statistics.fmean(
        normalized_policy_entropy([float(action["prior"]) for action in row["search"]["actions"]])
        for row in steps
    ) if steps else None
    result["mean_prior_visit_l1"] = statistics.fmean(
        sum(
            abs(float(action["prior"]) - float(action["visit_policy_probability"]))
            for action in row["search"]["actions"]
        )
        for row in steps
    ) if steps else None
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if model.resource_value_head is None:
        raise EngineError("Act sweep requires a combat_policy_value_v1 checkpoint")
    objective = CombatObjective.from_config(model.config)
    config = load_json(args.config.resolve())
    if getattr(args, "restore_mode", None) is not None:
        config.setdefault("engine_restore", {})["mode"] = args.restore_mode
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, args.include_controls)
    if args.experiments:
        requested = set(args.experiments)
        scenarios = [row for row in scenarios if row["experiment"] in requested]
        if not scenarios:
            raise EngineError(f"no scenarios matched experiments: {sorted(requested)}")
    if args.scenario_ids:
        requested_ids = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row["scenario_id"] in requested_ids]
        if not scenarios:
            raise EngineError(f"no scenarios matched ids: {sorted(requested_ids)}")
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sts2_act_sweep_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        for index, scenario in enumerate(scenarios):
            entrance_save = temp / f"scenario-{index}.save"
            root = _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base_save,
                scenario=scenario,
                path=entrance_save,
            )
            policy = _run_policy(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
            )
            if policy["root_signature"] != root["root_signature"]:
                raise EngineError(f"prepared/policy root mismatch in {scenario['scenario_id']}")
            searches = []
            for budget in args.budgets:
                search = _run_mcts(
                    args,
                    game_data_dir=game_data_dir,
                    entrance_save=entrance_save,
                    scenario=scenario,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    objective=objective,
                    config=config,
                    budget=budget,
                )
                if search["root_signature"] != policy["root_signature"]:
                    raise EngineError(f"root mismatch in {scenario['scenario_id']}")
                searches.append(search)
            results.append({
                **{key: value for key, value in scenario.items() if key != "player"},
                "snapshot": scenario["player"],
                "root": root,
                "policy_only": policy,
                "searches": searches,
            })
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "policy": {"status": policy["status"], "hp_loss": policy["hp_loss"]},
                "searches": [
                    {"budget": row["budget"], "status": row["status"], "hp_loss": row["hp_loss"]}
                    for row in searches
                ],
            }, ensure_ascii=False), flush=True)
    return {
        "schema_version": "combat-mcts-act-sweep-0.2.0",
        "search_version": SEARCH_VERSION,
        "generated_at": utc_now(),
        "status": "pass",
        "seed": args.seed,
        "search_seed": args.search_seed,
        "budgets": args.budgets,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "transitions": str(args.transitions.resolve()),
        "include_controls": args.include_controls,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "scenarios": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--restore-mode",
        choices=(
            "cached_batch_auto_prepared",
            "cached_batch_auto",
            "cached_batch_compact",
            "cached_batch",
            "legacy",
        ),
    )
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", default="act-grid-v0")
    parser.add_argument("--search-seed", type=int, default=20260815)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 32])
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=(
            "representative_progression",
            "monster_difficulty_control",
            "deck_complexity_control",
        ),
    )
    parser.add_argument("--scenario-ids", nargs="+")
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-actions", type=int, default=100)
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
        "scenario_count": len(report["scenarios"]),
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
