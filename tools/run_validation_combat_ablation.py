"""Evaluate P2 combat policies on every reconstructable validation fight.

The runner reconstructs each fight from the first player-visible observation,
preserves the recorded ascension, and compares policy-only and exact one-step
variants on the same generated engine root.  It supports deterministic outer
sharding so several CUDA processes can evaluate independent combats in
parallel.  Progress is written atomically after every completed combat.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch  # Keep the known Windows native DLL import order stable.
import pyarrow.parquet as pq


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
    _game_data_dir,
)
from benchmark_policy_guided_mcts import _engine  # noqa: E402
from run_combat_mcts_act_sweep import (  # noqa: E402
    _prepare_scenario_save,
    _resolve_optional_precombat_selects,
    _run_policy,
)
from run_combat_one_step_act_sweep import _run_one_step  # noqa: E402
from run_combat_policy_online import (  # noqa: E402
    _load_policy,
    _resolve_checkpoint,
)
from run_heldout_run_combat_comparison import (  # noqa: E402
    ENCOUNTER_BY_MONSTERS,
    _monster_key,
    _player_snapshot,
)
from sts2_dataset.combat_encounter import encounter_signature_from_observation  # noqa: E402
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_TRANSITIONS = REPO_ROOT / "data" / "human" / "combat_v1" / "transitions.parquet"
DEFAULT_COMBATS = REPO_ROOT / "data" / "human" / "combat_v1" / "combats.parquet"
DEFAULT_TARGETS = REPO_ROOT / "data" / "human" / "combat_value_v1" / "targets.parquet"
DEFAULT_P2 = REPO_ROOT / "artifacts" / "combat_policy_p2_cuda" / "latest.json"
DEFAULT_RESIDUAL = (
    REPO_ROOT / "artifacts" / "combat_policy_p2_encounter_residual_cuda" / "latest.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "validation_combat_ablation_p2.json"
ALL_METHODS = (
    "p2_policy",
    "residual_policy",
    "p2_one_step",
    "residual_one_step",
)


def _latest_checkpoint(path: Path) -> Path:
    payload = load_json(path.resolve())
    checkpoint = Path(payload["checkpoint"])
    return (checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint).resolve()


def _load_scenarios(
    args: argparse.Namespace,
    *,
    dataset_split: str = "validation",
    skip_unmapped_encounters: bool = False,
) -> list[dict[str, Any]]:
    combats = {
        str(row["combat_id"]): row
        for row in pq.read_table(args.combats.resolve()).to_pylist()
        if row["split"] == dataset_split
    }
    first_by_combat: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(args.transitions.resolve()).to_pylist():
        combat_id = str(row["combat_id"])
        if combat_id not in combats or not row["is_training_eligible"]:
            continue
        previous = first_by_combat.get(combat_id)
        if previous is None or int(row["record_sequence"]) < int(previous["record_sequence"]):
            first_by_combat[combat_id] = row
    targets = {
        str(row["transition_id"]): row
        for row in pq.read_table(args.targets.resolve()).to_pylist()
    }
    ordered = sorted(
        first_by_combat.items(),
        key=lambda item: (
            int(combats[item[0]]["act"]),
            str(combats[item[0]]["run_id"]),
            int(combats[item[0]]["floor"]),
            int(item[1]["record_sequence"]),
        ),
    )
    scenarios: list[dict[str, Any]] = []
    for global_index, (combat_id, row) in enumerate(ordered):
        if global_index % args.shard_count != args.shard_index:
            continue
        observation = json.loads(row["observation_json"])
        if observation["player"].get("character_id") != "CHARACTER.IRONCLAD":
            continue
        combat = combats[combat_id]
        monsters = _monster_key(observation)
        encounter = ENCOUNTER_BY_MONSTERS.get(monsters)
        if encounter is None:
            if skip_unmapped_encounters:
                continue
            raise EngineError(
                f"no EncounterModel mapping for {dataset_split} combat {combat_id}: {monsters!r}"
            )
        target = targets[str(row["transition_id"])]
        scenarios.append({
            "scenario_id": f"{dataset_split}-{global_index:03d}-act{combat['act']}-floor{combat['floor']}",
            "global_index": global_index,
            "source_combat_id": combat_id,
            "source_transition_id": str(row["transition_id"]),
            "run_id": str(combat["run_id"]),
            "act": int(combat["act"]),
            "floor": int(combat["floor"]),
            "room_type": str(combat["room_type"]),
            "ascension": int(combat["ascension"]),
            "combat_difficulty_tier": str(combat["combat_difficulty_tier"]),
            "encounter": encounter,
            "encounter_signature": encounter_signature_from_observation(observation),
            "recorded_monsters": list(monsters),
            "player": _player_snapshot(observation),
            "human": {
                "start_hp": int(target["current_hp"]),
                "start_max_hp": int(target["current_max_hp"]),
                "terminal_hp": int(target["terminal_hp"]),
                "terminal_max_hp": int(target["terminal_max_hp"]),
                "hp_loss": int(target["hp_loss_to_end"]),
                "death": bool(target["death"]),
                "potion_spent": int(target["potion_spent_to_end"]),
                "max_hp_delta": int(target["max_hp_delta_to_end"]),
            },
        })
    if not scenarios:
        raise EngineError(
            f"{dataset_split} shard {args.shard_index}/{args.shard_count} contains no Ironclad combats"
        )
    return scenarios


def _completed(result: dict[str, Any]) -> bool:
    if "status" not in result:
        return not bool(result.get("death"))
    return result.get("status") in {"complete", "combat_won", "victory"}


def _seed_search_neow_choice(options: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer Neow rewards that do not open a secondary reward workflow."""
    safe_markers = (
        "NEOWS_TORMENT",
        "NEOWS_TALISMAN",
        "NUTRITIOUS_OYSTER",
    )
    for marker in safe_markers:
        match = next(
            (
                row
                for row in options
                if marker in str(row.get("text_key") or "").upper()
            ),
            None,
        )
        if match is not None:
            return match
    secondary_terms = (
        "card reward",
        "choose",
        "procure",
        "remove",
        "random",
        "transform",
    )
    return min(
        options,
        key=lambda row: (
            int(
                "upon pickup" in str(row.get("description") or "").lower()
                and any(
                    term in str(row.get("description") or "").lower()
                    for term in secondary_terms
                )
            ),
            sum(
                term in str(row.get("description") or "").lower()
                for term in secondary_terms
            ),
            int(row.get("index") or 0),
        ),
    )


def _advance_seed_search_initial_event(engine: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Leave Neow using a reward that avoids optional secondary workflows."""
    initial_option_chosen = False
    for _ in range(50):
        decision = state.get("decision")
        if decision == "map_select":
            return state
        if decision == "event_choice":
            if initial_option_chosen:
                command = {"cmd": "action", "action": "proceed"}
            else:
                options = [
                    row for row in state.get("options") or [] if not row.get("is_locked")
                ]
                if not options:
                    raise EngineError("seed-search initial event exposed no unlocked option")
                choice = _seed_search_neow_choice(options)
                command = {
                    "cmd": "action",
                    "action": "choose_option",
                    "args": {"option_index": choice["index"]},
                }
                initial_option_chosen = True
        elif decision == "card_reward":
            command = {"cmd": "action", "action": "skip_card_reward"}
        elif decision == "bundle_select":
            command = {
                "cmd": "action",
                "action": "select_bundle",
                "args": {"bundle_index": 0},
            }
        elif decision == "card_select":
            command = (
                {"cmd": "action", "action": "skip_select"}
                if int(state.get("min_select") or 0) == 0
                else {
                    "cmd": "action",
                    "action": "select_cards",
                    "args": {"indices": "0"},
                }
            )
        else:
            command = {"cmd": "action", "action": "proceed"}
        state, _ = engine.send(command)
    raise EngineError(f"failed to leave seed-search initial event; final state={state!r}")


def _find_matching_base_save(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    scenario: dict[str, Any],
    path: Path,
) -> tuple[dict[str, Any], str, int]:
    """Find a public run seed that generates the recorded monster multiset.

    Encounter models may randomly choose concrete monster variants.  Comparing
    against a different variant would confound policy quality with encounter
    composition, so reconstruction searches only public run seeds and saves the
    matching pre-combat map state.  No hidden state from the human run is read.
    """
    expected = Counter(scenario["recorded_monsters"])
    seed_prefix = ":".join((
        args.seed,
        str(scenario["ascension"]),
        str(scenario["encounter"]),
        ",".join(scenario["recorded_monsters"]),
    ))
    initial_event_failures = 0
    for trial in range(args.encounter_seed_trials):
        with _engine(args, game_data_dir) as engine:
            seed = f"{seed_prefix}:{trial}"
            state, _ = engine.send({
                "cmd": "start_run",
                "character": "Ironclad",
                "ascension": int(scenario["ascension"]),
                "seed": seed,
                "lang": "en",
            })
            try:
                state = _advance_seed_search_initial_event(engine, state)
            except EngineError:
                initial_event_failures += 1
                continue
            saved, _ = engine.send({"cmd": "write_continue_save", "path": str(path)})
            if not saved.get("success"):
                raise EngineError(f"failed to write seed-search base save: {saved!r}")
            combat, _ = engine.send({
                "cmd": "enter_room",
                "type": "combat",
                "encounter": scenario["encounter"],
            })
            combat, _ = _resolve_optional_precombat_selects(engine, combat)
            if combat.get("decision") != "combat_play":
                raise EngineError(f"seed-search encounter did not enter combat: {combat!r}")
            actual = Counter(str(enemy["id"]) for enemy in combat.get("enemies") or [])
            if actual == expected:
                return json.loads(path.read_text(encoding="utf-8")), seed, trial + 1
    raise EngineError(
        f"could not reconstruct monster composition after {args.encounter_seed_trials} seeds: "
        f"scenario={scenario['scenario_id']}, encounter={scenario['encounter']}, "
        f"monsters={scenario['recorded_monsters']!r}, "
        f"initial_event_failures={initial_event_failures}"
    )


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    available = [row[key] for row in rows if key in row]
    losses = [float(result["hp_loss"]) for result in available]
    return {
        "combats": len(available),
        "completed": sum(_completed(result) for result in available),
        "deaths": sum(
            bool(result.get("death")) or result.get("status") == "death"
            for result in available
        ),
        "total_hp_loss": round(sum(losses), 3),
        "mean_hp_loss": round(statistics.fmean(losses), 3) if losses else None,
    }


def _summary(rows: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    keys = ("human", *methods)
    overall = {key: _aggregate(rows, key) for key in keys}
    human_loss = float(overall["human"]["total_hp_loss"])
    for key in methods:
        model_loss = float(overall[key]["total_hp_loss"])
        overall[key]["hp_loss_vs_human_percent"] = (
            round((model_loss - human_loss) / human_loss * 100.0, 3)
            if human_loss > 0 and overall[key]["combats"] == overall["human"]["combats"]
            else None
        )
    grouped: dict[str, Any] = {}
    group_specs = {
        "by_act": lambda row: str(row["act"]),
        "by_difficulty_tier": lambda row: str(row["combat_difficulty_tier"]),
    }
    for group_name, getter in group_specs.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[getter(row)].append(row)
        grouped[group_name] = {
            bucket: {key: _aggregate(bucket_rows, key) for key in keys}
            for bucket, bucket_rows in sorted(buckets.items())
        }
    return {"overall": overall, **grouped}


def _report(
    *,
    args: argparse.Namespace,
    methods: tuple[str, ...],
    checkpoints: dict[str, Path],
    scenarios: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    started: float,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "validation-combat-ablation-0.1.0",
        "generated_at": utc_now(),
        "status": status,
        "device": "cuda",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "planned_combats": len(scenarios),
        "completed_combats": len(rows),
        "fully_evaluated_combats": sum(
            all(method in row for method in methods) for row in rows
        ),
        "engine_unsupported_combats": [
            {
                "scenario_id": row["scenario_id"],
                "source_combat_id": row["source_combat_id"],
                "encounter": row["encounter"],
                "error": row["evaluation_error"],
            }
            for row in rows if "evaluation_error" in row
        ],
        "methods": list(methods),
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in checkpoints.items()
        },
        "comparison_semantics": {
            "between_models": "same reconstructed entrance save, encounter, ascension, and engine RNG root",
            "human_vs_model": "same visible entrance snapshot and encounter; the original human RNG root may differ",
            "encounter_adapter": "signature frozen from the first recorded combat observation",
        },
        "one_step": {
            "top_k": args.top_k,
            "determinizations": args.determinizations,
            "policy_log_weight": args.policy_log_weight,
            "minimum_value_advantage": args.minimum_value_advantage,
            "minimum_end_turn_advantage": args.minimum_end_turn_advantage,
            "cvar_alpha": args.cvar_alpha,
            "cvar_weight": args.cvar_weight,
        },
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "summary": _summary(rows, methods),
        "combats": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    methods = tuple(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise EngineError(f"unknown methods: {unknown}")
    p2_checkpoint = _resolve_checkpoint(args.p2_checkpoint or _latest_checkpoint(DEFAULT_P2))
    residual_checkpoint = _resolve_checkpoint(
        args.residual_checkpoint or _latest_checkpoint(DEFAULT_RESIDUAL)
    )
    checkpoints = {"p2": p2_checkpoint, "residual": residual_checkpoint}
    p2_model, p2_tensorizer, p2_device = _load_policy(p2_checkpoint, args.device)
    residual_model, residual_tensorizer, residual_device = _load_policy(
        residual_checkpoint, args.device
    )
    if p2_device != "cuda" or residual_device != "cuda":
        raise EngineError(
            f"CUDA is required; resolved p2={p2_device!r}, residual={residual_device!r}"
        )
    p2_objective = CombatObjective.from_config(p2_model.config)
    residual_objective = CombatObjective.from_config(residual_model.config)
    scenarios = _load_scenarios(args)
    output = args.output.resolve()
    rows: list[dict[str, Any]] = []
    if args.resume and output.exists():
        previous = load_json(output)
        if previous.get("methods") != list(methods):
            raise EngineError("resume output methods do not match the requested methods")
        rows = list(previous.get("combats") or [])
    completed_ids = {str(row["scenario_id"]) for row in rows}
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f"sts2_validation_shard{args.shard_index}_"
    ) as temp_dir:
        temp = Path(temp_dir)
        base_saves: dict[tuple[int, str, tuple[str, ...]], tuple[dict[str, Any], str, int]] = {}
        for scenario in scenarios:
            if scenario["scenario_id"] in completed_ids:
                continue
            entrance_save = temp / f"{scenario['scenario_id']}.save"
            base_key = (
                int(scenario["ascension"]),
                str(scenario["encounter"]),
                tuple(scenario["recorded_monsters"]),
            )
            if base_key not in base_saves:
                base_saves[base_key] = _find_matching_base_save(
                    args,
                    game_data_dir=game_data_dir,
                    scenario=scenario,
                    path=temp / f"matched-base-{len(base_saves):03d}.save",
                )
            base_save, reconstruction_seed, seed_trials = base_saves[base_key]
            root = _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base_save,
                scenario=scenario,
                path=entrance_save,
            )
            expected = Counter(scenario["recorded_monsters"])
            actual = Counter(str(enemy["id"]) for enemy in root["enemies"])
            if expected != actual:
                raise EngineError(
                    f"encounter composition mismatch for {scenario['scenario_id']}: "
                    f"expected={sorted(expected.elements())!r}, actual={sorted(actual.elements())!r}, "
                    f"encounter={scenario['encounter']}"
                )
            result = {
                **{key: value for key, value in scenario.items() if key != "player"},
                "snapshot": scenario["player"],
                "root": root,
                "reconstruction_seed": reconstruction_seed,
                "reconstruction_seed_trials": seed_trials,
            }
            model_specs = {
                "p2": (p2_model, p2_tensorizer, p2_device, p2_objective),
                "residual": (
                    residual_model,
                    residual_tensorizer,
                    residual_device,
                    residual_objective,
                ),
            }
            for method in methods:
                model_name = "residual" if method.startswith("residual") else "p2"
                model, tensorizer, device, objective = model_specs[model_name]
                try:
                    if method.endswith("_policy"):
                        evaluation = _run_policy(
                            args,
                            game_data_dir=game_data_dir,
                            entrance_save=entrance_save,
                            scenario=scenario,
                            model=model,
                            tensorizer=tensorizer,
                            device=device,
                            encounter_signature=scenario["encounter_signature"],
                        )
                    else:
                        evaluation = _run_one_step(
                            args,
                            game_data_dir=game_data_dir,
                            entrance_save=entrance_save,
                            scenario=scenario,
                            model=model,
                            tensorizer=tensorizer,
                            device=device,
                            objective=objective,
                            encounter_signature=scenario["encounter_signature"],
                        )
                except EngineError as exc:
                    result["evaluation_error"] = {
                        "failed_method": method,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                if evaluation["root_signature"] != root["root_signature"]:
                    raise EngineError(f"engine RNG root mismatch in {scenario['scenario_id']} {method}")
                result[method] = evaluation
            rows.append(result)
            rows.sort(key=lambda row: int(row["global_index"]))
            progress = _report(
                args=args,
                methods=methods,
                checkpoints=checkpoints,
                scenarios=scenarios,
                rows=rows,
                started=started,
                status="running",
            )
            write_json_atomic(output, progress)
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "progress": f"{len(rows)}/{len(scenarios)}",
                "human_hp_loss": scenario["human"]["hp_loss"],
                "models": {
                    method: {
                        "status": result[method]["status"],
                        "hp_loss": result[method]["hp_loss"],
                    }
                    for method in methods if method in result
                },
                "evaluation_error": result.get("evaluation_error"),
            }, ensure_ascii=False), flush=True)
    report = _report(
        args=args,
        methods=methods,
        checkpoints=checkpoints,
        scenarios=scenarios,
        rows=rows,
        started=started,
        status="pass",
    )
    write_json_atomic(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-checkpoint", type=Path)
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--combats", type=Path, default=DEFAULT_COMBATS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", default="validation-combat-ablation-v0")
    parser.add_argument("--encounter-seed-trials", type=int, default=128)
    parser.add_argument("--search-seed", type=int, default=20260818)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--policy-log-weight", type=float, default=0.05)
    parser.add_argument("--minimum-value-advantage", type=float, default=0.02)
    parser.add_argument("--minimum-end-turn-advantage", type=float, default=0.15)
    parser.add_argument("--minimum-potion-policy-probability", type=float, default=0.0)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
    parser.add_argument("--restore-mode", default="cached_batch_auto_prepared")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must satisfy 0 <= index < shard count")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "completed_combats": report["completed_combats"],
        "summary": report["summary"]["overall"],
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
