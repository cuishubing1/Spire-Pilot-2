"""Compare Human, P0, P1, and P1 one-step on one held-out run.

Each policy fight is a controlled reconstruction.  The player-visible combat
entrance snapshot (HP, max HP, gold, deck, relics, potions, Act, and encounter)
comes from HumanRecorder.  P0 and P1 load the same generated entrance save, so
their engine RNG root is identical.  The human column remains the outcome of
the original recorded fight and is not presented as a same-RNG counterfactual.
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

import torch  # Keep the known Windows DLL import order stable.
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    _game_data_dir,
)
from run_combat_mcts_act_sweep import (  # noqa: E402
    _create_base_save,
    _prepare_scenario_save,
    _run_policy,
)
from run_combat_one_step_act_sweep import _run_one_step  # noqa: E402
from run_combat_policy_online import _load_policy, _resolve_checkpoint  # noqa: E402
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.util import sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_RUN_ID = "human-20260813T153218409Z-111dbff7862d4059970daa1469aaf9fe"
DEFAULT_TRANSITIONS = REPO_ROOT / "data" / "human" / "combat_v1" / "transitions.parquet"
DEFAULT_COMBATS = REPO_ROOT / "data" / "human" / "combat_v1" / "combats.parquet"
DEFAULT_TARGETS = REPO_ROOT / "data" / "human" / "combat_value_v1" / "targets.parquet"
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts" / "heldout_a0_human_p0_p1_one_step_combat_comparison.json"
)


# HumanRecorder records the concrete monster models while sts2-cli starts a
# fight from an EncounterModel.  These are the encounters observed in the
# held-out A0 run.  Keys are order-independent monster multisets.
ENCOUNTER_BY_MONSTERS = {
    (
        "MONSTER.LEAF_SLIME_S",
        "MONSTER.TWIG_SLIME_M",
        "MONSTER.TWIG_SLIME_S",
    ): "SLIMES_WEAK",
    ("MONSTER.NIBBIT",): "NIBBITS_WEAK",
    ("MONSTER.SHRINKER_BEETLE",): "SHRINKER_BEETLE_WEAK",
    ("MONSTER.CUBEX_CONSTRUCT",): "CUBEX_CONSTRUCT_NORMAL",
    ("MONSTER.CORPSE_SLUG", "MONSTER.CORPSE_SLUG"): "CORPSE_SLUGS_WEAK",
    ("MONSTER.SLUDGE_SPINNER",): "SLUDGE_SPINNER_WEAK",
    ("MONSTER.SEAPUNK",): "SEAPUNK_WEAK",
    ("MONSTER.TOADPOLE", "MONSTER.TOADPOLE"): "TOADPOLES_WEAK",
    ("MONSTER.CALCIFIED_CULTIST", "MONSTER.SEAPUNK"): "SEAPUNK_NORMAL",
    ("MONSTER.HAUNTED_SHIP",): "HAUNTED_SHIP_NORMAL",
    ("MONSTER.PHANTASMAL_GARDENER",) * 4: "PHANTASMAL_GARDENERS_ELITE",
    ("MONSTER.CORPSE_SLUG",) * 3: "CORPSE_SLUGS_NORMAL",
    ("MONSTER.SKULKING_COLONY",): "SKULKING_COLONY_ELITE",
    ("MONSTER.TERROR_EEL",): "TERROR_EEL_ELITE",
    ("MONSTER.FOSSIL_STALKER",): "FOSSIL_STALKER_NORMAL",
    ("MONSTER.TWO_TAILED_RAT",) * 3: "TWO_TAILED_RATS_NORMAL",
    ("MONSTER.WATERFALL_GIANT",): "WATERFALL_GIANT_BOSS",
    ("MONSTER.BOWLBUG_NECTAR", "MONSTER.BOWLBUG_ROCK"): "BOWLBUGS_WEAK",
    ("MONSTER.THIEVING_HOPPER",): "THIEVING_HOPPER_WEAK",
    ("MONSTER.TUNNELER",): "TUNNELER_WEAK",
    ("MONSTER.BOWLBUG_EGG", "MONSTER.BOWLBUG_ROCK", "MONSTER.BOWLBUG_SILK"): "BOWLBUGS_NORMAL",
    ("MONSTER.BOWLBUG_ROCK", "MONSTER.BOWLBUG_SILK", "MONSTER.SLUMBERING_BEETLE"): "SLUMBERING_BEETLE_NORMAL",
    ("MONSTER.HUNTER_KILLER",): "HUNTER_KILLER_NORMAL",
    ("MONSTER.OVICOPTER",): "OVICOPTER_NORMAL",
    ("MONSTER.SPINY_TOAD",): "SPINY_TOAD_NORMAL",
    ("MONSTER.CHOMPER", "MONSTER.CHOMPER"): "CHOMPERS_NORMAL",
    ("MONSTER.THE_OBSCURA",): "THE_OBSCURA_NORMAL",
    ("MONSTER.EXOSKELETON",) * 4: "EXOSKELETONS_NORMAL",
    ("MONSTER.THE_INSATIABLE",): "THE_INSATIABLE_BOSS",
    ("MONSTER.LOUSE_PROGENITOR",): "LOUSE_PROGENITOR_NORMAL",
    ("MONSTER.KNOWLEDGE_DEMON",): "KNOWLEDGE_DEMON_BOSS",
    ("MONSTER.LIVING_SHIELD", "MONSTER.TURRET_OPERATOR"): "TURRET_OPERATOR_WEAK",
    ("MONSTER.DEVOTED_SCULPTOR",): "DEVOTED_SCULPTOR_WEAK",
    ("MONSTER.SCROLL_OF_BITING",) * 4: "SCROLLS_OF_BITING_NORMAL",
    ("MONSTER.FLAIL_KNIGHT", "MONSTER.MAGI_KNIGHT", "MONSTER.SPECTRAL_KNIGHT"): "KNIGHTS_ELITE",
    ("MONSTER.CUBEX_CONSTRUCT", "MONSTER.CUBEX_CONSTRUCT", "MONSTER.PUNCH_CONSTRUCT"): "CONSTRUCT_MENAGERIE_NORMAL",
    ("MONSTER.MECHA_KNIGHT",): "MECHA_KNIGHT_ELITE",
    ("MONSTER.GLOBE_HEAD",): "GLOBE_HEAD_NORMAL",
    ("MONSTER.QUEEN", "MONSTER.TORCH_HEAD_AMALGAM"): "QUEEN_BOSS",
    ("MONSTER.SOUL_NEXUS",): "SOUL_NEXUS_ELITE",
    ("MONSTER.AEONGLASS",): "AEONGLASS_BOSS",
    ("MONSTER.FUZZY_WURM_CRAWLER",): "FUZZY_WURM_CRAWLER_WEAK",
    ("MONSTER.ENTOMANCER",): "ENTOMANCER_ELITE",
    ("MONSTER.LAGAVULIN_MATRIARCH",): "LAGAVULIN_MATRIARCH_BOSS",
    ("MONSTER.BYGONE_EFFIGY",): "BYGONE_EFFIGY_ELITE",
    ("MONSTER.SEWER_CLAM",): "SEWER_CLAM_NORMAL",
    ("MONSTER.MYTE", "MONSTER.MYTE"): "MYTES_NORMAL",
    ("MONSTER.SCROLL_OF_BITING",) * 3: "SCROLLS_OF_BITING_WEAK",
    ("MONSTER.FROG_KNIGHT",): "FROG_KNIGHT_NORMAL",
    ("MONSTER.CRUSHER", "MONSTER.ROCKET"): "KAISER_CRAB_BOSS",
    ("MONSTER.FOGMOG",): "FOGMOG_NORMAL",
    ("MONSTER.THE_FORGOTTEN", "MONSTER.THE_LOST"): "THE_LOST_AND_FORGOTTEN_NORMAL",
    ("MONSTER.FABRICATOR",): "FABRICATOR_NORMAL",
    ("MONSTER.VANTOM",): "VANTOM_BOSS",
    (
        "MONSTER.BOWLBUG_NECTAR",
        "MONSTER.BOWLBUG_ROCK",
        "MONSTER.BOWLBUG_SILK",
    ): "BOWLBUGS_NORMAL",
    ("MONSTER.TEST_SUBJECT",): "TEST_SUBJECT_BOSS",
    (
        "MONSTER.KIN_FOLLOWER",
        "MONSTER.KIN_FOLLOWER",
        "MONSTER.KIN_PRIEST",
    ): "THE_KIN_BOSS",
    ("MONSTER.OWL_MAGISTRATE",): "OWL_MAGISTRATE_NORMAL",
    ("MONSTER.BOWLBUG_EGG", "MONSTER.BOWLBUG_ROCK"): "BOWLBUGS_WEAK",
    ("MONSTER.BYRDONIS",): "BYRDONIS_ELITE",
}


def _entry(model_id: str) -> str:
    return str(model_id).split(".", 1)[-1]


def _monster_key(observation: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(enemy["id"]) for enemy in observation["combat"].get("enemies") or []))


def _player_snapshot(observation: dict[str, Any]) -> dict[str, Any]:
    player = observation["player"]
    return {
        "hp": int(player["hp"]),
        "max_hp": int(player["max_hp"]),
        "gold": int(player.get("gold") or 0),
        "deck": [
            {"id": _entry(card["id"]), "upgrade_level": int(card.get("upgrade_level") or 0)}
            for card in player.get("deck") or []
        ],
        "relics": [_entry(relic["id"]) for relic in player.get("relics") or []],
        "potions": [_entry(potion["id"]) for potion in player.get("potions") or []],
    }


def _load_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    combat_rows = {
        str(row["combat_id"]): row
        for row in pq.read_table(args.combats.resolve()).to_pylist()
        if row["run_id"] == args.run_id
    }
    transition_rows = [
        row
        for row in pq.read_table(args.transitions.resolve()).to_pylist()
        if row["run_id"] == args.run_id and row["is_training_eligible"]
    ]
    first_by_combat: dict[str, dict[str, Any]] = {}
    for row in transition_rows:
        combat_id = str(row["combat_id"])
        previous = first_by_combat.get(combat_id)
        if previous is None or int(row["record_sequence"]) < int(previous["record_sequence"]):
            first_by_combat[combat_id] = row

    targets = {
        str(row["transition_id"]): row
        for row in pq.read_table(args.targets.resolve()).to_pylist()
    }
    scenarios: list[dict[str, Any]] = []
    ordered = sorted(
        first_by_combat.items(),
        key=lambda item: (
            int(combat_rows[item[0]]["act"]),
            int(combat_rows[item[0]]["floor"]),
            int(item[1]["record_sequence"]),
        ),
    )
    for combat_index, (combat_id, row) in enumerate(ordered):
        observation = json.loads(row["observation_json"])
        combat = combat_rows[combat_id]
        scenario_id = f"heldout-{combat_index:02d}-act{combat['act']}-floor{combat['floor']}"
        if args.scenario_ids and scenario_id not in set(args.scenario_ids):
            continue
        monsters = _monster_key(observation)
        encounter = ENCOUNTER_BY_MONSTERS.get(monsters)
        if encounter is None:
            raise EngineError(f"no EncounterModel mapping for {combat_id}: {monsters!r}")
        target = targets[str(row["transition_id"])]
        scenarios.append({
            "scenario_id": scenario_id,
            "combat_index": combat_index,
            "source_combat_id": combat_id,
            "source_transition_id": row["transition_id"],
            "act": int(combat["act"]),
            "floor": int(combat["floor"]),
            "room_type": combat["room_type"],
            "encounter": encounter,
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
        raise EngineError(f"no eligible combat transitions for run {args.run_id}")
    return scenarios


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    results = [row[key] for row in rows]
    hp_losses = [float(result["hp_loss"]) for result in results]
    completed = sum(
        (
            not bool(result.get("death"))
            if "status" not in result
            else result.get("status") in {"complete", "combat_won", "victory"}
        )
        for result in results
    )
    return {
        "combats": len(results),
        "completed": completed,
        "deaths": sum(bool(result.get("death")) or result.get("status") == "death" for result in results),
        "total_hp_loss": round(sum(hp_losses), 3),
        "mean_hp_loss": round(statistics.fmean(hp_losses), 3) if hp_losses else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    p0_checkpoint = _resolve_checkpoint(args.p0_checkpoint)
    p1_checkpoint = _resolve_checkpoint(args.p1_checkpoint)
    p0_model, p0_tensorizer, p0_device = _load_policy(p0_checkpoint, args.device)
    p1_model, p1_tensorizer, p1_device = _load_policy(p1_checkpoint, args.device)
    if p0_device != "cuda" or p1_device != "cuda":
        raise EngineError(f"CUDA is required; resolved P0={p0_device!r}, P1={p1_device!r}")
    if p1_model.state_value_head is None:
        raise EngineError("P1 one-step evaluation requires an independent state value head")
    objective = CombatObjective.from_config(p1_model.config)

    scenarios = _load_scenarios(args)
    if args.scenario_ids:
        wanted = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row["scenario_id"] in wanted]
        if not scenarios:
            raise EngineError(f"no held-out scenarios matched: {sorted(wanted)}")
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sts2_heldout_combat_compare_") as temp_dir:
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
            expected_monsters = Counter(scenario["recorded_monsters"])
            actual_monsters = Counter(str(enemy["id"]) for enemy in root["enemies"])
            root["recorded_monsters"] = list(scenario["recorded_monsters"])
            root["generated_monsters"] = sorted(actual_monsters.elements())
            root["monster_composition_match"] = expected_monsters == actual_monsters
            p0 = _run_policy(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=p0_model,
                tensorizer=p0_tensorizer,
                device=p0_device,
            )
            p1 = _run_policy(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=p1_model,
                tensorizer=p1_tensorizer,
                device=p1_device,
            )
            p1_one_step = _run_one_step(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=p1_model,
                tensorizer=p1_tensorizer,
                device=p1_device,
                objective=objective,
            )
            signatures = {
                root["root_signature"],
                p0["root_signature"],
                p1["root_signature"],
                p1_one_step["root_signature"],
            }
            if len(signatures) != 1:
                raise EngineError(f"P0/P1/one-step RNG root mismatch in {scenario['scenario_id']}")
            result = {
                **{key: value for key, value in scenario.items() if key != "player"},
                "snapshot": scenario["player"],
                "root": root,
                "p0": p0,
                "p1": p1,
                "p1_one_step": p1_one_step,
            }
            results.append(result)
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "encounter": scenario["encounter"],
                "human_hp_loss": scenario["human"]["hp_loss"],
                "p0": {"status": p0["status"], "hp_loss": p0["hp_loss"]},
                "p1": {"status": p1["status"], "hp_loss": p1["hp_loss"]},
                "p1_one_step": {
                    "status": p1_one_step["status"],
                    "hp_loss": p1_one_step["hp_loss"],
                    "changes": p1_one_step["policy_action_change_count"],
                    "lookahead_ms": p1_one_step["total_lookahead_ms"],
                },
            }, ensure_ascii=False), flush=True)

    by_act: dict[str, Any] = {}
    for act in (1, 2, 3):
        act_rows = [row for row in results if int(row["act"]) == act]
        by_act[str(act)] = {
            "human": _aggregate(act_rows, "human"),
            "p0": _aggregate(act_rows, "p0"),
            "p1": _aggregate(act_rows, "p1"),
            "p1_one_step": _aggregate(act_rows, "p1_one_step"),
        }
    return {
        "schema_version": "heldout-run-combat-comparison-0.2.0",
        "generated_at": utc_now(),
        "status": "pass",
        "run_id": args.run_id,
        "comparison_semantics": {
            "p0_vs_p1": "same visible entrance snapshot, generated save, encounter, and engine RNG root",
            "p1_vs_one_step": "same public root and generated save; one-step uses visible draw multiset determinizations only",
            "human_vs_policy": "same visible entrance snapshot and encounter; original human RNG root may differ",
            "reconstruction_limitations": [
                "relic and potion model IDs are restored, but arbitrary runtime counters are not",
                "the original hidden draw order and RNG state are not copied into model input or generated save",
            ],
        },
        "device": "cuda",
        "p0_checkpoint": str(p0_checkpoint),
        "p0_checkpoint_sha256": sha256_file(p0_checkpoint),
        "p1_checkpoint": str(p1_checkpoint),
        "p1_checkpoint_sha256": sha256_file(p1_checkpoint),
        "one_step": {
            "top_k": args.top_k,
            "policy_log_weight": args.policy_log_weight,
            "minimum_value_advantage": args.minimum_value_advantage,
            "minimum_end_turn_advantage": args.minimum_end_turn_advantage,
            "determinizations": args.determinizations,
            "cvar_alpha": args.cvar_alpha,
            "cvar_weight": args.cvar_weight,
        },
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "summary": {
            "human": _aggregate(results, "human"),
            "p0": _aggregate(results, "p0"),
            "p1": _aggregate(results, "p1"),
            "p1_one_step": {
                **_aggregate(results, "p1_one_step"),
                "policy_action_changes": sum(
                    int(row["p1_one_step"]["policy_action_change_count"])
                    for row in results
                ),
                "lookahead_decisions": sum(
                    int(row["p1_one_step"]["lookahead_decision_count"])
                    for row in results
                ),
                "total_lookahead_ms": round(sum(
                    float(row["p1_one_step"]["total_lookahead_ms"])
                    for row in results
                ), 3),
            },
            "by_act": by_act,
        },
        "combats": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--p0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--combats", type=Path, default=DEFAULT_COMBATS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scenario-id", dest="scenario_ids", action="append")
    parser.add_argument("--seed", default="heldout-a0-controlled-reconstruction-v0")
    parser.add_argument("--search-seed", type=int, default=20260816)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--policy-log-weight", type=float, default=0.05)
    parser.add_argument("--minimum-value-advantage", type=float, default=0.02)
    parser.add_argument("--minimum-end-turn-advantage", type=float, default=0.15)
    parser.add_argument(
        "--minimum-potion-policy-probability",
        type=float,
        default=0.0,
        help=(
            "Optional behavior-policy support floor for search-introduced potion use; "
            "zero keeps rare but tactically necessary potion actions eligible."
        ),
    )
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "summary": report["summary"],
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
