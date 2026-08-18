"""Probe Combat Tool V0 directives on real sts2-cli combat roots.

This is an interface sensitivity test, not a gameplay-strength benchmark.  It
holds each engine root fixed and changes only the typed upper-level directive.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch  # Keep Windows DLL load order stable before pyarrow imports.


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
from run_combat_mcts_act_sweep import (  # noqa: E402
    DEFAULT_TRANSITIONS,
    _create_base_save,
    _entry,
    _first_a0_ironclad_snapshots,
    _prepare_scenario_save,
    _scenario_specs,
)
from run_combat_policy_online import _load_policy, _state_summary  # noqa: E402
from sts2_dataset.combat_directive import CombatDirectiveV0  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
)
from sts2_dataset.combat_tool import CombatToolV0  # noqa: E402
from sts2_dataset.legal_actions import enumerate_legal_actions  # noqa: E402
from sts2_dataset.util import sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_tool_v0_sensitivity.json"


def _target_profile(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    observation = sample["observation"]
    enemies = [row for row in observation.get("enemies") or [] if float(row.get("hp") or 0) > 0]
    if len(enemies) < 2:
        return {}, None
    target = min(enemies, key=lambda row: (float(row.get("hp") or 0), str(row.get("entity_ref"))))
    target_ref = str(target["entity_ref"])
    return {
        "action_preferences": {"target_biases": {target_ref: 1.25}},
    }, {
        "target_ref": target_ref,
        "enemy_id": target.get("id"),
        "hp": target.get("hp"),
    }


def _profiles(sample: dict[str, Any]) -> list[dict[str, Any]]:
    focus_directive, focus_target = _target_profile(sample)
    rows = [
        {"profile": "default", "directive": {}},
        {
            "profile": "conserve_potions",
            "directive": {"resource_policy": {"max_potion_uses": 0}},
        },
        {
            "profile": "lower_potion_threshold",
            "directive": {
                "objective": {"potion_cost": 0.0},
                "action_preferences": {"action_type_biases": {"use_potion": 1.0}},
            },
        },
    ]
    if focus_directive:
        rows.append({
            "profile": "focus_lowest_hp_enemy",
            "directive": focus_directive,
            "focus_target": focus_target,
        })
    return rows


def _candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action_type": candidate.get("action_type"),
        "source_id": candidate.get("source_id"),
        "source_ref": candidate.get("source_ref"),
        "target_ref": candidate.get("target_ref"),
        "probability": row.get("probability"),
        "policy_probability": row.get("policy_probability"),
        "eligible": row.get("eligible"),
        "exclusion_reasons": row.get("exclusion_reasons"),
        "score_breakdown": row.get("score_breakdown"),
        "resource_prediction": row.get("resource_prediction"),
        "engine_preview": row.get("engine_preview"),
    }


def _action_type_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        action_type = str(row["candidate"].get("action_type") or "unknown")
        entry = result.setdefault(action_type, {
            "probability_mass": 0.0,
            "eligible_count": 0,
            "excluded_count": 0,
            "best": None,
        })
        if row["eligible"]:
            entry["eligible_count"] += 1
            entry["probability_mass"] += float(row["probability"])
            if entry["best"] is None:
                entry["best"] = _candidate_summary(row)
        else:
            entry["excluded_count"] += 1
    for entry in result.values():
        entry["probability_mass"] = round(float(entry["probability_mass"]), 8)
    return result


def _full_combat_result(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    tool: CombatToolV0,
    directive: CombatDirectiveV0,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    potion_uses = 0
    with _engine(args, game_data_dir) as engine:
        state, _ = engine.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
        state, _ = engine.send({
            "cmd": "enter_room",
            "type": "combat",
            "encounter": scenario["encounter"],
        })
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        for step_index in range(args.max_actions):
            decision_type = str(state.get("decision") or "")
            if decision_type not in SEARCH_DECISIONS:
                break
            before = _state_summary(state)
            if decision_type == "combat_play":
                sample = headless_state_to_model_sample(
                    state,
                    transition_id=f"tool-full:{scenario['scenario_id']}:{step_index}",
                    combat_id=scenario["scenario_id"],
                )
                runtime_directive = CombatDirectiveV0(
                    **{
                        **directive.__dict__,
                        "potion_uses_so_far": potion_uses,
                    }
                )
                tool_response = tool.decide(sample, directive=runtime_directive, top_k=3)
                chosen = tool_response["chosen"]
                candidate = chosen["candidate"]
                command = candidate_to_headless_command(candidate)
                if candidate.get("action_type") == "use_potion":
                    potion_uses += 1
                step_meta = {
                    "chosen_candidate": candidate,
                    "probability": chosen["probability"],
                    "predicted_risk": tool_response["predicted_risk"],
                    "request_replan": tool_response["request_replan"],
                    "inference_ms": tool_response["inference_ms"],
                }
            else:
                actions = enumerate_legal_actions(state)
                if not actions:
                    break
                action = actions[0]
                command = {"cmd": "action", "action": action.action, "args": action.args}
                step_meta = {
                    "chosen_candidate": {
                        "candidate_id": action.action_id,
                        "action_type": action.action,
                        "source_type": "card_selection_fallback",
                    },
                    "probability": 1.0,
                    "predicted_risk": None,
                    "request_replan": None,
                    "inference_ms": 0.0,
                }
            state, engine_ms = engine.send(command)
            steps.append({
                "step": step_index,
                "before": before,
                **step_meta,
                "engine_ms": round(engine_ms, 3),
                "after": _state_summary(state),
            })
    final_hp = float(((state.get("player") or {}).get("hp") or 0.0))
    final_decision = str(state.get("decision") or "")
    if final_decision == "game_over":
        status = "victory" if state.get("victory") else "death"
    elif final_decision in POST_COMBAT_DECISIONS:
        status = "combat_won"
    elif final_decision in SEARCH_DECISIONS:
        status = "step_limit"
    else:
        status = f"unsupported_subdecision:{final_decision or 'unknown'}"
    return {
        "status": status,
        "final_decision": final_decision,
        "initial_hp": initial_hp,
        "final_hp": final_hp,
        "hp_loss": max(0.0, initial_hp - final_hp),
        "decision_count": len(steps),
        "rounds": max((int(row["before"].get("round") or 0) for row in steps), default=0),
        "potion_uses": potion_uses,
        "replan_request_count": sum(
            bool((row.get("request_replan") or {}).get("required")) for row in steps
        ),
        "steps": steps,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if model.resource_value_head is None and model.state_value_head is None:
        raise EngineError(
            "Combat Tool sensitivity requires a candidate-resource or state-value checkpoint"
        )
    tool = CombatToolV0(model, tensorizer, device=device)
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, include_controls=False)
    if args.scenario_ids:
        requested = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row["scenario_id"] in requested]
        missing = requested - {row["scenario_id"] for row in scenarios}
        if missing:
            raise EngineError(f"unknown or unavailable scenario ids: {sorted(missing)}")

    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sts2_combat_tool_") as temp_dir:
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
            with _engine(args, game_data_dir) as engine:
                state, _ = engine.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
                state, _ = engine.send({
                    "cmd": "enter_room",
                    "type": "combat",
                    "encounter": scenario["encounter"],
                })
            if state.get("decision") != "combat_play":
                raise EngineError(f"scenario did not enter combat: {scenario['scenario_id']}")
            sample = headless_state_to_model_sample(
                state,
                transition_id=f"combat-tool:{scenario['scenario_id']}",
                combat_id=scenario["scenario_id"],
            )
            profile_results: list[dict[str, Any]] = []
            default_candidate_id: str | None = None
            for profile in _profiles(sample):
                directive = CombatDirectiveV0.from_dict(profile["directive"])
                decision = tool.decide(sample, directive=directive, top_k=args.top_k)
                chosen = _candidate_summary(decision["chosen"])
                if default_candidate_id is None:
                    default_candidate_id = str(chosen["candidate_id"])
                profile_results.append({
                    **{key: value for key, value in profile.items() if key != "directive"},
                    "directive": directive.to_dict(),
                    "changed_from_default": str(chosen["candidate_id"]) != default_candidate_id,
                    "chosen": chosen,
                    "top_k": [_candidate_summary(row) for row in decision["top_k"]],
                    "action_types": _action_type_summary(decision["ranked_actions"]),
                    "uncertainty": decision["uncertainty"],
                    "predicted_risk": decision["predicted_risk"],
                    "request_replan": decision["request_replan"],
                    "inference_ms": decision["inference_ms"],
                    "full_combat": (
                        _full_combat_result(
                            args,
                            game_data_dir=game_data_dir,
                            entrance_save=entrance_save,
                            scenario=scenario,
                            tool=tool,
                            directive=directive,
                        )
                        if args.full_combat else None
                    ),
                })
            results.append({
                "scenario_id": scenario["scenario_id"],
                "act": scenario["act"],
                "encounter": scenario["encounter"],
                "deck_snapshot": scenario["player"],
                "root": root,
                "visible_state": {
                    "round": state.get("round"),
                    "energy": state.get("energy"),
                    "player_hp": (state.get("player") or {}).get("hp"),
                    "hand": [_entry(row.get("id") or "") for row in state.get("hand") or []],
                    "enemies": [
                        {
                            "id": row.get("id"),
                            "hp": row.get("hp"),
                            "max_hp": row.get("max_hp"),
                            "intent": row.get("intent"),
                        }
                        for row in state.get("enemies") or []
                    ],
                },
                "profiles": profile_results,
            })
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "profiles": [
                    {
                        "profile": row["profile"],
                        "candidate_id": row["chosen"]["candidate_id"],
                        "changed": row["changed_from_default"],
                    }
                    for row in profile_results
                ],
            }, ensure_ascii=False), flush=True)
    return {
        "schema_version": "combat-tool-sensitivity-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "transitions": str(args.transitions.resolve()),
        "device": device,
        "seed": args.seed,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "scenarios": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--scenario-ids", nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--full-combat", action="store_true")
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--seed", default="combat-tool-v0")
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
