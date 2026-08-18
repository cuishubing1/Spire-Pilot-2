"""Run Combat Policy V0 through full runs with recorded non-combat plans."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

# Keep the known Windows DLL import order stable.
import torch
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    EngineProcess,
    _game_data_dir,
)
from benchmark_policy_guided_mcts import _cache_key  # noqa: E402
from run_combat_one_step_act_sweep import one_step_current_root  # noqa: E402
from run_combat_turn_boundary_eval import turn_boundary_current_root  # noqa: E402
from run_combat_policy_online import (  # noqa: E402
    _add_objective_arguments,
    _load_policy,
    _objective_from_args,
    _rank_actions,
    _resolve_checkpoint,
    _state_summary,
)
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    first_card_select_candidate,
    headless_state_to_model_sample,
    visible_intent_end_turn_hp_loss,
)
from sts2_dataset.fixed_plan import build_fixed_noncombat_plan, fixed_plan_command  # noqa: E402
from sts2_dataset.human import HumanRecordingError  # noqa: E402
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.util import sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_TEMPLATES = (
    "human-20260813T153218409Z-111dbff7862d4059970daa1469aaf9fe",  # held-out A0 win
)
EPISODES_PATH = REPO_ROOT / "data" / "human" / "dataset" / "episodes.parquet"
TRANSITIONS_PATH = REPO_ROOT / "data" / "human" / "dataset" / "transitions.parquet"
COMBATS_PATH = REPO_ROOT / "data" / "human" / "combat_v1" / "combats.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_policy_fixed_plan_a0.json"


def _position(state: dict[str, Any]) -> tuple[int, int]:
    context = state.get("context") or {}
    act = int(context.get("act", state.get("act", 0)) or 0)
    floor = int(context.get("total_floor", context.get("floor", state.get("floor", 0))) or 0)
    return act, floor


def _adaptive_turn_boundary_trigger(
    state: dict[str, Any],
    *,
    minimum_hp_loss: float,
    minimum_hp_fraction: float,
) -> dict[str, Any]:
    """Ground a high-risk search trigger in visible intent and current HP."""

    if minimum_hp_loss < 0.0:
        raise ValueError("adaptive turn-boundary minimum HP loss must be non-negative")
    if not 0.0 <= minimum_hp_fraction <= 1.0:
        raise ValueError("adaptive turn-boundary HP fraction must be in [0, 1]")
    sample = headless_state_to_model_sample(
        state,
        transition_id="adaptive-turn-boundary:trigger",
        combat_id="adaptive-turn-boundary",
    )
    observation = sample["observation"]
    estimate = visible_intent_end_turn_hp_loss(observation)
    hp = float((observation.get("global") or {}).get("hp") or 0.0)
    hp_loss = float((estimate or {}).get("hp_loss") or 0.0)
    fraction = hp_loss / max(hp, 1.0)
    triggered = hp_loss >= minimum_hp_loss and fraction >= minimum_hp_fraction
    return {
        "triggered": triggered,
        "visible_end_turn_hp_loss": hp_loss,
        "current_hp": hp,
        "current_hp_fraction": fraction,
        "minimum_hp_loss": float(minimum_hp_loss),
        "minimum_hp_fraction": float(minimum_hp_fraction),
    }


def _brief_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        "probability": row["probability"],
        "policy_probability": row.get("policy_probability"),
        "action_type": candidate["action_type"],
        "source_id": candidate.get("source_id"),
        "source_index": candidate.get("source_index"),
        "target_id": candidate.get("target_id"),
        "target_index": candidate.get("target_index"),
        "resource_prediction": row.get("resource_prediction"),
    }


def _combat_visible_snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the visible fields needed to audit one policy combat decision."""
    if state.get("decision") != "combat_play":
        return None
    player = state.get("player") or {}
    return {
        "round": state.get("round"),
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": state.get("energy"),
            "powers": state.get("player_powers"),
        },
        "hand": [
            {
                "index": value.get("index"),
                "id": value.get("id"),
                "name": value.get("name"),
                "cost": value.get("cost"),
                "can_play": value.get("can_play"),
                "target_type": value.get("target_type"),
                "stats": value.get("stats"),
                "damage_by_target": value.get("damage_by_target"),
            }
            for value in state.get("hand") or [] if isinstance(value, dict)
        ],
        "enemies": [
            {
                "index": value.get("index"),
                "id": value.get("id"),
                "name": value.get("name"),
                "hp": value.get("hp"),
                "max_hp": value.get("max_hp"),
                "block": value.get("block"),
                "intents": value.get("intents"),
                "powers": value.get("powers"),
            }
            for value in state.get("enemies") or [] if isinstance(value, dict)
        ],
        "piles": {
            "draw_count": state.get("draw_pile_count"),
            "discard_count": state.get("discard_pile_count"),
            "exhaust_count": state.get("exhaust_pile_count"),
        },
    }


def _decision_details(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": state.get("decision"),
        "position": _position(state),
        "cards": [
            {"index": value.get("index"), "id": value.get("id"), "upgraded": value.get("upgraded")}
            for value in state.get("cards") or [] if isinstance(value, dict)
        ],
        "options": [
            {
                "index": value.get("index"),
                "option_id": value.get("option_id"),
                "is_locked": value.get("is_locked"),
                "is_enabled": value.get("is_enabled"),
            }
            for value in state.get("options") or [] if isinstance(value, dict)
        ],
        "shop": {
            name: [
                {"index": value.get("index"), "id": value.get("id"), "cost": value.get("cost")}
                for value in state.get(name) or [] if isinstance(value, dict)
            ]
            for name in ("cards", "relics", "potions")
        } if state.get("decision") == "shop" else None,
    }


def _fallback_command(state: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any] | None:
    """Deterministic, non-learning fallback when a recorded object disappeared."""
    decision = str(state.get("decision") or "")
    action_id = str((entry.get("action") or {}).get("action_id") or "")
    if decision == "card_select":
        cards = [value for value in state.get("cards") or [] if isinstance(value, dict)]
        count = int(state.get("min_select") or 0)
        indices = [int(value["index"]) for value in cards[:count]]
        if len(indices) != count:
            return None
        return {"cmd": "action", "action": "select_cards", "args": {"indices": ",".join(map(str, indices))}}
    if decision == "card_reward":
        if action_id == "choose_reward_alternative" or not state.get("cards"):
            return {"cmd": "action", "action": "skip_card_reward"}
        cards = [value for value in state.get("cards") or [] if isinstance(value, dict)]
        return {"cmd": "action", "action": "select_card_reward", "args": {"card_index": int(cards[0]["index"])}}
    if decision in {"event_choice", "rest_site"}:
        options = [
            value for value in state.get("options") or []
            if not value.get("is_locked") and value.get("is_enabled") is not False
        ]
        if not options:
            return None
        return {"cmd": "action", "action": "choose_option", "args": {"option_index": int(options[0]["index"])}}
    if decision == "shop" and action_id in {"buy_shop_item", "remove_card"}:
        # Consume the unavailable purchase but keep the shop open for the next
        # recorded action, normally another purchase or leave_shop.
        return {}
    return None


def _support_command(
    state: dict[str, Any], *, prefer_event_exit: bool = False
) -> dict[str, Any] | None:
    """Resolve an engine prompt created by the policy but absent from the source plan."""
    decision = str(state.get("decision") or "")
    if decision == "card_select":
        cards = [value for value in state.get("cards") or [] if isinstance(value, dict)]
        count = int(state.get("min_select") or 0)
        if len(cards) < count:
            return None
        indices = [int(value["index"]) for value in cards[:count]]
        return {"cmd": "action", "action": "select_cards", "args": {"indices": ",".join(map(str, indices))}}
    if decision == "bundle_select":
        bundles = [value for value in state.get("bundles") or [] if isinstance(value, dict)]
        if not bundles:
            return None
        return {"cmd": "action", "action": "select_bundle", "args": {"bundle_index": int(bundles[0]["index"])}}
    if decision == "event_choice":
        options = [value for value in state.get("options") or [] if not value.get("is_locked")]
        if not options:
            return None
        selected = options[-1] if prefer_event_exit else options[0]
        return {
            "cmd": "action",
            "action": "choose_option",
            "args": {"option_index": int(selected["index"])},
        }
    if decision == "card_reward":
        return {"cmd": "action", "action": "skip_card_reward"}
    return None


def _engine(args: argparse.Namespace, game_data_dir: Path) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet.resolve(),
        engine_dll=args.engine_dll.resolve(),
        game_data_dir=game_data_dir,
        sts2_lib=args.sts2_lib.resolve(),
        timeout_s=args.timeout,
    )


def _finish_combat(active: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any] | None:
    if active is None:
        return None
    player = state.get("player") or {}
    active["end_hp"] = player.get("hp")
    active["end_decision"] = state.get("decision")
    if isinstance(active.get("start_hp"), int) and isinstance(active.get("end_hp"), int):
        active["net_hp_change"] = active["end_hp"] - active["start_hp"]
    return active


def _search_failure_policy_lookahead(
    *,
    mode: str,
    error: EngineError,
    state: dict[str, Any],
    run_id: str,
    decision_index: int,
    combat_index: int,
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
) -> dict[str, Any]:
    """Represent a failed search root as an auditable P1 policy fallback."""
    sample = headless_state_to_model_sample(
        state,
        transition_id=f"fixed-search-fallback:{run_id}:{decision_index}",
        combat_id=f"fixed:{run_id}:{combat_index}",
    )
    ranked, inference_ms = _rank_actions(
        model, tensorizer, sample, device=device, objective=objective
    )
    policy_candidate = ranked[0]["candidate"]
    report: dict[str, Any] = {
        "status": "engine_restore_fallback",
        "fallback_policy": "p1",
        "error_type": type(error).__name__,
        "error": str(error),
        "chosen_candidate": policy_candidate,
        "policy_candidate": policy_candidate,
        "root_inference_ms": float(inference_ms),
        "wall_ms": 0.0,
        "evaluations": [
            {
                "candidate": row["candidate"],
                "policy_probability": float(row["policy_probability"]),
            }
            for row in ranked
        ],
    }
    if mode == "turn_boundary":
        report.update({"value_inference_ms": 0.0, "expanded_paths": 0})
    elif mode == "one_step":
        report["successor_value_inference_ms"] = 0.0
    else:
        raise ValueError(f"unsupported search fallback mode: {mode!r}")
    return report


def run_episode(
    args: argparse.Namespace,
    *,
    episode: dict[str, Any],
    source_transitions: list[dict[str, Any]],
    source_combats: list[dict[str, Any]],
    model: Any,
    tensorizer: Any,
    device: str,
    game_data_dir: Path,
    resume_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = build_fixed_noncombat_plan(source_transitions)
    objective = _objective_from_args(model, args)
    search_mode = (
        "turn_boundary"
        if args.turn_boundary
        else "adaptive"
        if args.adaptive_turn_boundary
        else "one_step"
        if args.one_step
        else "policy"
    )
    search_enabled = search_mode != "policy"
    lookahead_objective = CombatObjective.from_config(model.config) if search_enabled else None
    if search_enabled and model.state_value_head is None:
        raise HumanRecordingError(
            f"{search_mode} fixed-plan evaluation requires a state value head"
        )
    run_id = str(episode["run_id"])
    badges = json.loads(episode.get("badge_ids_json") or "[]")
    result: dict[str, Any] = {
        "template_run_id": run_id,
        "seed": episode["seed"],
        "source_ascension": int(episode["ascension"]),
        "execution_ascension": args.ascension,
        "source_victory": episode.get("victory"),
        "badges": badges,
        "source_split_combats": dict(Counter(value["split"] for value in source_combats)),
        "plan_actions": len(plan),
        "plan_actions_consumed": 0,
        "status": "running",
        "combat_actions": 0,
        "fixed_actions": 0,
        "fallback_actions": 0,
        "support_actions": 0,
        "one_step_enabled": bool(args.one_step),
        "one_step_decisions": 0,
        "one_step_action_changes": 0,
        "turn_boundary_enabled": bool(args.turn_boundary),
        "adaptive_turn_boundary_enabled": bool(args.adaptive_turn_boundary),
        "adaptive_turn_boundary_triggers": 0,
        "turn_boundary_decisions": 0,
        "turn_boundary_action_changes": 0,
        "turn_boundary_expanded_paths": 0,
        "search_fallbacks": 0,
        "search_restore_failures": 0,
        "combat_engine_rejections": 0,
        "combats": [],
        "trace": [],
    }
    cursor = 0
    active_combat: dict[str, Any] | None = None
    state: dict[str, Any] = {}
    plan_entry: dict[str, Any] | None = None
    inference_ms_values: list[float] = []
    engine_ms_values: list[float] = []
    lookahead_ms_values: list[float] = []
    pending_combat_entrance: dict[str, Any] | None = None
    combat_entrance_save: Path | None = None
    combat_enter_command: dict[str, Any] | None = None
    combat_prefix: list[dict[str, Any]] = []
    prior_lookahead_ms = 0.0
    decision_start = 0
    try:
        with tempfile.TemporaryDirectory(prefix="sts2_fixed_plan_") as temp_dir, ExitStack() as stack:
            engine = stack.enter_context(_engine(args, game_data_dir))
            worker_count = (
                args.search_workers
                if args.turn_boundary or args.adaptive_turn_boundary
                else 1
            )
            workers = [
                stack.enter_context(_engine(args, game_data_dir))
                for _ in range(worker_count)
            ]
            worker = workers[0]
            temp_path = Path(temp_dir)
            result["engine_startup_ms"] = round(engine.startup_ms, 3)
            result["worker_startup_ms"] = round(worker.startup_ms, 3)
            result["worker_startup_ms_all"] = [
                round(value.startup_ms, 3) for value in workers
            ]
            state, start_ms = engine.send({
                "cmd": "start_run",
                "character": "Ironclad",
                "ascension": args.ascension,
                "seed": episode["seed"],
                "lang": "en",
                "badges": badges,
            })
            result["start_run_ms"] = round(start_ms, 3)
            if resume_snapshot is not None:
                replay_fields = (
                    "plan_actions_consumed",
                    "combat_actions",
                    "fixed_actions",
                    "fallback_actions",
                    "support_actions",
                    "one_step_decisions",
                    "one_step_action_changes",
                    "turn_boundary_decisions",
                    "turn_boundary_action_changes",
                    "turn_boundary_expanded_paths",
                    "adaptive_turn_boundary_triggers",
                    "search_fallbacks",
                    "search_restore_failures",
                    "combat_engine_rejections",
                    "max_act",
                    "max_floor",
                    "combats",
                    "trace",
                )
                for field in replay_fields:
                    if field in resume_snapshot:
                        result[field] = json.loads(json.dumps(resume_snapshot[field]))
                cursor = int(result["plan_actions_consumed"])
                prior_lookahead_ms = float(resume_snapshot.get("total_lookahead_ms") or 0.0)
                result["resume_prior_metrics"] = {
                    "mean_inference_ms": resume_snapshot.get("mean_inference_ms"),
                    "mean_engine_ms": resume_snapshot.get("mean_engine_ms"),
                    "mean_lookahead_ms": resume_snapshot.get("mean_lookahead_ms"),
                    "total_lookahead_ms": prior_lookahead_ms,
                }

                for replay_index, trace_entry in enumerate(result["trace"]):
                    command = trace_entry.get("command")
                    if not command or trace_entry.get("source") == "engine_rejected_shop_action":
                        continue
                    entrance_path: Path | None = None
                    if state.get("decision") in {"map_select", "event_choice"}:
                        entrance_path = temp_path / f"resume-entrance-{replay_index}.save"
                        saved, _ = engine.send({
                            "cmd": "write_continue_save", "path": str(entrance_path)
                        })
                        if not saved.get("success"):
                            raise HumanRecordingError(
                                f"resume replay could not save entrance {replay_index}: {saved!r}"
                            )
                    before_decision = state.get("decision")
                    state, _ = engine.send(command)
                    if (
                        entrance_path is not None
                        and before_decision in {"map_select", "event_choice"}
                        and state.get("decision") == "combat_play"
                    ):
                        pending_combat_entrance = {
                            "path": entrance_path,
                            "command": command,
                        }

                expected = resume_snapshot.get("blocked_state") or {}
                expected_position = tuple(expected.get("position") or ())
                if expected.get("decision") and state.get("decision") != expected["decision"]:
                    raise HumanRecordingError(
                        f"resume replay decision mismatch: {state.get('decision')!r} "
                        f"!= {expected['decision']!r}"
                    )
                if expected_position and _position(state) != expected_position:
                    raise HumanRecordingError(
                        f"resume replay position mismatch: {_position(state)!r} "
                        f"!= {expected_position!r}"
                    )
                prior_indices = [
                    int(row["decision_index"])
                    for row in result["trace"]
                    if row.get("decision_index") is not None
                ]
                decision_start = max(prior_indices, default=-1) + 1
                result["resumed_from"] = {
                    "decision_index": decision_start,
                    "decision": state.get("decision"),
                    "position": list(_position(state)),
                    "replayed_commands": sum(
                        bool(row.get("command"))
                        and row.get("source") != "engine_rejected_shop_action"
                        for row in result["trace"]
                    ),
                }
            # HumanRecorder sees the explicit floor-0 start-node click before
            # Neow.  Sts2Headless StartRun has already entered that unique node
            # and exposes Neow directly, so record exactly that one operation as
            # an engine-implicit plan action.
            if (
                resume_snapshot is None
                and
                state.get("decision") == "event_choice"
                and plan
                and plan[0]["phase"] == "map_select"
                and int(plan[0]["source_act"]) == 1
                and int(plan[0]["source_floor"]) == 0
            ):
                result["trace"].append({
                    "decision_index": -1,
                    "act": 1,
                    "floor": 0,
                    "decision": "map_select",
                    "source": "engine_implicit_initial_node",
                    "selected": {
                        "source_action": plan[0]["action"],
                        "source_record_sequence": plan[0]["record_sequence"],
                    },
                    "after_decision": "event_choice",
                })
                cursor = 1
                result["plan_actions_consumed"] = cursor
            for decision_index in range(decision_start, args.max_decisions):
                decision = str(state.get("decision") or "")
                act, floor = _position(state)
                result["max_act"] = max(int(result.get("max_act", 0)), act)
                result["max_floor"] = max(int(result.get("max_floor", 0)), floor)

                if decision == "game_over":
                    if active_combat is not None:
                        result["combats"].append(_finish_combat(active_combat, state))
                        active_combat = None
                    pending_combat_entrance = None
                    combat_entrance_save = None
                    combat_enter_command = None
                    combat_prefix = []
                    result["status"] = "victory" if state.get("victory") else "death"
                    result["victory"] = bool(state.get("victory"))
                    result["final_state"] = _state_summary(state)
                    break

                if decision == "combat_play":
                    if active_combat is None:
                        player = state.get("player") or {}
                        active_combat = {
                            "combat_index": len(result["combats"]),
                            "act": act,
                            "floor": floor,
                            "room_type": (state.get("context") or {}).get("room_type"),
                            "start_hp": player.get("hp"),
                            "enemies": [value.get("id") for value in state.get("enemies") or []],
                            "actions": 0,
                            "one_step_decisions": 0,
                            "one_step_action_changes": 0,
                            "turn_boundary_decisions": 0,
                            "turn_boundary_action_changes": 0,
                            "turn_boundary_expanded_paths": 0,
                            "adaptive_turn_boundary_triggers": 0,
                            "search_fallbacks": 0,
                            "search_restore_failures": 0,
                            "search_disabled_error": None,
                            "combat_engine_rejections": 0,
                        }
                        if search_enabled:
                            if pending_combat_entrance is None:
                                raise HumanRecordingError(
                                    f"{search_mode} combat started without a captured entrance save"
                                )
                            combat_entrance_save = pending_combat_entrance["path"]
                            combat_enter_command = pending_combat_entrance["command"]
                            combat_prefix = []
                            for worker_index, search_worker in enumerate(workers):
                                cached, _ = search_worker.send({
                                    "cmd": "cache_save",
                                    "name": _cache_key(combat_entrance_save),
                                    "path": str(combat_entrance_save),
                                })
                                if cached.get("type") != "ok":
                                    raise HumanRecordingError(
                                        f"{search_mode} worker {worker_index} could not cache "
                                        f"combat entrance: {cached!r}"
                                    )
                            pending_combat_entrance = None
                    if int(active_combat["actions"]) >= args.max_combat_actions:
                        raise HumanRecordingError("combat action limit exceeded")
                    lookahead = None
                    adaptive_trigger = None
                    if args.adaptive_turn_boundary:
                        adaptive_trigger = _adaptive_turn_boundary_trigger(
                            state,
                            minimum_hp_loss=args.adaptive_minimum_hp_loss,
                            minimum_hp_fraction=args.adaptive_minimum_hp_fraction,
                        )
                        if adaptive_trigger["triggered"]:
                            result["adaptive_turn_boundary_triggers"] += 1
                            active_combat["adaptive_turn_boundary_triggers"] += 1
                    if args.turn_boundary or bool(
                        adaptive_trigger and adaptive_trigger["triggered"]
                    ):
                        assert combat_entrance_save is not None
                        assert combat_enter_command is not None
                        assert lookahead_objective is not None
                        disabled_error = active_combat.get("search_disabled_error")
                        if disabled_error:
                            lookahead = _search_failure_policy_lookahead(
                                mode="turn_boundary",
                                error=EngineError(str(disabled_error)),
                                state=state,
                                run_id=run_id,
                                decision_index=decision_index,
                                combat_index=len(result["combats"]),
                                model=model,
                                tensorizer=tensorizer,
                                device=device,
                                objective=objective,
                            )
                        else:
                            try:
                                lookahead = turn_boundary_current_root(
                                workers=workers,
                                entrance_save=combat_entrance_save,
                                enter_command=combat_enter_command,
                                root_prefix=combat_prefix,
                                root_state=state,
                                model=model,
                                tensorizer=tensorizer,
                                device=device,
                                objective=lookahead_objective,
                                root_top_k=args.root_top_k,
                                beam_width=args.beam_width,
                                max_player_actions=args.max_player_actions,
                                policy_log_weight=args.policy_log_weight,
                                continuation_policy_weight=args.continuation_policy_weight,
                                minimum_value_advantage=args.minimum_value_advantage,
                                minimum_end_turn_advantage=args.minimum_end_turn_advantage,
                                unsupported_penalty=args.unsupported_penalty,
                                determinization_count=args.determinizations,
                                cvar_alpha=args.cvar_alpha,
                                cvar_weight=args.cvar_weight,
                                search_seed=args.search_seed + decision_index * 100003,
                                step=decision_index,
                                )
                            except EngineError as exc:
                                active_combat["search_disabled_error"] = str(exc)
                                result["search_restore_failures"] += 1
                                active_combat["search_restore_failures"] += 1
                                lookahead = _search_failure_policy_lookahead(
                                    mode="turn_boundary",
                                    error=exc,
                                    state=state,
                                    run_id=run_id,
                                    decision_index=decision_index,
                                    combat_index=len(result["combats"]),
                                    model=model,
                                    tensorizer=tensorizer,
                                    device=device,
                                    objective=objective,
                                )
                        if lookahead.get("status") == "engine_restore_fallback":
                            result["search_fallbacks"] += 1
                            active_combat["search_fallbacks"] += 1
                        ranked = [
                            {
                                "candidate": value["candidate"],
                                "probability": value["policy_probability"],
                                "policy_probability": value["policy_probability"],
                                "resource_prediction": None,
                            }
                            for value in lookahead["evaluations"]
                        ]
                        selected = next(
                            value for value in ranked
                            if value["candidate"]["candidate_id"]
                            == lookahead["chosen_candidate"]["candidate_id"]
                        )
                        inference_ms = float(lookahead["root_inference_ms"]) + float(
                            lookahead["value_inference_ms"]
                        )
                        lookahead_ms_values.append(float(lookahead["wall_ms"]))
                        changed = (
                            lookahead["chosen_candidate"]["candidate_id"]
                            != lookahead["policy_candidate"]["candidate_id"]
                        )
                        result["turn_boundary_decisions"] += 1
                        result["turn_boundary_action_changes"] += int(changed)
                        result["turn_boundary_expanded_paths"] += int(
                            lookahead["expanded_paths"]
                        )
                        active_combat["turn_boundary_decisions"] += 1
                        active_combat["turn_boundary_action_changes"] += int(changed)
                        active_combat["turn_boundary_expanded_paths"] += int(
                            lookahead["expanded_paths"]
                        )
                        source = "combat_turn_boundary"
                    elif args.one_step:
                        assert combat_entrance_save is not None
                        assert combat_enter_command is not None
                        assert lookahead_objective is not None
                        disabled_error = active_combat.get("search_disabled_error")
                        if disabled_error:
                            lookahead = _search_failure_policy_lookahead(
                                mode="one_step",
                                error=EngineError(str(disabled_error)),
                                state=state,
                                run_id=run_id,
                                decision_index=decision_index,
                                combat_index=len(result["combats"]),
                                model=model,
                                tensorizer=tensorizer,
                                device=device,
                                objective=objective,
                            )
                        else:
                            try:
                                lookahead = one_step_current_root(
                                worker=worker,
                                entrance_save=combat_entrance_save,
                                enter_command=combat_enter_command,
                                root_prefix=combat_prefix,
                                root_state=state,
                                model=model,
                                tensorizer=tensorizer,
                                device=device,
                                objective=lookahead_objective,
                                top_k=args.top_k,
                                policy_log_weight=args.policy_log_weight,
                                minimum_value_advantage=args.minimum_value_advantage,
                                minimum_end_turn_advantage=args.minimum_end_turn_advantage,
                                unsupported_penalty=args.unsupported_penalty,
                                determinization_count=args.determinizations,
                                cvar_alpha=args.cvar_alpha,
                                cvar_weight=args.cvar_weight,
                                search_seed=args.search_seed + decision_index * 100003,
                                step=decision_index,
                                minimum_potion_policy_probability=(
                                    args.minimum_potion_policy_probability
                                ),
                                )
                            except EngineError as exc:
                                active_combat["search_disabled_error"] = str(exc)
                                result["search_restore_failures"] += 1
                                active_combat["search_restore_failures"] += 1
                                lookahead = _search_failure_policy_lookahead(
                                    mode="one_step",
                                    error=exc,
                                    state=state,
                                    run_id=run_id,
                                    decision_index=decision_index,
                                    combat_index=len(result["combats"]),
                                    model=model,
                                    tensorizer=tensorizer,
                                    device=device,
                                    objective=objective,
                                )
                        if lookahead.get("status") == "engine_restore_fallback":
                            result["search_fallbacks"] += 1
                            active_combat["search_fallbacks"] += 1
                        ranked = [
                            {
                                "candidate": value["candidate"],
                                "probability": value["policy_probability"],
                                "policy_probability": value["policy_probability"],
                                "resource_prediction": None,
                            }
                            for value in lookahead["evaluations"]
                        ]
                        selected = next(
                            value for value in ranked
                            if value["candidate"]["candidate_id"]
                            == lookahead["chosen_candidate"]["candidate_id"]
                        )
                        inference_ms = float(lookahead["root_inference_ms"]) + float(
                            lookahead["successor_value_inference_ms"]
                        )
                        lookahead_ms_values.append(float(lookahead["wall_ms"]))
                        changed = (
                            lookahead["chosen_candidate"]["candidate_id"]
                            != lookahead["policy_candidate"]["candidate_id"]
                        )
                        result["one_step_decisions"] += 1
                        result["one_step_action_changes"] += int(changed)
                        active_combat["one_step_decisions"] += 1
                        active_combat["one_step_action_changes"] += int(changed)
                        source = "combat_one_step"
                    else:
                        sample = headless_state_to_model_sample(
                            state,
                            transition_id=f"fixed:{run_id}:{decision_index}",
                            combat_id=f"fixed:{run_id}:{len(result['combats'])}",
                        )
                        ranked, inference_ms = _rank_actions(
                            model, tensorizer, sample, device=device, objective=objective
                        )
                        selected = ranked[0]
                        source = "combat_policy"
                    if lookahead is not None and adaptive_trigger is not None:
                        lookahead["adaptive_trigger"] = adaptive_trigger
                    command = candidate_to_headless_command(selected["candidate"])
                    inference_ms_values.append(inference_ms)
                    plan_entry = None
                else:
                    if active_combat is not None and decision in {"card_select", "bundle_select"}:
                        selection_candidate = None
                        if decision == "card_select":
                            selection_candidate = first_card_select_candidate(state)
                            support = candidate_to_headless_command(selection_candidate)
                        else:
                            support = _support_command(state)
                        if support is None:
                            raise HumanRecordingError(
                                f"unsupported combat subdecision {decision!r}"
                            )
                        before = _state_summary(state)
                        state, engine_ms = engine.send(support)
                        engine_ms_values.append(engine_ms)
                        if search_enabled:
                            combat_prefix.append(support)
                        result["support_actions"] += 1
                        result["trace"].append({
                            "decision_index": decision_index,
                            "act": act,
                            "floor": floor,
                            "decision": decision,
                            "source": "deterministic_combat_subdecision",
                            "before": before,
                            "selected": selection_candidate,
                            "command": support,
                            "engine_ms": round(engine_ms, 3),
                            "after_decision": state.get("decision"),
                        })
                        continue
                    if active_combat is not None:
                        result["combats"].append(_finish_combat(active_combat, state))
                        active_combat = None
                        combat_entrance_save = None
                        combat_enter_command = None
                        combat_prefix = []
                    if cursor >= len(plan):
                        raise HumanRecordingError(f"fixed non-combat plan exhausted at {decision!r}")
                    plan_entry = plan[cursor]
                    expected_act = int(plan_entry["source_act"])
                    expected_floor = int(plan_entry["source_floor"])
                    if (act, floor) != (expected_act, expected_floor) or decision != plan_entry["phase"]:
                        if decision == "shop" and plan_entry["phase"] == "card_select":
                            cursor += 1
                            result["plan_actions_consumed"] = cursor
                            result["fallback_actions"] += 1
                            result["trace"].append({
                                "decision_index": decision_index,
                                "act": act,
                                "floor": floor,
                                "decision": decision,
                                "source": "engine_omitted_planned_shop_card_select",
                                "selected": {
                                    "source_action": plan_entry["action"],
                                    "source_record_sequence": plan_entry["record_sequence"],
                                },
                                "command": None,
                                "after_decision": decision,
                            })
                            continue
                        if (
                            (act, floor) == (expected_act, expected_floor)
                            and decision == "map_select"
                            and plan_entry["phase"] != "map_select"
                        ):
                            cursor += 1
                            result["plan_actions_consumed"] = cursor
                            result["fallback_actions"] += 1
                            result["trace"].append({
                                "decision_index": decision_index,
                                "act": act,
                                "floor": floor,
                                "decision": decision,
                                "source": f"engine_omitted_planned_{plan_entry['phase']}",
                                "selected": {
                                    "source_action": plan_entry["action"],
                                    "source_record_sequence": plan_entry["record_sequence"],
                                },
                                "command": None,
                                "after_decision": decision,
                            })
                            continue
                        support = _support_command(
                            state,
                            prefer_event_exit=plan_entry["phase"] == "map_select",
                        )
                        if support is not None:
                            before = _state_summary(state)
                            state, engine_ms = engine.send(support)
                            engine_ms_values.append(engine_ms)
                            result["support_actions"] += 1
                            result["trace"].append({
                                "decision_index": decision_index,
                                "act": act,
                                "floor": floor,
                                "decision": decision,
                                "source": "deterministic_prompt_support",
                                "before": before,
                                "command": support,
                                "engine_ms": round(engine_ms, 3),
                                "after_decision": state.get("decision"),
                            })
                            continue
                        raise HumanRecordingError(
                            f"fixed plan expects {plan_entry['phase']!r} at act {expected_act} floor {expected_floor}; "
                            f"engine has {decision!r} at act {act} floor {floor}"
                        )
                    try:
                        command = fixed_plan_command(state, plan_entry)
                        source = "fixed_human_plan"
                    except HumanRecordingError as exc:
                        command = _fallback_command(state, plan_entry)
                        if command is None:
                            raise
                        result["fallback_actions"] += 1
                        source = "deterministic_plan_fallback"
                        fallback_reason = str(exc)

                    if not command:
                        cursor += 1
                        result["plan_actions_consumed"] = cursor
                        result["trace"].append({
                            "decision_index": decision_index,
                            "act": act,
                            "floor": floor,
                            "decision": decision,
                            "source": source,
                            "selected": {
                                "source_action": plan_entry["action"],
                                "source_record_sequence": plan_entry["record_sequence"],
                                "fallback_reason": fallback_reason,
                            },
                            "command": None,
                            "after_decision": decision,
                        })
                        continue

                before = _state_summary(state)
                map_before = None
                if decision == "map_select":
                    context = state.get("context") or {}
                    map_before = {
                        "boss": context.get("boss"),
                        "choices": state.get("choices") or [],
                    }
                is_combat_source = source in {
                    "combat_policy", "combat_one_step", "combat_turn_boundary"
                }
                combat_before = _combat_visible_snapshot(state) if is_combat_source else None
                rejected_combat_commands: list[dict[str, Any]] = []
                entrance_probe: dict[str, Any] | None = None
                if search_enabled and not is_combat_source and decision in {"map_select", "event_choice"}:
                    entrance_path = temp_path / f"entrance-{decision_index}.save"
                    saved, _ = engine.send({
                        "cmd": "write_continue_save", "path": str(entrance_path)
                    })
                    if not saved.get("success"):
                        raise HumanRecordingError(
                            f"failed to capture possible combat entrance: {saved!r}"
                        )
                    entrance_probe = {"path": entrance_path, "command": command}
                try:
                    state, engine_ms = engine.send(command)
                except EngineError as exc:
                    if is_combat_source and args.turn_boundary:
                        assert combat_entrance_save is not None
                        assert combat_enter_command is not None
                        assert lookahead_objective is not None
                        assert active_combat is not None
                        excluded_candidate_ids: set[str] = set()
                        retry_error: EngineError = exc
                        retry_engine_ms = 0.0
                        while len(excluded_candidate_ids) < 8:
                            failed_id = str(selected["candidate"]["candidate_id"])
                            excluded_candidate_ids.add(failed_id)
                            rejected_combat_commands.append({
                                "candidate": _brief_candidate(selected),
                                "command": command,
                                "error": str(retry_error),
                            })
                            result["combat_engine_rejections"] += 1
                            active_combat["combat_engine_rejections"] += 1
                            refreshed, refresh_ms = engine.send({"cmd": "get_state"})
                            retry_engine_ms += refresh_ms
                            if refreshed.get("decision") != "combat_play":
                                raise retry_error
                            state = refreshed
                            retry_lookahead = turn_boundary_current_root(
                                workers=workers,
                                entrance_save=combat_entrance_save,
                                enter_command=combat_enter_command,
                                root_prefix=combat_prefix,
                                root_state=state,
                                model=model,
                                tensorizer=tensorizer,
                                device=device,
                                objective=lookahead_objective,
                                root_top_k=args.root_top_k,
                                beam_width=args.beam_width,
                                max_player_actions=args.max_player_actions,
                                policy_log_weight=args.policy_log_weight,
                                continuation_policy_weight=args.continuation_policy_weight,
                                minimum_value_advantage=args.minimum_value_advantage,
                                minimum_end_turn_advantage=args.minimum_end_turn_advantage,
                                unsupported_penalty=args.unsupported_penalty,
                                determinization_count=args.determinizations,
                                cvar_alpha=args.cvar_alpha,
                                cvar_weight=args.cvar_weight,
                                search_seed=args.search_seed + decision_index * 100003,
                                step=decision_index,
                                excluded_candidate_ids=excluded_candidate_ids,
                            )
                            retry_ranked = [
                                {
                                    "candidate": value["candidate"],
                                    "probability": value["policy_probability"],
                                    "policy_probability": value["policy_probability"],
                                    "resource_prediction": None,
                                }
                                for value in retry_lookahead["evaluations"]
                            ]
                            retry_selected = next(
                                value for value in retry_ranked
                                if value["candidate"]["candidate_id"]
                                == retry_lookahead["chosen_candidate"]["candidate_id"]
                            )
                            retry_command = candidate_to_headless_command(
                                retry_selected["candidate"]
                            )
                            retry_inference_ms = float(
                                retry_lookahead["root_inference_ms"]
                            ) + float(retry_lookahead["value_inference_ms"])
                            inference_ms_values.append(retry_inference_ms)
                            lookahead_ms_values.append(float(retry_lookahead["wall_ms"]))
                            retry_expanded = int(retry_lookahead["expanded_paths"])
                            result["turn_boundary_expanded_paths"] += retry_expanded
                            active_combat["turn_boundary_expanded_paths"] += retry_expanded
                            try:
                                state, retry_action_ms = engine.send(retry_command)
                            except EngineError as next_exc:
                                selected = retry_selected
                                ranked = retry_ranked
                                command = retry_command
                                lookahead = retry_lookahead
                                retry_error = next_exc
                                continue
                            selected = retry_selected
                            ranked = retry_ranked
                            command = retry_command
                            lookahead = retry_lookahead
                            engine_ms = retry_engine_ms + retry_action_ms
                            break
                        else:
                            raise retry_error
                    elif is_combat_source:
                        assert active_combat is not None
                        rejected_ids = {str(selected["candidate"]["candidate_id"])}
                        rejected_combat_commands.append({
                            "candidate": _brief_candidate(selected),
                            "command": command,
                            "error": str(exc),
                        })
                        result["combat_engine_rejections"] += 1
                        active_combat["combat_engine_rejections"] += 1
                        retry_error: EngineError = exc
                        for alternative in ranked:
                            alternative_id = str(
                                alternative["candidate"]["candidate_id"]
                            )
                            if alternative_id in rejected_ids:
                                continue
                            retry_command = candidate_to_headless_command(
                                alternative["candidate"]
                            )
                            try:
                                state, engine_ms = engine.send(retry_command)
                            except EngineError as next_exc:
                                rejected_ids.add(alternative_id)
                                rejected_combat_commands.append({
                                    "candidate": _brief_candidate(alternative),
                                    "command": retry_command,
                                    "error": str(next_exc),
                                })
                                result["combat_engine_rejections"] += 1
                                active_combat["combat_engine_rejections"] += 1
                                retry_error = next_exc
                                continue
                            selected = alternative
                            command = retry_command
                            break
                        else:
                            raise retry_error
                    else:
                        action_id = str((plan_entry or {}).get("action", {}).get("action_id") or "")
                        if (
                            not is_combat_source
                            and decision == "shop"
                            and action_id in {"buy_shop_item", "remove_card"}
                        ):
                            cursor += 1
                            result["plan_actions_consumed"] = cursor
                            result["fallback_actions"] += 1
                            result["trace"].append({
                                "decision_index": decision_index,
                                "act": act,
                                "floor": floor,
                                "decision": decision,
                                "source": "engine_rejected_shop_action",
                                "before": before,
                                "selected": {
                                    "source_action": plan_entry["action"],
                                    "source_record_sequence": plan_entry["record_sequence"],
                                    "fallback_reason": str(exc),
                                },
                                "command": command,
                                "after_decision": decision,
                            })
                            continue
                        raise
                engine_ms_values.append(engine_ms)
                if is_combat_source:
                    assert active_combat is not None
                    if search_enabled:
                        combat_prefix.append(command)
                    active_combat["actions"] = int(active_combat["actions"]) + 1
                    result["combat_actions"] += 1
                    selected_summary = _brief_candidate(selected)
                    top3 = [_brief_candidate(value) for value in ranked[:3]]
                else:
                    if entrance_probe is not None and state.get("decision") == "combat_play":
                        pending_combat_entrance = entrance_probe
                    cursor += 1
                    result["plan_actions_consumed"] = cursor
                    result["fixed_actions"] += 1
                    selected_summary = {
                        "source_action": plan_entry["action"],
                        "source_record_sequence": plan_entry["record_sequence"],
                    }
                    if source == "deterministic_plan_fallback":
                        selected_summary["fallback_reason"] = fallback_reason
                    top3 = None
                trace_entry = {
                    "decision_index": decision_index,
                    "act": act,
                    "floor": floor,
                    "decision": decision,
                    "source": source,
                    "before": before,
                    "selected": selected_summary,
                    "top3": top3,
                    "command": command,
                    "engine_ms": round(engine_ms, 3),
                    "after_decision": state.get("decision"),
                }
                if rejected_combat_commands:
                    trace_entry["rejected_combat_commands"] = rejected_combat_commands
                if map_before is not None:
                    trace_entry["map_before"] = map_before
                if is_combat_source:
                    trace_entry["combat_before"] = combat_before
                    if source in {"combat_one_step", "combat_turn_boundary"}:
                        trace_entry["lookahead"] = lookahead
                    trace_entry["combat_after"] = (
                        _combat_visible_snapshot(state)
                        if state.get("decision") == "combat_play"
                        else _state_summary(state)
                    )
                result["trace"].append(trace_entry)
            else:
                result["status"] = "decision_limit"
                result["blocked_state"] = _decision_details(state)
    except Exception as exc:
        result["status"] = "blocked"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["blocked_state"] = _decision_details(state)
        if plan_entry is not None:
            result["blocked_plan_entry"] = plan_entry
        if active_combat is not None:
            result["active_combat"] = active_combat
    result["mean_inference_ms"] = (
        round(statistics.fmean(inference_ms_values), 3) if inference_ms_values else None
    )
    result["mean_engine_ms"] = (
        round(statistics.fmean(engine_ms_values), 3) if engine_ms_values else None
    )
    result["total_lookahead_ms"] = round(
        prior_lookahead_ms + sum(lookahead_ms_values), 3
    )
    result["mean_lookahead_ms"] = (
        round(statistics.fmean(lookahead_ms_values), 3) if lookahead_ms_values else None
    )
    result["total_turn_boundary_search_ms"] = round(sum(
        float((entry.get("lookahead") or {}).get("wall_ms") or 0.0)
        for entry in result["trace"]
        if entry.get("source") == "combat_turn_boundary"
    ), 3)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.one_step and args.turn_boundary:
        raise HumanRecordingError("--one-step and --turn-boundary are mutually exclusive")
    if args.turn_boundary and args.adaptive_turn_boundary:
        raise HumanRecordingError(
            "--turn-boundary and --adaptive-turn-boundary are mutually exclusive"
        )
    if args.search_workers < 1:
        raise HumanRecordingError("--search-workers must be at least 1")
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if (args.turn_boundary or args.adaptive_turn_boundary) and device != "cuda":
        raise HumanRecordingError(
            f"turn-boundary continuous evaluation requires CUDA; resolved {device!r}"
        )
    game_data_dir = _game_data_dir(args.game_dir)
    episodes = {
        value["run_id"]: value
        for value in pq.read_table(args.episodes_path).to_pylist()
    }
    transitions = pq.read_table(args.transitions_path).to_pylist()
    combats = pq.read_table(args.combats_path).to_pylist()
    requested = args.run_id or list(DEFAULT_TEMPLATES)
    missing = [run_id for run_id in requested if run_id not in episodes]
    if missing:
        raise HumanRecordingError(f"unknown template run IDs: {missing!r}")
    resume_by_run: dict[str, dict[str, Any]] = {}
    if args.resume_report is not None:
        resume_payload = json.loads(args.resume_report.read_text(encoding="utf-8"))
        resume_by_run = {
            str(value["template_run_id"]): value
            for value in resume_payload.get("runs") or []
        }
        missing_resume = [run_id for run_id in requested if run_id not in resume_by_run]
        if missing_resume:
            raise HumanRecordingError(
                f"resume report has no snapshots for run IDs: {missing_resume!r}"
            )
    report = {
        "schema_version": "combat-policy-fixed-plan-eval-0.6.0",
        "generated_at": utc_now(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "source_data": {
            "episodes_path": str(args.episodes_path.resolve()),
            "transitions_path": str(args.transitions_path.resolve()),
            "combats_path": str(args.combats_path.resolve()),
        },
        "execution_ascension": args.ascension,
        "one_step": {
            "enabled": bool(args.one_step),
            "top_k": args.top_k,
            "policy_log_weight": args.policy_log_weight,
            "minimum_value_advantage": args.minimum_value_advantage,
            "minimum_end_turn_advantage": args.minimum_end_turn_advantage,
            "minimum_potion_policy_probability": (
                args.minimum_potion_policy_probability
            ),
            "determinizations": args.determinizations,
            "cvar_alpha": args.cvar_alpha,
            "cvar_weight": args.cvar_weight,
        },
        "turn_boundary": {
            "enabled": bool(args.turn_boundary or args.adaptive_turn_boundary),
            "adaptive": bool(args.adaptive_turn_boundary),
            "adaptive_minimum_hp_loss": args.adaptive_minimum_hp_loss,
            "adaptive_minimum_hp_fraction": args.adaptive_minimum_hp_fraction,
            "search_workers": args.search_workers,
            "root_top_k": args.root_top_k,
            "beam_width": args.beam_width,
            "max_player_actions": args.max_player_actions,
            "continuation_policy_weight": args.continuation_policy_weight,
        },
        "selection_note": (
            "The default is a complete run-held-out Ironclad A0 win. The original seed and "
            "recorded non-combat decisions are reused, while combat actions come from the policy. "
            "Combat outcomes alter later RNG consumption, so this is a fixed-plan continuous "
            "evaluation rather than a byte-identical replay of the human run."
        ),
        "runs": [],
    }
    for run_id in requested:
        episode = episodes[run_id]
        run_transitions = [value for value in transitions if value["run_id"] == run_id]
        run_combats = [value for value in combats if value["run_id"] == run_id]
        report["runs"].append(run_episode(
            args,
            episode=episode,
            source_transitions=run_transitions,
            source_combats=run_combats,
            model=model,
            tensorizer=tensorizer,
            device=device,
            game_data_dir=game_data_dir,
            resume_snapshot=resume_by_run.get(run_id),
        ))
    report["summary"] = {
        "runs": len(report["runs"]),
        "victories": sum(value["status"] == "victory" for value in report["runs"]),
        "deaths": sum(value["status"] == "death" for value in report["runs"]),
        "blocked": sum(value["status"] == "blocked" for value in report["runs"]),
        "max_floor": max((int(value.get("max_floor", 0)) for value in report["runs"]), default=0),
        "one_step_decisions": sum(int(value["one_step_decisions"]) for value in report["runs"]),
        "one_step_action_changes": sum(
            int(value["one_step_action_changes"]) for value in report["runs"]
        ),
        "turn_boundary_decisions": sum(
            int(value["turn_boundary_decisions"]) for value in report["runs"]
        ),
        "turn_boundary_action_changes": sum(
            int(value["turn_boundary_action_changes"]) for value in report["runs"]
        ),
        "turn_boundary_expanded_paths": sum(
            int(value["turn_boundary_expanded_paths"]) for value in report["runs"]
        ),
        "adaptive_turn_boundary_triggers": sum(
            int(value["adaptive_turn_boundary_triggers"])
            for value in report["runs"]
        ),
        "search_fallbacks": sum(
            int(value["search_fallbacks"]) for value in report["runs"]
        ),
        "search_restore_failures": sum(
            int(value["search_restore_failures"]) for value in report["runs"]
        ),
        "combat_engine_rejections": sum(
            int(value["combat_engine_rejections"]) for value in report["runs"]
        ),
        "total_lookahead_ms": round(sum(
            float(value["total_lookahead_ms"]) for value in report["runs"]
        ), 3),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", help="source HumanRecorder run ID; repeatable")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes-path", type=Path, default=EPISODES_PATH)
    parser.add_argument("--transitions-path", type=Path, default=TRANSITIONS_PATH)
    parser.add_argument("--combats-path", type=Path, default=COMBATS_PATH)
    parser.add_argument(
        "--resume-report",
        type=Path,
        help="replay a prior report's completed commands and continue at its blocked decision",
    )
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--max-combat-actions", type=int, default=500)
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--turn-boundary", action="store_true")
    parser.add_argument(
        "--adaptive-turn-boundary",
        action="store_true",
        help="upgrade high visible-HP-loss decisions to turn-boundary search",
    )
    parser.add_argument("--adaptive-minimum-hp-loss", type=float, default=8.0)
    parser.add_argument("--adaptive-minimum-hp-fraction", type=float, default=0.4)
    parser.add_argument("--search-seed", type=int, default=20260816)
    parser.add_argument("--search-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--root-top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-player-actions", type=int, default=3)
    parser.add_argument("--policy-log-weight", type=float, default=0.05)
    parser.add_argument("--continuation-policy-weight", type=float, default=0.01)
    parser.add_argument("--minimum-value-advantage", type=float, default=0.02)
    parser.add_argument("--minimum-end-turn-advantage", type=float, default=0.15)
    parser.add_argument(
        "--minimum-potion-policy-probability",
        type=float,
        default=0.0,
        help=(
            "minimum P1 support required before one-step search may introduce "
            "potion use; upper-level directives may lower this threshold"
        ),
    )
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _add_objective_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(json.dumps({
        "summary": report["summary"],
        "runs": [
            {
                "seed": value["seed"],
                "source_ascension": value["source_ascension"],
                "status": value["status"],
                "max_act": value.get("max_act"),
                "max_floor": value.get("max_floor"),
                "combats": len(value["combats"]),
                "combat_actions": value["combat_actions"],
                "turn_boundary_decisions": value["turn_boundary_decisions"],
                "turn_boundary_action_changes": value["turn_boundary_action_changes"],
                "turn_boundary_expanded_paths": value["turn_boundary_expanded_paths"],
                "adaptive_turn_boundary_triggers": value[
                    "adaptive_turn_boundary_triggers"
                ],
                "plan_actions_consumed": value["plan_actions_consumed"],
                "error": value.get("error"),
            }
            for value in report["runs"]
        ],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
