"""Collect P2 on-policy roots from train fights and label Top-k actions to terminal."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_policy_guided_mcts import (  # noqa: E402
    POST_COMBAT_DECISIONS,
    SEARCH_DECISIONS,
    _cache_key,
    _determinization,
    _engine,
    _restore_search_root,
    _utility,
    _visible_draw_multiset,
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
    _enter_command,
    _prepare_scenario_save,
    _resolve_optional_precombat_selects,
)
from run_combat_mcts_comparison import _result, sha256_file_bytes  # noqa: E402
from run_combat_policy_online import _load_policy, _rank_actions, _state_summary  # noqa: E402
from run_validation_combat_ablation import (  # noqa: E402
    DEFAULT_COMBATS,
    DEFAULT_P2,
    DEFAULT_TARGETS,
    DEFAULT_TRANSITIONS,
    _find_matching_base_save,
    _latest_checkpoint,
    _load_scenarios,
)
from sts2_dataset.combat_counterfactual import (  # noqa: E402
    COUNTERFACTUAL_TEACHER_VERSION,
    build_pairwise_labels,
    on_policy_trigger_reasons,
    select_counterfactual_roots,
    summarize_counterfactual_action,
    summarize_counterfactual_root,
)
from sts2_dataset.combat_failure import _load_training_reference  # noqa: E402
from sts2_dataset.combat_lookahead import (  # noqa: E402
    policy_top_k,
    required_search_categories,
)
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    first_card_select_candidate,
    headless_state_to_model_sample,
    visible_intent_end_turn_hp_loss,
)
from sts2_dataset.combat_search import normalized_policy_entropy  # noqa: E402
from sts2_dataset.util import (  # noqa: E402
    canonical_json,
    load_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


DEFAULT_FAILURE_RATCHET = REPO_ROOT / "artifacts" / "combat_failure_ratchet_v0.json"
DEFAULT_SAMPLES = REPO_ROOT / "data" / "human" / "combat_v1" / "model_v0" / "samples.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_on_policy_counterfactual_v0.json"


def _priority_encounters(path: Path) -> Counter[str]:
    report = load_json(path.resolve())
    result: Counter[str] = Counter()
    for row in report.get("failure_combats") or []:
        weight = 1 + len(row.get("flags") or [])
        result[str(row["encounter"])] += weight
    return result


def _select_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    priorities: Counter[str],
    limit: int,
    requested_encounters: list[str] | None,
) -> list[dict[str, Any]]:
    requested = set(requested_encounters or [])
    candidates = [
        row
        for row in scenarios
        if (not requested or row["encounter"] in requested)
        and (requested or row["encounter"] in priorities)
    ]
    if not candidates:
        raise EngineError("no train scenarios matched the requested failure encounters")
    by_act: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_act[int(row["act"])].append(row)
    for rows in by_act.values():
        rows.sort(
            key=lambda row: (
                -priorities[str(row["encounter"])],
                str(row["encounter"]),
                str(row["run_id"]),
                int(row["floor"]),
            )
        )
    selected: list[dict[str, Any]] = []
    seen_encounters: set[str] = set()
    while len(selected) < limit:
        added = False
        for act in sorted(by_act):
            rows = by_act[act]
            index = next(
                (
                    idx
                    for idx, row in enumerate(rows)
                    if str(row["encounter"]) not in seen_encounters
                ),
                0 if rows else None,
            )
            if index is None:
                continue
            row = rows.pop(index)
            selected.append(row)
            seen_encounters.add(str(row["encounter"]))
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def _ranked_summary(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": row["candidate"],
            "policy_probability": float(row["policy_probability"]),
        }
        for row in ranked
    ]


def _action_train_count(training: dict[str, Any], candidate: dict[str, Any]) -> int:
    key = (str(candidate.get("action_type") or ""), candidate.get("source_id"))
    return int(training["action_label_count"][key])


def _counterfactual_world(
    *,
    worker: Any,
    entrance_save: Path,
    scenario: dict[str, Any],
    root: dict[str, Any],
    candidate: dict[str, Any],
    draw_order: list[str],
    determinization_id: str,
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    max_actions: int,
    restore_mode: str,
) -> dict[str, Any]:
    root_state = root["engine_state"]
    state, restore_ms = _restore_search_root(
        worker=worker,
        entrance_save=entrance_save,
        enter_command=_enter_command(scenario),
        root_prefix=root["prefix"],
        draw_order=draw_order,
        mode=restore_mode,
    )
    if _state_summary(state) != _state_summary(root_state):
        raise EngineError("counterfactual restore changed the visible root")
    root_player = root_state.get("player") or {}
    root_hp = float(root_player.get("hp") or 0.0)
    root_max_hp = float(root_player.get("max_hp") or 1.0)
    root_potions = len(root_player.get("potions") or [])
    trace: list[dict[str, Any]] = []
    command = candidate_to_headless_command(candidate)
    state, engine_ms = worker.send(command)
    trace.append(
        {
            "source": "counterfactual_root_action",
            "candidate": candidate,
            "command": command,
            "after": _state_summary(state),
        }
    )
    total_engine_ms = float(engine_ms)
    for step in range(max_actions):
        decision = str(state.get("decision") or "")
        if decision not in SEARCH_DECISIONS:
            break
        if decision == "combat_play":
            sample = headless_state_to_model_sample(
                state,
                transition_id=f"counterfactual:{root['root_fingerprint']}:{determinization_id}:{step}",
                combat_id=str(scenario["source_combat_id"]),
                encounter_signature=str(scenario["encounter_signature"]),
            )
            ranked, inference_ms = _rank_actions(
                model, tensorizer, sample, device=device, objective=None
            )
            chosen = max(ranked, key=lambda row: float(row["policy_probability"]))
            continuation_candidate = chosen["candidate"]
            probability = float(chosen["policy_probability"])
            source = "fixed_p2_continuation"
        else:
            continuation_candidate = first_card_select_candidate(state)
            inference_ms = 0.0
            probability = 1.0
            source = "deterministic_first_card_select"
        command = candidate_to_headless_command(continuation_candidate)
        before = _state_summary(state)
        state, current_ms = worker.send(command)
        total_engine_ms += float(current_ms)
        trace.append(
            {
                "source": source,
                "before": before,
                "candidate": continuation_candidate,
                "policy_probability": probability,
                "inference_ms": round(float(inference_ms), 3),
                "command": command,
                "after": _state_summary(state),
            }
        )
    decision = str(state.get("decision") or "")
    terminal = decision == "game_over" or decision in POST_COMBAT_DECISIONS
    player = state.get("player") or {}
    terminal_hp = float(player.get("hp") or 0.0)
    terminal_max_hp = float(player.get("max_hp") or root_max_hp)
    potion_spent = float(max(0, root_potions - len(player.get("potions") or [])))
    hp_loss = max(0.0, root_hp - terminal_hp)
    death = bool(decision == "game_over" and not state.get("victory"))
    max_hp_delta = terminal_max_hp - root_max_hp
    utility = _utility(
        hp_loss_fraction=hp_loss / max(root_max_hp, 1.0),
        death_probability=float(death),
        potion_spent=potion_spent,
        max_hp_delta=max_hp_delta,
        objective=objective,
    )
    return {
        "determinization_id": determinization_id,
        "terminal": terminal,
        "terminal_decision": decision,
        "death": death,
        "terminal_hp": terminal_hp,
        "terminal_max_hp": terminal_max_hp,
        "hp_loss": hp_loss,
        "potion_spent": potion_spent,
        "max_hp_delta": max_hp_delta,
        "utility": utility,
        "restore_ms": round(float(restore_ms), 3),
        "engine_ms": round(total_engine_ms, 3),
        "continuation_actions": max(0, len(trace) - 1),
        "continuation_policy": "frozen_p2_argmax_with_deterministic_first_card_select",
        "trace": trace,
    }


def _label_root(
    *,
    args: argparse.Namespace,
    worker: Any,
    entrance_save: Path,
    scenario: dict[str, Any],
    root: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
) -> dict[str, Any]:
    ranked = root["ranked"]
    shortlist = policy_top_k(
        ranked,
        args.top_k,
        required_categories=required_search_categories(root["sample"]["observation"]),
    )
    visible_draw = _visible_draw_multiset(root["engine_state"])
    worlds = [
        _determinization(
            visible_draw,
            search_seed=args.search_seed + int(root["step"]) * 100003,
            simulation=index,
        )
        for index in range(args.determinizations)
    ]
    actions: list[dict[str, Any]] = []
    for policy_row in shortlist:
        candidate = policy_row["candidate"]
        outcomes: list[dict[str, Any]] = []
        for draw_order, determinization_id in worlds:
            try:
                outcome = _counterfactual_world(
                    worker=worker,
                    entrance_save=entrance_save,
                    scenario=scenario,
                    root=root,
                    candidate=candidate,
                    draw_order=draw_order,
                    determinization_id=determinization_id,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    objective=objective,
                    max_actions=args.max_continuation_actions,
                    restore_mode=args.restore_mode,
                )
            except EngineError as exc:
                outcome = {
                    "determinization_id": determinization_id,
                    "terminal": False,
                    "terminal_decision": "engine_error",
                    "death": False,
                    "terminal_hp": 0.0,
                    "hp_loss": 0.0,
                    "potion_spent": 0.0,
                    "utility": -float(args.unsupported_penalty),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            outcomes.append(outcome)
        action = summarize_counterfactual_action(
            candidate, outcomes, cvar_alpha=args.cvar_alpha
        )
        action["policy_probability"] = float(policy_row["policy_probability"])
        actions.append(action)
    root_summary = summarize_counterfactual_root(actions)
    return {
        "schema_version": COUNTERFACTUAL_TEACHER_VERSION,
        "root_fingerprint": root["root_fingerprint"],
        "scenario_id": scenario["scenario_id"],
        "step": root["step"],
        "trigger_reasons": root["trigger_reasons"],
        "priority_score": root["priority_score"],
        "root_sample": root["sample"],
        "information_boundary": "player_visible_root_plus_common_visible_draw_determinizations",
        "continuation_policy": "frozen_p2_argmax",
        "determinization_count": args.determinizations,
        "actions": actions,
        "pairwise_labels": build_pairwise_labels(actions),
        **root_summary,
    }


def _run_combat(
    *,
    args: argparse.Namespace,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    training: dict[str, Any],
) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    compact_steps: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    with _engine(args, game_data_dir) as engine, _engine(args, game_data_dir) as worker:
        state, _ = engine.send(
            {"cmd": "load_save", "path": str(entrance_save), "lang": "en"}
        )
        state, _ = engine.send(_enter_command(scenario))
        state, precombat_prefix = _resolve_optional_precombat_selects(engine, state)
        prefix.extend(precombat_prefix)
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        root_signature = sha256_file_bytes(
            json.dumps(state, sort_keys=True).encode("utf-8")
        )
        cached, _ = worker.send(
            {
                "cmd": "cache_save",
                "name": _cache_key(entrance_save),
                "path": str(entrance_save),
            }
        )
        if cached.get("type") != "ok":
            raise EngineError(f"counterfactual worker could not cache entrance: {cached!r}")
        for step in range(args.max_actions):
            decision = str(state.get("decision") or "")
            if decision not in SEARCH_DECISIONS:
                break
            before = _state_summary(state)
            if decision == "combat_play":
                sample = headless_state_to_model_sample(
                    state,
                    transition_id=f"on-policy:{scenario['scenario_id']}:{step}",
                    combat_id=str(scenario["source_combat_id"]),
                    encounter_signature=str(scenario["encounter_signature"]),
                )
                ranked, inference_ms = _rank_actions(
                    model, tensorizer, sample, device=device, objective=None
                )
                ranked.sort(key=lambda row: float(row["policy_probability"]), reverse=True)
                chosen = ranked[0]
                candidate = chosen["candidate"]
                probabilities = [float(row["policy_probability"]) for row in ranked]
                entropy = normalized_policy_entropy(probabilities)
                margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else 1.0
                player = state.get("player") or {}
                hp_ratio = float(player.get("hp") or 0.0) / max(
                    float(player.get("max_hp") or 1.0), 1.0
                )
                exact_round = (
                    training["encounter_profile_quantiles"]
                    .get(str(scenario["encounter_signature"]), {})
                    .get("round")
                )
                visible_loss = visible_intent_end_turn_hp_loss(sample["observation"])
                incoming_loss = float((visible_loss or {}).get("hp_loss") or 0.0)
                action_train_count = _action_train_count(training, candidate)
                reasons = on_policy_trigger_reasons(
                    hp_ratio=hp_ratio,
                    round_number=int((state.get("round") or 1)),
                    exact_encounter_round_p95=exact_round[3] if exact_round else None,
                    incoming_hp_loss=incoming_loss,
                    policy_entropy=entropy,
                    policy_margin=margin,
                    chosen_action_train_count=action_train_count,
                    low_hp_threshold=args.low_hp_threshold,
                    incoming_hp_loss_threshold=args.incoming_hp_loss_threshold,
                    high_entropy_threshold=args.high_entropy_threshold,
                    low_margin_threshold=args.low_margin_threshold,
                    rare_action_threshold=args.rare_action_threshold,
                )
                fingerprint_payload = {
                    "observation": sample["observation"],
                    "candidates": sample["candidates"],
                }
                root = {
                    "root_fingerprint": sha256_file_bytes(
                        canonical_json(fingerprint_payload).encode("utf-8")
                    ),
                    "step": step,
                    "round": int(state.get("round") or 1),
                    "hp_ratio": hp_ratio,
                    "policy_entropy": entropy,
                    "policy_margin": margin,
                    "visible_incoming_hp_loss": incoming_loss,
                    "chosen_action_train_count": action_train_count,
                    "trigger_reasons": reasons,
                    "sample": sample,
                    "ranked": ranked,
                    "engine_state": state,
                    "prefix": list(prefix),
                }
                roots.append(root)
                command = candidate_to_headless_command(candidate)
                compact_steps.append(
                    {
                        "step": step,
                        "before": before,
                        "chosen_candidate": candidate,
                        "policy_probability": float(chosen["policy_probability"]),
                        "policy_entropy": entropy,
                        "policy_margin": margin,
                        "visible_incoming_hp_loss": incoming_loss,
                        "chosen_action_train_count": action_train_count,
                        "trigger_reasons": reasons,
                        "inference_ms": round(float(inference_ms), 3),
                    }
                )
            else:
                candidate = first_card_select_candidate(state)
                command = candidate_to_headless_command(candidate)
                compact_steps.append(
                    {
                        "step": step,
                        "before": before,
                        "chosen_candidate": candidate,
                        "policy_probability": 1.0,
                        "policy_entropy": 0.0,
                        "policy_margin": 1.0,
                        "trigger_reasons": ["deterministic_card_select"],
                    }
                )
            state, engine_ms = engine.send(command)
            prefix.append(command)
            compact_steps[-1]["engine_ms"] = round(float(engine_ms), 3)
            compact_steps[-1]["after"] = _state_summary(state)
        result = _result(state, initial_hp=initial_hp, steps=compact_steps)
        result["root_signature"] = root_signature
        human_loss = float(scenario["human"]["hp_loss"])
        combat_failure = bool(
            result.get("status") == "death"
            or float(result["hp_loss"]) - human_loss >= args.high_regret_hp
            or float(result["hp_loss"]) >= args.high_absolute_hp_loss
        )
        selected = select_counterfactual_roots(
            roots,
            combat_failure=combat_failure,
            limit=args.roots_per_combat,
            strategy=args.root_selection,
        )
        teachers = [
            _label_root(
                args=args,
                worker=worker,
                entrance_save=entrance_save,
                scenario=scenario,
                root=root,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
            )
            for root in selected
        ]
    return {
        "policy_result": result,
        "human": scenario["human"],
        "combat_failure": combat_failure,
        "triggered_root_count": sum(bool(root["trigger_reasons"]) for root in roots),
        "selected_root_count": len(selected),
        "teacher_eligible_root_count": sum(
            bool(teacher["teacher_eligible"]) for teacher in teachers
        ),
        "informative_root_count": sum(
            bool(teacher["informative"]) for teacher in teachers
        ),
        "policy_suboptimal_root_count": sum(
            bool(teacher.get("policy_suboptimal")) for teacher in teachers
        ),
        "terminal_world_count": sum(
            int(action["terminal_world_count"])
            for teacher in teachers
            for action in teacher["actions"]
        ),
        "teacher_roots": teachers,
    }


def _report(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    scenarios: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    started: float,
    status: str,
    provenance: dict[str, Any],
    resumed_combats: int,
) -> dict[str, Any]:
    return {
        "schema_version": "combat-on-policy-counterfactual-0.2.0",
        "generated_at": utc_now(),
        "status": status,
        "device": "cuda",
        "dataset_split": "train",
        "checkpoint": provenance["checkpoint"],
        "sources": provenance["sources"],
        "collection_config": provenance["collection_config"],
        "collection_signature": provenance["collection_signature"],
        "selection": {
            "combat_limit": args.combat_limit,
            **provenance["collection_config"],
        },
        "planned_combats": len(scenarios),
        "completed_combats": len(rows),
        "resumed_combats": resumed_combats,
        "combat_failures": sum(bool(row["combat_failure"]) for row in rows),
        "selected_roots": sum(int(row["selected_root_count"]) for row in rows),
        "teacher_eligible_roots": sum(
            int(row["teacher_eligible_root_count"]) for row in rows
        ),
        "informative_roots": sum(int(row["informative_root_count"]) for row in rows),
        "policy_suboptimal_roots": sum(
            int(row["policy_suboptimal_root_count"]) for row in rows
        ),
        "terminal_worlds": sum(int(row["terminal_world_count"]) for row in rows),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "combats": rows,
    }


def _collection_provenance(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint_block = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    sources = {
        "transitions_sha256": sha256_file(args.transitions.resolve()),
        "combats_sha256": sha256_file(args.combats.resolve()),
        "targets_sha256": sha256_file(args.targets.resolve()),
        "samples_sha256": sha256_file(args.samples.resolve()),
        "failure_ratchet_sha256": sha256_file(args.failure_ratchet.resolve()),
    }
    collection_config = {
        "roots_per_combat": args.roots_per_combat,
        "top_k": args.top_k,
        "determinizations": args.determinizations,
        "search_seed": args.search_seed,
        "cvar_alpha": args.cvar_alpha,
        "high_regret_hp": args.high_regret_hp,
        "high_absolute_hp_loss": args.high_absolute_hp_loss,
        "low_hp_threshold": args.low_hp_threshold,
        "incoming_hp_loss_threshold": args.incoming_hp_loss_threshold,
        "high_entropy_threshold": args.high_entropy_threshold,
        "low_margin_threshold": args.low_margin_threshold,
        "rare_action_threshold": args.rare_action_threshold,
        "unsupported_penalty": args.unsupported_penalty,
        "restore_mode": args.restore_mode,
        "max_actions": args.max_actions,
        "max_continuation_actions": args.max_continuation_actions,
        "seed": args.seed,
        "encounter_seed_trials": args.encounter_seed_trials,
        "requested_encounters": sorted(args.encounters or []),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "continuation_policy": "frozen_p2_argmax",
        "root_selection": args.root_selection,
    }
    signature_payload = {
        "schema_version": "combat-on-policy-counterfactual-0.2.0",
        "checkpoint": checkpoint_block,
        "sources": sources,
        "collection_config": collection_config,
    }
    return {
        **signature_payload,
        "collection_signature": sha256_file_bytes(
            canonical_json(signature_payload).encode("utf-8")
        ),
    }


def _resume_rows(
    output: Path,
    *,
    provenance: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    previous = load_json(output)
    if previous.get("schema_version") != "combat-on-policy-counterfactual-0.2.0":
        raise EngineError("resume report schema is incompatible with strict resume")
    if previous.get("collection_signature") != provenance["collection_signature"]:
        raise EngineError("resume report collection signature does not match this run")
    by_id: dict[str, dict[str, Any]] = {}
    for row in previous.get("combats") or []:
        scenario_id = str(row["scenario_id"])
        if scenario_id in by_id:
            raise EngineError(f"resume report contains duplicate scenario: {scenario_id}")
        by_id[scenario_id] = row
    selected_ids = {str(row["scenario_id"]) for row in scenarios}
    unexpected = sorted(set(by_id) - selected_ids)
    if unexpected:
        raise EngineError(
            f"resume report contains scenarios outside the current selection: {unexpected[:3]}"
        )
    return [
        by_id[str(scenario["scenario_id"])]
        for scenario in scenarios
        if str(scenario["scenario_id"]) in by_id
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint or _latest_checkpoint(DEFAULT_P2))
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if device != "cuda":
        raise EngineError(f"CUDA is required; resolved device={device!r}")
    objective = CombatObjective.from_config(model.config)
    training = _load_training_reference(args.samples.resolve())
    priorities = _priority_encounters(args.failure_ratchet.resolve())
    all_scenarios = _load_scenarios(
        args, dataset_split="train", skip_unmapped_encounters=True
    )
    scenarios = _select_scenarios(
        all_scenarios,
        priorities=priorities,
        limit=args.combat_limit,
        requested_encounters=args.encounters,
    )
    if args.list_only:
        return {
            "schema_version": "combat-on-policy-counterfactual-list-0.1.0",
            "status": "pass",
            "scenarios": [
                {
                    "scenario_id": row["scenario_id"],
                    "act": row["act"],
                    "floor": row["floor"],
                    "ascension": row["ascension"],
                    "encounter": row["encounter"],
                    "human_hp_loss": row["human"]["hp_loss"],
                    "priority": priorities[row["encounter"]],
                }
                for row in scenarios
            ],
        }
    game_data_dir = _game_data_dir(args.game_dir)
    output = args.output.resolve()
    provenance = _collection_provenance(args, checkpoint=checkpoint)
    rows = (
        _resume_rows(output, provenance=provenance, scenarios=scenarios)
        if args.resume
        else []
    )
    resumed_combats = len(rows)
    completed_ids = {str(row["scenario_id"]) for row in rows}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sts2_on_policy_cf_") as temp_dir:
        temp = Path(temp_dir)
        base_saves: dict[tuple[int, str, tuple[str, ...]], tuple[dict[str, Any], str, int]] = {}
        for index, scenario in enumerate(scenarios):
            if str(scenario["scenario_id"]) in completed_ids:
                continue
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
                    path=temp / f"base-{len(base_saves):03d}.save",
                )
            base_save, reconstruction_seed, seed_trials = base_saves[base_key]
            entrance_save = temp / f"scenario-{index:03d}.save"
            prepared_root = _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base_save,
                scenario=scenario,
                path=entrance_save,
            )
            evaluation = _run_combat(
                args=args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                training=training,
            )
            if evaluation["policy_result"]["root_signature"] != prepared_root["root_signature"]:
                raise EngineError(f"prepared root mismatch in {scenario['scenario_id']}")
            row = {
                **{key: value for key, value in scenario.items() if key != "player"},
                "snapshot": scenario["player"],
                "reconstruction_seed": reconstruction_seed,
                "reconstruction_seed_trials": seed_trials,
                **evaluation,
            }
            rows.append(row)
            write_json_atomic(
                output,
                _report(
                    args=args,
                    checkpoint=checkpoint,
                    scenarios=scenarios,
                    rows=rows,
                    started=started,
                    status="running",
                    provenance=provenance,
                    resumed_combats=resumed_combats,
                ),
            )
            print(
                json.dumps(
                    {
                        "completed": scenario["scenario_id"],
                        "progress": f"{len(rows)}/{len(scenarios)}",
                        "encounter": scenario["encounter"],
                        "policy_status": evaluation["policy_result"]["status"],
                        "human_hp_loss": scenario["human"]["hp_loss"],
                        "policy_hp_loss": evaluation["policy_result"]["hp_loss"],
                        "selected_roots": evaluation["selected_root_count"],
                        "eligible_roots": evaluation["teacher_eligible_root_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    report = _report(
        args=args,
        checkpoint=checkpoint,
        scenarios=scenarios,
        rows=rows,
        started=started,
        status="pass",
        provenance=provenance,
        resumed_combats=resumed_combats,
    )
    write_json_atomic(output, report)
    return report


def _resolve_checkpoint(value: Path) -> Path:
    return value.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--combats", type=Path, default=DEFAULT_COMBATS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--failure-ratchet", type=Path, default=DEFAULT_FAILURE_RATCHET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--combat-limit", type=int, default=3)
    parser.add_argument("--encounters", nargs="+")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--roots-per-combat", type=int, default=2)
    parser.add_argument(
        "--root-selection", choices=("diverse", "earliest"), default="diverse"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=20260818)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--high-regret-hp", type=float, default=15.0)
    parser.add_argument("--high-absolute-hp-loss", type=float, default=40.0)
    parser.add_argument("--low-hp-threshold", type=float, default=0.40)
    parser.add_argument("--incoming-hp-loss-threshold", type=float, default=10.0)
    parser.add_argument("--high-entropy-threshold", type=float, default=0.55)
    parser.add_argument("--low-margin-threshold", type=float, default=0.15)
    parser.add_argument("--rare-action-threshold", type=int, default=20)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
    parser.add_argument("--restore-mode", default="cached_batch_auto_prepared")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--max-continuation-actions", type=int, default=500)
    parser.add_argument("--seed", default="train-on-policy-counterfactual-v0")
    parser.add_argument("--encounter-seed-trials", type=int, default=128)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.combat_limit < 1 or args.roots_per_combat < 1:
        parser.error("combat and root limits must be positive")
    if args.top_k < 2 or args.determinizations < 1:
        parser.error("Top-k must be at least 2 and determinizations must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report if args.list_only else {
        "status": report["status"],
        "completed_combats": report["completed_combats"],
        "combat_failures": report["combat_failures"],
        "selected_roots": report["selected_roots"],
        "teacher_eligible_roots": report["teacher_eligible_roots"],
        "informative_roots": report["informative_roots"],
        "policy_suboptimal_roots": report["policy_suboptimal_roots"],
        "terminal_worlds": report["terminal_worlds"],
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
