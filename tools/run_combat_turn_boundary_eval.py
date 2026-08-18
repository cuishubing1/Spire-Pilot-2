"""Evaluate P1 policy, one-step, and turn-boundary beam search.

The turn-boundary search compares root actions at a common temporal boundary:
the next player turn or exact combat termination.  P1 supplies root and
continuation candidates, while sts2-cli executes every retained sequence.
Only the visible draw-pile multiset is determinized.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch  # Keep Windows native runtime initialization order stable.


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_policy_guided_mcts import (  # noqa: E402
    POST_COMBAT_DECISIONS,
    SEARCH_DECISIONS,
    _cache_key,
    _candidate_command,
    _determinization,
    _engine,
    _expand_search_node,
    _leaf_outcome,
    _restore_search_root,
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
    _create_base_save,
    _prepare_scenario_save,
    _resolve_optional_precombat_selects,
    _run_policy,
)
from run_combat_mcts_comparison import _result, sha256_file_bytes  # noqa: E402
from run_combat_one_step_act_sweep import _run_one_step  # noqa: E402
from run_combat_policy_online import _load_policy, _rank_actions, _state_summary  # noqa: E402
from run_heldout_run_combat_comparison import (  # noqa: E402
    DEFAULT_COMBATS,
    DEFAULT_RUN_ID,
    DEFAULT_TARGETS,
    DEFAULT_TRANSITIONS,
    _load_scenarios,
)
from sts2_dataset.combat_lookahead import (  # noqa: E402
    apply_exact_terminal_death_veto,
    apply_policy_advantage_gate,
    choose_one_step_candidate,
    policy_top_k,
    required_search_categories,
    regularized_one_step_score,
)
from sts2_dataset.combat_engine_features import exact_transition_features  # noqa: E402
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    first_card_select_candidate,
)
from sts2_dataset.combat_search import (  # noqa: E402
    lower_tail_cvar,
    normalized_policy_entropy,
    risk_adjusted_root_score,
)
from sts2_dataset.legal_actions import enumerate_legal_actions  # noqa: E402
from sts2_dataset.util import canonical_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


TURN_BOUNDARY_SEARCH_VERSION = "combat-turn-boundary-search-0.6.0"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_turn_boundary_eval.json"


def _round(state: dict[str, Any]) -> int:
    return int(state.get("round") or 0)


def _is_forbidden_potion_action(
    candidate: dict[str, Any], forbidden_potion_ids: set[str] | None
) -> bool:
    return (
        candidate.get("action_type") in {"use_potion", "discard_potion"}
        and str(candidate.get("source_id") or "") in (forbidden_potion_ids or set())
    )


def _at_boundary(state: dict[str, Any], *, root_round: int) -> bool:
    decision = str(state.get("decision") or "")
    return (
        decision == "game_over"
        or decision in POST_COMBAT_DECISIONS
        or (decision == "combat_play" and _round(state) > root_round)
    )


def _path_prior_score(probabilities: list[float], weight: float) -> float:
    if not probabilities or weight <= 0.0:
        return 0.0
    return weight * statistics.fmean(math.log(max(float(value), 1e-8)) for value in probabilities)


def _execute_sequence(
    *,
    worker: Any,
    entrance_save: Path,
    enter_command: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    root_state: dict[str, Any],
    draw_order: list[str],
    commands: list[dict[str, Any]],
    restore_mode: str,
) -> tuple[dict[str, Any], float]:
    state, elapsed_ms = _restore_search_root(
        worker=worker,
        entrance_save=entrance_save,
        enter_command=enter_command,
        root_prefix=root_prefix,
        draw_order=draw_order,
        mode=restore_mode,
        suffix_commands=commands,
    )
    # The response is now the state *after* the batched candidate sequence, so
    # it cannot be compared directly with root_state here. Root-only restore
    # equivalence is covered by the shared restore gate and one-step evaluator;
    # the CLI suffix regression compares this final state with stepwise actions.
    return state, elapsed_ms


def _end_turn_row(ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            row for row in ranked
            if row["candidate"].get("action_type") == "end_turn"
        ),
        None,
    )


def _command_key(command: dict[str, Any]) -> str:
    return canonical_json({
        "action": command.get("action"),
        "args": command.get("args") or {},
    })


def _candidate_command_key(candidate: dict[str, Any]) -> str:
    return _command_key(_candidate_command(candidate))


def _shared_world_candidate_prefix(worlds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only actions selected identically in every determinization.

    The prefix ends at end_turn.  Any divergence caused by drawing or generating
    cards therefore forces a fresh search after the newly visible information.
    """
    sequences = [world.get("candidate_sequence") or [] for world in worlds]
    if not sequences:
        return []
    shared: list[dict[str, Any]] = []
    for index in range(min(len(sequence) for sequence in sequences)):
        candidates = [sequence[index] for sequence in sequences]
        if len({_candidate_command_key(candidate) for candidate in candidates}) != 1:
            break
        shared.append(candidates[0])
        if candidates[0].get("action_type") == "end_turn":
            break
    return shared


def _best_supported_leaf(leaves: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer leaves that actually reached the advertised search boundary."""

    if not leaves:
        raise EngineError("turn-boundary beam produced no leaf")
    supported = [
        leaf
        for leaf in leaves
        if bool(leaf.get("supported_boundary")) and not leaf.get("engine_error")
    ]
    pool = supported or leaves
    return max(
        pool,
        key=lambda row: (
            float(row["path_score"]),
            -len(row["commands"]),
        ),
    )


def _choose_turn_boundary_root(
    evaluations: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    *,
    minimum_value_advantage: float,
    minimum_end_turn_advantage: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    policy_choice = next(
        row for row in evaluations
        if row["candidate"]["candidate_id"] == shortlist[0]["candidate"]["candidate_id"]
    )
    if any(bool(row.get("selection_eligible", True)) for row in evaluations):
        search_choice = choose_one_step_candidate(evaluations)
        return apply_policy_advantage_gate(
            search_choice=search_choice,
            policy_choice=policy_choice,
            minimum_advantage=minimum_value_advantage,
            minimum_end_turn_advantage=minimum_end_turn_advantage,
        )

    # Search is advisory. A simulator coverage failure must not terminate an
    # otherwise executable continuous run: let the real root engine validate
    # the policy action, whose rejection path can exclude it and retry.
    return policy_choice, {
        "reason": "all_search_candidates_ineligible",
        "engine_errors": sorted({
            str(world["engine_error"])
            for row in evaluations
            for world in row["worlds"]
            if world.get("engine_error")
        }),
    }


def _command_is_legal(state: dict[str, Any], command: dict[str, Any]) -> bool:
    wanted = _command_key(command)
    return any(
        _command_key({"action": action.action, "args": action.args}) == wanted
        for action in enumerate_legal_actions(state)
    )


def _search_root_in_world(
    *,
    worker: Any,
    entrance_save: Path,
    enter_command: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    root_state: dict[str, Any],
    root_row: dict[str, Any],
    draw_order: list[str],
    determinization_id: str,
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    beam_width: int,
    max_player_actions: int,
    continuation_policy_weight: float,
    unsupported_penalty: float,
    node_index_start: int,
    inference_lock: threading.Lock | None = None,
    forbidden_potion_ids: set[str] | None = None,
    restore_mode: str = "cached_batch_auto_prepared",
) -> tuple[dict[str, Any], dict[str, float]]:
    root_round = _round(root_state)
    root_candidate = root_row["candidate"]
    frontier = [{
        "commands": [_candidate_command(root_candidate)],
        "probabilities": [float(root_row["policy_probability"])],
        "candidates": [root_candidate],
        "forced_end_turn": False,
    }]
    leaves: list[dict[str, Any]] = []
    total_restore_ms = 0.0
    total_engine_inference_ms = 0.0
    expanded_paths = 0
    node_index = node_index_start

    while frontier:
        next_paths: list[dict[str, Any]] = []
        for path in frontier:
            try:
                state, restore_ms = _execute_sequence(
                    worker=worker,
                    entrance_save=entrance_save,
                    enter_command=enter_command,
                    root_prefix=root_prefix,
                    root_state=root_state,
                    draw_order=draw_order,
                    commands=path["commands"],
                    restore_mode=restore_mode,
                )
            except EngineError as exc:
                # A nominally legal action can still be rejected by the game
                # engine (for example, a redundant non-stackable power).  That
                # invalidates this branch, not the entire root search.
                outcome = _leaf_outcome(
                    root_state,
                    root_state=root_state,
                    node=None,
                    objective=objective,
                    determinization_id=determinization_id,
                    unsupported_penalty=unsupported_penalty,
                    depth=len(path["commands"]),
                )
                expanded_paths += 1
                leaves.append({
                    **path,
                    "outcome": asdict(outcome),
                    "exact_transition": exact_transition_features(
                        root_state, root_state, root_candidate
                    ),
                    "path_score": -1.0e9,
                    "leaf_round": root_round,
                    "leaf_decision": "engine_rejected_action",
                    "engine_error": str(exc),
                    "supported_boundary": False,
                })
                continue
            total_restore_ms += restore_ms
            expanded_paths += 1
            decision = str(state.get("decision") or "")
            node = None
            if decision == "combat_play":
                if inference_lock is None:
                    node = _expand_search_node(
                        state,
                        model=model,
                        tensorizer=tensorizer,
                        device=device,
                        node_index=node_index,
                    )
                else:
                    with inference_lock:
                        node = _expand_search_node(
                            state,
                            model=model,
                            tensorizer=tensorizer,
                            device=device,
                            node_index=node_index,
                        )
                node_index += 1
                total_engine_inference_ms += node.inference_ms

            reached_boundary = _at_boundary(state, root_round=root_round)
            if reached_boundary or decision != "combat_play":
                outcome = _leaf_outcome(
                    state,
                    root_state=root_state,
                    node=node,
                    objective=objective,
                    determinization_id=determinization_id,
                    unsupported_penalty=unsupported_penalty,
                    depth=len(path["commands"]),
                )
                leaves.append({
                    **path,
                    "outcome": asdict(outcome),
                    "exact_transition": exact_transition_features(
                        root_state, state, root_candidate
                    ),
                    "path_score": float(outcome.value) + _path_prior_score(
                        path["probabilities"][1:], continuation_policy_weight
                    ),
                    "leaf_round": _round(state),
                    "leaf_decision": decision,
                    "supported_boundary": reached_boundary,
                })
                continue

            assert node is not None
            if len(path["commands"]) >= max_player_actions:
                end_row = _end_turn_row(node.ranked)
                if end_row is None:
                    outcome = _leaf_outcome(
                        state,
                        root_state=root_state,
                        node=node,
                        objective=objective,
                        determinization_id=determinization_id,
                        unsupported_penalty=unsupported_penalty,
                        depth=len(path["commands"]),
                    )
                    leaves.append({
                        **path,
                        "outcome": asdict(outcome),
                        "exact_transition": exact_transition_features(
                            root_state, state, root_candidate
                        ),
                        "path_score": float(outcome.value) - unsupported_penalty,
                        "leaf_round": _round(state),
                        "leaf_decision": decision,
                        "supported_boundary": False,
                    })
                else:
                    next_paths.append({
                        "commands": path["commands"] + [_candidate_command(end_row["candidate"])],
                        "probabilities": path["probabilities"] + [
                            float(end_row["policy_probability"])
                        ],
                        "candidates": path["candidates"] + [end_row["candidate"]],
                        "forced_end_turn": True,
                        "partial_score": _path_prior_score(
                            path["probabilities"][1:] + [float(end_row["policy_probability"])],
                            continuation_policy_weight,
                        ),
                    })
                continue

            partial_outcome = _leaf_outcome(
                state,
                root_state=root_state,
                node=node,
                objective=objective,
                determinization_id=determinization_id,
                unsupported_penalty=unsupported_penalty,
                depth=len(path["commands"]),
            )
            continuation_ranked = [
                row for row in node.ranked
                if not _is_forbidden_potion_action(
                    row["candidate"], forbidden_potion_ids
                )
            ]
            for row in policy_top_k(
                continuation_ranked,
                beam_width,
                required_categories=required_search_categories(
                    node.sample["observation"]
                ),
            ):
                probabilities = path["probabilities"] + [float(row["policy_probability"])]
                next_paths.append({
                    "commands": path["commands"] + [_candidate_command(row["candidate"])],
                    "probabilities": probabilities,
                    "candidates": path["candidates"] + [row["candidate"]],
                    "forced_end_turn": False,
                    "partial_score": float(partial_outcome.value) + _path_prior_score(
                        probabilities[1:], continuation_policy_weight
                    ),
                })
        if not next_paths:
            break
        frontier = sorted(
            next_paths,
            key=lambda row: (
                -float(row.get("partial_score", 0.0)),
                tuple(command.get("action", "") for command in row["commands"]),
            ),
        )[:beam_width]

    best = _best_supported_leaf(leaves)
    return best, {
        "restore_and_engine_ms": total_restore_ms,
        "value_inference_ms": total_engine_inference_ms,
        "expanded_paths": float(expanded_paths),
    }


def turn_boundary_current_root(
    *,
    workers: list[Any],
    entrance_save: Path,
    enter_command: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    root_state: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    root_top_k: int,
    beam_width: int,
    max_player_actions: int,
    policy_log_weight: float,
    continuation_policy_weight: float,
    minimum_value_advantage: float,
    minimum_end_turn_advantage: float,
    unsupported_penalty: float,
    determinization_count: int,
    cvar_alpha: float,
    cvar_weight: float,
    search_seed: int,
    step: int,
    excluded_candidate_ids: set[str] | None = None,
    forbidden_potion_ids: set[str] | None = None,
    required_potion_ids: set[str] | None = None,
    restore_mode: str = "cached_batch_auto_prepared",
) -> dict[str, Any]:
    started = time.perf_counter()
    from sts2_dataset.combat_online import headless_state_to_model_sample

    sample = headless_state_to_model_sample(
        root_state,
        transition_id=f"turn-boundary:root:{step}",
        combat_id="turn-boundary",
    )
    ranked, root_inference_ms = _rank_actions(
        model, tensorizer, sample, device=device, objective=None
    )
    if excluded_candidate_ids:
        ranked = [
            row for row in ranked
            if row["candidate"]["candidate_id"] not in excluded_candidate_ids
        ]
    if forbidden_potion_ids:
        ranked = [
            row for row in ranked
            if not _is_forbidden_potion_action(row["candidate"], forbidden_potion_ids)
        ]
    if not ranked:
        raise EngineError("turn-boundary search has no candidates after exclusions")
    required_categories = required_search_categories(sample["observation"])
    semantic_ranked = policy_top_k(
        ranked,
        len(ranked),
        required_categories=required_categories,
    )
    shortlist = policy_top_k(
        ranked,
        root_top_k,
        required_categories=required_categories,
    )
    if required_potion_ids:
        existing = {
            str(row["candidate"].get("candidate_id") or "") for row in shortlist
        }
        for row in ranked:
            candidate = row["candidate"]
            if (
                candidate.get("action_type") == "use_potion"
                and str(candidate.get("source_id") or "") in required_potion_ids
                and str(candidate.get("candidate_id") or "") not in existing
            ):
                shortlist.append(row)
                existing.add(str(candidate.get("candidate_id") or ""))
    visible_draw = _visible_draw_multiset(root_state)
    determinizations = [
        _determinization(visible_draw, search_seed=search_seed, simulation=index)
        for index in range(determinization_count)
    ]
    evaluations: list[dict[str, Any]] = []
    total_restore_ms = 0.0
    total_value_inference_ms = 0.0
    expanded_paths = 0

    if not workers:
        raise EngineError("turn-boundary search requires at least one engine worker")
    jobs = [
        (root_index, world_index, root_row, draw_order, determinization_id)
        for root_index, root_row in enumerate(shortlist)
        for world_index, (draw_order, determinization_id) in enumerate(determinizations)
    ]
    buckets: list[list[tuple[Any, ...]]] = [[] for _ in workers]
    for index, job in enumerate(jobs):
        buckets[index % len(workers)].append(job)
    inference_lock = threading.Lock()

    def run_bucket(worker: Any, bucket: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for root_index, world_index, root_row, draw_order, determinization_id in bucket:
            best, counters = _search_root_in_world(
                worker=worker,
                entrance_save=entrance_save,
                enter_command=enter_command,
                root_prefix=root_prefix,
                root_state=root_state,
                root_row=root_row,
                draw_order=draw_order,
                determinization_id=determinization_id,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                beam_width=beam_width,
                max_player_actions=max_player_actions,
                continuation_policy_weight=continuation_policy_weight,
                unsupported_penalty=unsupported_penalty,
                node_index_start=(root_index * 1000) + (world_index * 100) + 1,
                inference_lock=inference_lock,
                forbidden_potion_ids=forbidden_potion_ids,
                restore_mode=restore_mode,
            )
            rows.append((root_index, world_index, determinization_id, best, counters))
        return rows

    worker_results: list[tuple[Any, ...]] = []
    active = [(worker, bucket) for worker, bucket in zip(workers, buckets) if bucket]
    if len(active) == 1:
        worker_results.extend(run_bucket(*active[0]))
    else:
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = [executor.submit(run_bucket, worker, bucket) for worker, bucket in active]
            for future in futures:
                worker_results.extend(future.result())
    indexed = {
        (int(root_index), int(world_index)): (determinization_id, best, counters)
        for root_index, world_index, determinization_id, best, counters in worker_results
    }

    for root_index, root_row in enumerate(shortlist):
        worlds: list[dict[str, Any]] = []
        for world_index in range(len(determinizations)):
            determinization_id, best, counters = indexed[(root_index, world_index)]
            total_restore_ms += counters["restore_and_engine_ms"]
            total_value_inference_ms += counters["value_inference_ms"]
            expanded_paths += int(counters["expanded_paths"])
            worlds.append({
                "determinization_id": determinization_id,
                "outcome": best["outcome"],
                "exact_transition": best["exact_transition"],
                "path_score": best["path_score"],
                "leaf_round": best["leaf_round"],
                "leaf_decision": best["leaf_decision"],
                "forced_end_turn": best["forced_end_turn"],
                "candidate_sequence": best["candidates"],
                "engine_error": best.get("engine_error"),
                "supported_boundary": bool(best.get("supported_boundary")),
            })
        values = [float(world["outcome"]["value"]) for world in worlds]
        mean_value = statistics.fmean(values)
        tail_value = lower_tail_cvar(values, cvar_alpha)
        risk_value = risk_adjusted_root_score(
            mean_value=mean_value,
            lower_tail_value=tail_value,
            mean_weight=1.0 - cvar_weight,
            cvar_weight=cvar_weight,
        )
        evaluations.append({
            "candidate": root_row["candidate"],
            "policy_probability": float(root_row["policy_probability"]),
            "selection_score": regularized_one_step_score(
                value=risk_value,
                policy_probability=float(root_row["policy_probability"]),
                policy_log_weight=policy_log_weight,
            ),
            "selection_eligible": all(
                bool(world.get("supported_boundary"))
                and not world.get("engine_error")
                for world in worlds
            ),
            "mean_value": mean_value,
            "lower_tail_cvar": tail_value,
            "risk_adjusted_value": risk_value,
            "worlds": worlds,
        })

    exact_terminal_death_vetoes = apply_exact_terminal_death_veto(evaluations)

    chosen, fallback = _choose_turn_boundary_root(
        evaluations,
        shortlist,
        minimum_value_advantage=minimum_value_advantage,
        minimum_end_turn_advantage=minimum_end_turn_advantage,
    )
    chosen_evaluation = next(
        row for row in evaluations
        if row["candidate"]["candidate_id"] == chosen["candidate"]["candidate_id"]
    )
    shared_sequence = (
        []
        if fallback and fallback.get("reason") == "all_search_candidates_ineligible"
        else _shared_world_candidate_prefix(chosen_evaluation["worlds"])
    )
    return {
        "schema_version": TURN_BOUNDARY_SEARCH_VERSION,
        "root_top_k": len(shortlist),
        "raw_root_candidate_count": len(ranked),
        "semantic_root_candidate_count": len(semantic_ranked),
        "deduplicated_root_candidate_count": len(ranked) - len(semantic_ranked),
        "required_root_categories": list(required_categories),
        "forbidden_potion_ids": sorted(forbidden_potion_ids or set()),
        "required_potion_ids": sorted(required_potion_ids or set()),
        "beam_width": beam_width,
        "max_player_actions": max_player_actions,
        "determinization_count": determinization_count,
        "search_worker_count": len(workers),
        "information_boundary": "visible_draw_multiset_determinization",
        "leaf_boundary": "next_player_turn_or_combat_terminal",
        "policy_candidate": shortlist[0]["candidate"],
        "chosen_candidate": chosen["candidate"],
        "shared_candidate_sequence": shared_sequence,
        "shared_candidate_sequence_length": len(shared_sequence),
        "fallback": fallback,
        "exact_terminal_death_vetoes": exact_terminal_death_vetoes,
        "root_policy_entropy": normalized_policy_entropy(
            [float(row["policy_probability"]) for row in ranked]
        ),
        "root_inference_ms": round(root_inference_ms, 3),
        "value_inference_ms": round(total_value_inference_ms, 3),
        "engine_restore_and_action_ms": round(total_restore_ms, 3),
        "expanded_paths": expanded_paths,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "evaluations": evaluations,
    }


def _run_turn_boundary(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    initial_prefix_commands: list[dict[str, Any]] | None = None,
    forced_root_candidate: dict[str, Any] | None = None,
    expected_root_signature: str | None = None,
    forbidden_potion_ids_until_next_turn: set[str] | None = None,
    forbidden_potion_ids: set[str] | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    pending_plan: list[dict[str, Any]] = []
    pending_plan_source_step: int | None = None
    with ExitStack() as stack:
        real_engine = stack.enter_context(_engine(args, game_data_dir))
        workers = [
            stack.enter_context(_engine(args, game_data_dir))
            for _ in range(args.search_workers)
        ]
        state, _ = real_engine.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
        state, _ = real_engine.send({
            "cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]
        })
        state, precombat_prefix = _resolve_optional_precombat_selects(real_engine, state)
        prefix.extend(precombat_prefix)
        for command in initial_prefix_commands or []:
            state, _ = real_engine.send(command)
            prefix.append(command)
        if state.get("decision") != "combat_play":
            raise EngineError(f"initial prefix did not reach combat_play: {state!r}")
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        initial_round = _round(state)
        root_signature = sha256_file_bytes(json.dumps(state, sort_keys=True).encode("utf-8"))
        if expected_root_signature is not None and root_signature != expected_root_signature:
            raise EngineError(
                f"forced rollout root mismatch: {root_signature} != {expected_root_signature}"
            )
        for worker in workers:
            cached, _ = worker.send({
                "cmd": "cache_save", "name": _cache_key(entrance_save), "path": str(entrance_save)
            })
            if cached.get("type") != "ok":
                raise EngineError(f"turn-boundary worker could not cache save: {cached!r}")
        for step in range(args.max_actions):
            decision = str(state.get("decision") or "")
            if decision not in SEARCH_DECISIONS:
                break
            before = _state_summary(state)
            if decision == "combat_play":
                search = None
                plan_reused = False
                plan_invalidated = False
                plan_source_step = pending_plan_source_step
                forced_action = step == 0 and forced_root_candidate is not None
                if forced_action:
                    candidate = forced_root_candidate
                    command = candidate_to_headless_command(candidate)
                    if not _command_is_legal(state, command):
                        raise EngineError(
                            f"forced root candidate is not legal: {candidate!r}"
                        )
                    pending_plan.clear()
                    pending_plan_source_step = None
                elif args.reuse_turn_plan and pending_plan:
                    planned_candidate = pending_plan[0]
                    planned_command = candidate_to_headless_command(planned_candidate)
                    if _command_is_legal(state, planned_command):
                        candidate = pending_plan.pop(0)
                        command = planned_command
                        plan_reused = True
                    else:
                        pending_plan.clear()
                        pending_plan_source_step = None
                        plan_invalidated = True
                if not forced_action and not plan_reused:
                    active_forbidden_potions = set(forbidden_potion_ids or set())
                    if _round(state) <= initial_round:
                        active_forbidden_potions.update(
                            forbidden_potion_ids_until_next_turn or set()
                        )
                    search = turn_boundary_current_root(
                        workers=workers,
                        entrance_save=entrance_save,
                        enter_command={
                            "cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]
                        },
                        root_prefix=prefix,
                        root_state=state,
                        model=model,
                        tensorizer=tensorizer,
                        device=device,
                        objective=objective,
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
                        search_seed=args.search_seed + step * 100003,
                        step=step,
                        forbidden_potion_ids=active_forbidden_potions or None,
                        restore_mode=args.restore_mode,
                    )
                    candidate = search["chosen_candidate"]
                    command = candidate_to_headless_command(candidate)
                    if args.reuse_turn_plan:
                        shared = list(search["shared_candidate_sequence"])
                        if shared and _candidate_command_key(shared[0]) == _candidate_command_key(candidate):
                            pending_plan = shared[1:]
                            pending_plan_source_step = step if pending_plan else None
                            plan_source_step = step
                        else:
                            pending_plan.clear()
                            pending_plan_source_step = None
            else:
                pending_plan.clear()
                pending_plan_source_step = None
                actions = enumerate_legal_actions(state)
                candidate = first_card_select_candidate(state)
                command = candidate_to_headless_command(candidate)
                search = None
                plan_reused = False
                plan_invalidated = False
                plan_source_step = None
            state, engine_ms = real_engine.send(command)
            prefix.append(command)
            steps.append({
                "step": step,
                "before": before,
                "chosen_candidate": candidate,
                "turn_boundary": search,
                "turn_plan_reused": plan_reused,
                "turn_plan_invalidated": plan_invalidated,
                "turn_plan_source_step": plan_source_step,
                "forced_root_action": forced_action if decision == "combat_play" else False,
                "engine_ms": round(engine_ms, 3),
                "after": _state_summary(state),
            })
    result = _result(state, initial_hp=initial_hp, steps=steps)
    result["root_signature"] = root_signature
    result["initial_prefix_length"] = len(initial_prefix_commands or [])
    result["forced_root_candidate_id"] = (
        str(forced_root_candidate.get("candidate_id"))
        if forced_root_candidate is not None else None
    )
    result["forbidden_potion_ids_until_next_turn"] = sorted(
        forbidden_potion_ids_until_next_turn or set()
    )
    result["forbidden_potion_ids"] = sorted(forbidden_potion_ids or set())
    searches = [row["turn_boundary"] for row in steps if row["turn_boundary"] is not None]
    result["search_decision_count"] = len(searches)
    result["policy_action_change_count"] = sum(
        row["chosen_candidate"]["candidate_id"] != row["policy_candidate"]["candidate_id"]
        for row in searches
    )
    result["total_search_ms"] = round(sum(float(row["wall_ms"]) for row in searches), 3)
    result["mean_search_ms"] = round(
        statistics.fmean(float(row["wall_ms"]) for row in searches), 3
    ) if searches else None
    result["expanded_paths"] = sum(int(row["expanded_paths"]) for row in searches)
    result["turn_plan_reused_action_count"] = sum(
        bool(row.get("turn_plan_reused")) for row in steps
    )
    result["turn_plan_invalidation_count"] = sum(
        bool(row.get("turn_plan_invalidated")) for row in steps
    )
    return result


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows]
    summary: dict[str, Any] = {
        "combats": len(values),
        "completed": sum(value.get("status") != "death" for value in values),
        "deaths": sum(value.get("status") == "death" for value in values),
        "total_hp_loss": round(sum(float(value["hp_loss"]) for value in values), 3),
        "mean_hp_loss": round(statistics.fmean(float(value["hp_loss"]) for value in values), 3),
    }
    if key == "turn_boundary":
        summary.update({
            "search_decisions": sum(int(value["search_decision_count"]) for value in values),
            "policy_action_changes": sum(
                int(value["policy_action_change_count"]) for value in values
            ),
            "expanded_paths": sum(int(value["expanded_paths"]) for value in values),
            "total_search_ms": round(
                sum(float(value["total_search_ms"]) for value in values), 3
            ),
            "reused_turn_plan_actions": sum(
                int(value["turn_plan_reused_action_count"]) for value in values
            ),
        })
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.search_workers < 1:
        raise EngineError("search-workers must be at least 1")
    checkpoint = args.checkpoint.resolve()
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if device != "cuda":
        raise EngineError(f"CUDA is required; resolved device={device!r}")
    if model.state_value_head is None:
        raise EngineError("turn-boundary search requires a state value head")
    objective = CombatObjective.from_config(model.config)
    scenarios = _load_scenarios(args)
    if args.scenario_ids:
        wanted = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row["scenario_id"] in wanted]
        if not scenarios:
            raise EngineError(f"no held-out scenarios matched: {sorted(wanted)}")
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    results = []
    with tempfile.TemporaryDirectory(prefix="sts2_turn_boundary_eval_") as temp_dir:
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
            one_step = _run_one_step(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
            )
            turn_boundary = _run_turn_boundary(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
            )
            signatures = {
                root["root_signature"],
                policy["root_signature"],
                one_step["root_signature"],
                turn_boundary["root_signature"],
            }
            if len(signatures) != 1:
                raise EngineError(f"root mismatch in {scenario['scenario_id']}")
            row = {
                **{key: value for key, value in scenario.items() if key != "player"},
                "snapshot": scenario["player"],
                "root": root,
                "policy": policy,
                "one_step": one_step,
                "turn_boundary": turn_boundary,
            }
            results.append(row)
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "policy": {"status": policy["status"], "hp_loss": policy["hp_loss"]},
                "one_step": {"status": one_step["status"], "hp_loss": one_step["hp_loss"]},
                "turn_boundary": {
                    "status": turn_boundary["status"],
                    "hp_loss": turn_boundary["hp_loss"],
                    "changes": turn_boundary["policy_action_change_count"],
                    "search_ms": turn_boundary["total_search_ms"],
                },
            }, ensure_ascii=False), flush=True)
    summary = {
        key: _summarize(results, key)
        for key in ("policy", "one_step", "turn_boundary")
    }
    summary["by_act"] = {
        str(act): {
            key: _summarize([row for row in results if int(row["act"]) == act], key)
            for key in ("policy", "one_step", "turn_boundary")
        }
        for act in sorted({int(row["act"]) for row in results})
    }
    return {
        "schema_version": "combat-turn-boundary-eval-0.2.0",
        "search_version": TURN_BOUNDARY_SEARCH_VERSION,
        "generated_at": utc_now(),
        "status": "pass",
        "run_id": args.run_id,
        "device": device,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "configuration": {
            "environment_seed": args.seed,
            "search_seed": args.search_seed,
            "reuse_turn_plan": args.reuse_turn_plan,
            "search_workers": args.search_workers,
            "engine_restore_mode": args.restore_mode,
            "root_top_k": args.root_top_k,
            "beam_width": args.beam_width,
            "max_player_actions": args.max_player_actions,
            "determinizations": args.determinizations,
            "policy_log_weight": args.policy_log_weight,
            "continuation_policy_weight": args.continuation_policy_weight,
            "minimum_value_advantage": args.minimum_value_advantage,
            "minimum_end_turn_advantage": args.minimum_end_turn_advantage,
            "cvar_alpha": args.cvar_alpha,
            "cvar_weight": args.cvar_weight,
        },
        "comparison_semantics": {
            "policy_vs_one_step_vs_turn_boundary": (
                "same visible entrance snapshot, generated save, encounter, and engine RNG root"
            ),
            "turn_boundary_leaf": "next player turn or exact combat terminal",
            "parallelism": (
                "root-action by visible-determinization evaluations run on independent "
                "sts2-cli processes; CUDA inference is serialized and aggregation is unchanged"
            ),
            "development_boundary": (
                "the repeatedly inspected A0 run is a development set, not a final test set"
            ),
        },
        "summary": summary,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "scenarios": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--combats", type=Path, default=DEFAULT_COMBATS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scenario-id", dest="scenario_ids", action="append")
    parser.add_argument("--seed", default="heldout-a0-controlled-reconstruction-v0")
    parser.add_argument("--search-seed", type=int, default=20260816)
    parser.add_argument("--root-top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-player-actions", type=int, default=3)
    parser.add_argument(
        "--reuse-turn-plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse only the action prefix shared by every determinization.",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=1,
        help="Independent sts2-cli workers used for root/world evaluations.",
    )
    parser.add_argument("--policy-log-weight", type=float, default=0.05)
    parser.add_argument("--continuation-policy-weight", type=float, default=0.01)
    parser.add_argument("--minimum-value-advantage", type=float, default=0.02)
    parser.add_argument("--minimum-end-turn-advantage", type=float, default=0.15)
    parser.add_argument("--minimum-potion-policy-probability", type=float, default=0.0)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--restore-mode",
        choices=(
            "cached_batch_auto_prepared",
            "cached_batch_auto",
            "cached_batch_compact",
            "cached_batch",
        ),
        default="cached_batch_auto_prepared",
    )
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # Compatibility with the one-step evaluator called in the same report.
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "scenarios": len(report["scenarios"]),
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
