"""Run the first policy-guided determinized PUCT search against sts2-cli.

This is a simulator/search gate, not a strength claim.  Every simulation
restores the combat entrance save, samples a draw-pile order from the visible
multiset, replays its selected action path in the real engine, and backs up
separate value/risk/resource statistics. The cached-batch restore mode keeps
the save JSON in the engine process and replays the shared real-action prefix
without per-action IPC or intermediate state serialization. New checkpoints use an independently
supervised state value head; older action-resource checkpoints remain a
compatibility fallback and are explicitly marked provisional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch  # Import before project modules; see run_combat_policy_online.py.


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
from run_combat_policy_online import (  # noqa: E402
    _advance_initial_event,
    _load_policy,
)
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_engine_features import ground_future_max_hp_delta  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
)
from sts2_dataset.legal_actions import enumerate_legal_actions  # noqa: E402
from sts2_dataset.combat_tool import rank_combat_actions  # noqa: E402
from sts2_dataset.combat_search import (  # noqa: E402
    SEARCH_VERSION,
    SearchEdgeStats,
    SearchOutcome,
    build_search_teacher_record,
    forced_root_action_index,
    lower_tail_cvar,
    paired_root_determinization_index,
    puct_score,
    risk_adjusted_root_score,
    root_coverage_budget,
    root_visit_threshold,
)
from sts2_dataset.util import (  # noqa: E402
    canonical_json,
    load_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


DEFAULT_LATEST = REPO_ROOT / "artifacts" / "combat_policy_value_v1" / "latest.json"
DEFAULT_CONFIG = REPO_ROOT / "config" / "combat_search_v0.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "policy_guided_mcts_v0.json"
COMPACT_PREFIX_MIN_ACTIONS = 4
POST_COMBAT_DECISIONS = {"card_reward", "map_select", "reward", "rewards"}
SEARCH_DECISIONS = {"combat_play", "card_select"}


@dataclass
class TreeNode:
    key: str
    sample: dict[str, Any]
    ranked: list[dict[str, Any]]
    candidates: dict[str, dict[str, Any]]
    edges: dict[str, SearchEdgeStats]
    inference_ms: float
    state_value: dict[str, float] | None = None


def _engine(args: argparse.Namespace, game_data_dir: Path) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet.resolve(),
        engine_dll=args.engine_dll.resolve(),
        game_data_dir=game_data_dir,
        sts2_lib=args.sts2_lib.resolve(),
        timeout_s=args.timeout,
    )


def _resolve_checkpoint(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    latest = load_json(DEFAULT_LATEST)
    checkpoint = Path(latest["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    return checkpoint.resolve()


def _monster_choice(state: dict[str, Any]) -> dict[str, Any]:
    try:
        return next(value for value in state.get("choices") or [] if value.get("type") == "Monster")
    except StopIteration as exc:
        raise EngineError("map state contains no Monster choice") from exc


def _enter_command(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": "action",
        "action": "select_map_node",
        "args": {"col": choice["col"], "row": choice["row"]},
    }


def _entry(identity: str) -> str:
    return identity.split(".", 1)[-1]


def _visible_draw_multiset(state: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    for row in (state.get("piles") or {}).get("draw") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        cards.extend([_entry(str(row["id"]))] * int(row.get("count") or 1))
    return cards


def _cache_key(entrance_save: Path) -> str:
    return f"combat:{hashlib.sha256(str(entrance_save.resolve()).encode('utf-8')).hexdigest()[:16]}"


def _prefix_steps(root_prefix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for command in root_prefix:
        if command.get("cmd") != "action":
            raise EngineError(f"batched restore only supports action prefixes: {command!r}")
        steps.append({
            "action": command.get("action"),
            "args": command.get("args") or {},
        })
    return steps


def _restore_search_root(
    *,
    worker: EngineProcess,
    entrance_save: Path,
    enter_command: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    draw_order: list[str],
    mode: str,
    suffix_commands: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], float]:
    if mode in {
        "cached_batch",
        "cached_batch_compact",
        "cached_batch_auto",
        "cached_batch_auto_prepared",
    }:
        command: dict[str, Any] = {
            "cmd": "restore_combat",
            "cache": _cache_key(entrance_save),
            "lang": "en",
            "entry": enter_command,
            "prefix": _prefix_steps(root_prefix),
        }
        if draw_order:
            command["draw_order"] = draw_order
        if suffix_commands:
            command["suffix"] = _prefix_steps(suffix_commands)
        if mode == "cached_batch_compact" or (
            mode in {"cached_batch_auto", "cached_batch_auto_prepared"}
            and len(root_prefix) + len(suffix_commands or [])
            >= COMPACT_PREFIX_MIN_ACTIONS
        ):
            command["prefix_projection"] = "compact"
        if mode == "cached_batch_auto_prepared":
            command["reuse_prepared_save"] = True
        return worker.send(command)
    if mode != "legacy":
        raise EngineError(f"unsupported engine restore mode: {mode}")

    elapsed_ms = 0.0
    state, current_ms = worker.send({
        "cmd": "reload_save", "path": str(entrance_save), "lang": "en"
    })
    elapsed_ms += current_ms
    if state.get("decision") != "map_select":
        raise EngineError(f"worker reload failed: {state!r}")
    state, current_ms = worker.send(enter_command)
    elapsed_ms += current_ms
    for command in root_prefix:
        state, current_ms = worker.send(command)
        elapsed_ms += current_ms
    if draw_order:
        _, current_ms = worker.send({"cmd": "set_draw_order", "cards": draw_order})
        elapsed_ms += current_ms
    for command in suffix_commands or []:
        state, current_ms = worker.send(command)
        elapsed_ms += current_ms
    return state, elapsed_ms


def _determinization(cards: list[str], *, search_seed: int, simulation: int) -> tuple[list[str], str]:
    digest = hashlib.sha256(f"{search_seed}:{simulation}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(cards)
    rng.shuffle(order)
    identity = hashlib.sha256(canonical_json(order).encode("utf-8")).hexdigest()[:16]
    return order, identity


def _information_key(sample: dict[str, Any]) -> str:
    # Combat Observation V0 sorts unordered piles and contains no audit draw
    # order.  Candidate IDs are included to keep source/target bindings exact.
    payload = {
        "observation": sample["observation"],
        "candidate_ids": sorted(row["candidate_id"] for row in sample["candidates"]),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _expand_node(
    state: dict[str, Any],
    *,
    sample: dict[str, Any] | None,
    model: Any,
    tensorizer: Any,
    device: str,
    node_index: int,
    reuse_entity_encoding: bool,
    encounter_signature: str | None = None,
) -> TreeNode:
    if sample is None:
        sample = headless_state_to_model_sample(
            state,
            transition_id=f"mcts:node:{node_index}",
            combat_id="mcts",
            encounter_signature=encounter_signature,
        )
    ranked, inference_ms, diagnostics = rank_combat_actions(
        model,
        tensorizer,
        sample,
        device=device,
        objective=None,
        reuse_entity_encoding=reuse_entity_encoding,
    )
    state_value: dict[str, float] | None = None
    if model.state_value_head is not None:
        state_risk = diagnostics.get("state_risk")
        if not isinstance(state_risk, dict):
            raise EngineError("state value head did not expose state-risk diagnostics")
        state_value = {
            "hp_loss_fraction": float(state_risk["hp_loss_fraction"]),
            "death_probability": float(state_risk["death_probability"]),
            "potion_spent": float(state_risk["potion_spent"]),
            "max_hp_delta": float(state_risk["max_hp_delta"]),
        }
    candidates = {row["candidate_id"]: row for row in sample["candidates"]}
    probability = {
        row["candidate"]["candidate_id"]: float(row["policy_probability"])
        for row in ranked
    }
    edges = {
        action_id: SearchEdgeStats(prior=probability[action_id])
        for action_id in candidates
    }
    return TreeNode(
        key=_information_key(sample),
        sample=sample,
        ranked=ranked,
        candidates=candidates,
        edges=edges,
        inference_ms=inference_ms,
        state_value=state_value,
    )


def _card_select_information_key(
    state: dict[str, Any], candidates: list[dict[str, Any]]
) -> str:
    payload = {
        "decision": "card_select",
        "context": state.get("context"),
        "cards": state.get("cards"),
        "min_select": state.get("min_select"),
        "max_select": state.get("max_select"),
        "candidate_ids": sorted(row["candidate_id"] for row in candidates),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _expand_card_select_node(state: dict[str, Any]) -> TreeNode:
    actions = enumerate_legal_actions(state)
    if not actions:
        raise EngineError("card_select exposed no legal combinations")
    probability = 1.0 / len(actions)
    candidate_rows: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        candidate = {
            "candidate_id": action.action_id,
            "candidate_index": index,
            "action_type": action.action,
            "source_type": "card_selection",
            "source_id": None,
            "source_index": None,
            "source_ref": None,
            "target_kind": "selection",
            "target_id": None,
            "target_index": None,
            "target_ref": None,
            "engine_action": {"action_id": action.action, "args": action.args},
        }
        candidate_rows.append(candidate)
        ranked.append({"candidate": candidate, "policy_probability": probability})
    candidates = {row["candidate_id"]: row for row in candidate_rows}
    return TreeNode(
        key=_card_select_information_key(state, candidate_rows),
        sample={"observation": state, "candidates": candidate_rows},
        ranked=ranked,
        candidates=candidates,
        edges={
            action_id: SearchEdgeStats(prior=probability) for action_id in candidates
        },
        inference_ms=0.0,
    )


def _expand_search_node(
    state: dict[str, Any],
    *,
    sample: dict[str, Any] | None = None,
    model: Any,
    tensorizer: Any,
    device: str,
    node_index: int,
    reuse_entity_encoding: bool,
    encounter_signature: str | None = None,
) -> TreeNode:
    decision = str(state.get("decision") or "")
    if decision == "combat_play":
        return _expand_node(
            state,
            sample=sample,
            model=model,
            tensorizer=tensorizer,
            device=device,
            node_index=node_index,
            reuse_entity_encoding=reuse_entity_encoding,
            encounter_signature=encounter_signature,
        )
    if decision == "card_select":
        return _expand_card_select_node(state)
    raise EngineError(f"unsupported search decision: {decision or 'unknown'}")


def _search_information_key(state: dict[str, Any], sample: dict[str, Any] | None) -> str:
    if state.get("decision") == "combat_play":
        if sample is None:
            raise EngineError("combat node requires a model sample")
        return _information_key(sample)
    if state.get("decision") == "card_select":
        actions = enumerate_legal_actions(state)
        candidates = [
            {"candidate_id": action.action_id} for action in actions
        ]
        return _card_select_information_key(state, candidates)
    raise EngineError(f"unsupported search decision: {state.get('decision')!r}")


def _candidate_command(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("action_type") in {"select_cards", "skip_select"}:
        engine_action = candidate["engine_action"]
        return {
            "cmd": "action",
            "action": engine_action["action_id"],
            "args": dict(engine_action.get("args") or {}),
        }
    return candidate_to_headless_command(candidate)


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "total": round(sum(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def _state_debug_signature(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "sha256": hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest(),
        "decision": state.get("decision"),
        "round": state.get("round"),
        "energy": state.get("energy"),
        "player": {
            "hp": player.get("hp"),
            "block": player.get("block"),
        },
        "hand": [
            {
                "id": row.get("id"),
                "cost": row.get("cost"),
                "entity_ref": row.get("entity_ref"),
            }
            for row in state.get("hand") or []
        ],
        "enemies": [
            {
                "id": row.get("id"),
                "hp": row.get("hp"),
                "block": row.get("block"),
                "intents": row.get("intents"),
            }
            for row in state.get("enemies") or []
        ],
    }


def _state_difference_paths(
    expected: Any, actual: Any, *, limit: int = 20
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if type(left) is not type(right):
            differences.append({
                "path": path,
                "expected": left,
                "actual": right,
            })
            return
        if isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    differences.append({
                        "path": f"{path}.{key}",
                        "expected": left.get(key, "<missing>"),
                        "actual": right.get(key, "<missing>"),
                    })
                else:
                    visit(left[key], right[key], f"{path}.{key}")
                if len(differences) >= limit:
                    return
            return
        if isinstance(left, list):
            if len(left) != len(right):
                differences.append({
                    "path": f"{path}.length",
                    "expected": len(left),
                    "actual": len(right),
                })
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                visit(left_value, right_value, f"{path}[{index}]")
                if len(differences) >= limit:
                    return
            return
        if left != right:
            differences.append({"path": path, "expected": left, "actual": right})

    visit(expected, actual, "$state")
    return differences


def _resource_counts(state: dict[str, Any]) -> tuple[float, float, int]:
    player = state.get("player") or {}
    return (
        float(player.get("hp") or 0.0),
        float(player.get("max_hp") or 1.0),
        len(player.get("potions") or []),
    )


def _utility(
    *,
    hp_loss_fraction: float,
    death_probability: float,
    potion_spent: float,
    max_hp_delta: float,
    objective: CombatObjective,
) -> float:
    # Immediate HP loss is already included exactly in root-to-leaf HP loss;
    # adding the former learned immediate head would double count it.
    return (
        -objective.hp_loss_weight * float(hp_loss_fraction)
        -objective.death_penalty * float(death_probability)
        -objective.potion_cost * float(potion_spent)
        +objective.max_hp_gain_weight * float(max_hp_delta)
    )


def _leaf_outcome(
    state: dict[str, Any],
    *,
    root_state: dict[str, Any],
    node: TreeNode | None,
    objective: CombatObjective,
    determinization_id: str,
    unsupported_penalty: float,
    depth: int,
) -> SearchOutcome:
    root_hp, root_max_hp, root_potions = _resource_counts(root_state)
    current_hp, current_max_hp, current_potions = _resource_counts(state)
    actual_hp_loss_fraction = max(0.0, root_hp - current_hp) / max(root_max_hp, 1.0)
    actual_potion_spent = float(max(0, root_potions - current_potions))
    actual_max_hp_delta = current_max_hp - root_max_hp
    decision = str(state.get("decision") or "")

    if decision == "game_over" or decision in POST_COMBAT_DECISIONS:
        death = float(decision == "game_over" and not bool(state.get("victory")))
        value = _utility(
            hp_loss_fraction=actual_hp_loss_fraction,
            death_probability=death,
            potion_spent=actual_potion_spent,
            max_hp_delta=actual_max_hp_delta,
            objective=objective,
        )
        return SearchOutcome(
            value=value,
            death_probability=death,
            end_hp=current_hp,
            potion_spent=actual_potion_spent,
            max_hp_delta=actual_max_hp_delta,
            terminal=True,
            leaf_source="exact_engine_terminal",
            determinization_id=determinization_id,
            depth=depth,
            max_hp_delta_raw=actual_max_hp_delta,
            max_hp_growth_cap=0.0,
        )

    if decision != "combat_play" or node is None:
        value = _utility(
            hp_loss_fraction=actual_hp_loss_fraction,
            death_probability=0.0,
            potion_spent=actual_potion_spent,
            max_hp_delta=actual_max_hp_delta,
            objective=objective,
        ) - float(unsupported_penalty)
        return SearchOutcome(
            value=value,
            death_probability=0.0,
            end_hp=current_hp,
            potion_spent=actual_potion_spent,
            max_hp_delta=actual_max_hp_delta,
            terminal=False,
            leaf_source=f"unsupported_subdecision:{decision or 'unknown'}",
            determinization_id=determinization_id,
            depth=depth,
            max_hp_delta_raw=actual_max_hp_delta,
            max_hp_growth_cap=0.0,
        )

    if node.state_value is not None:
        weighted = dict(node.state_value)
        leaf_source = "independent_state_value_leaf"
    else:
        # Compatibility fallback for the value-v1 checkpoint. These
        # action-conditioned predictions are supervised only for the recorded
        # human action, so new checkpoints should prefer the independent V(s).
        weighted = {
            "hp_loss_fraction": 0.0,
            "death_probability": 0.0,
            "potion_spent": 0.0,
            "max_hp_delta": 0.0,
        }
        total_probability = 0.0
        for row in node.ranked:
            resource = row.get("resource_prediction")
            if not isinstance(resource, dict):
                raise EngineError("MCTS V0 requires a checkpoint with resource value heads")
            probability = float(row["policy_probability"])
            total_probability += probability
            for key in weighted:
                weighted[key] += probability * float(resource[key])
        if total_probability <= 0.0:
            raise EngineError("leaf policy probabilities sum to zero")
        for key in weighted:
            weighted[key] /= total_probability
        leaf_source = "provisional_policy_weighted_resource_leaf"

    predicted_hp_loss = weighted["hp_loss_fraction"] * max(current_max_hp, 1.0)
    total_hp_loss_fraction = (
        max(0.0, root_hp - current_hp) + predicted_hp_loss
    ) / max(root_max_hp, 1.0)
    death = max(0.0, min(1.0, weighted["death_probability"]))
    potion_spent = actual_potion_spent + weighted["potion_spent"]
    growth = ground_future_max_hp_delta(
        weighted["max_hp_delta"], node.sample["observation"]
    )
    grounded_future_max_hp_delta = float(growth["grounded_prediction"])
    max_hp_delta = actual_max_hp_delta + grounded_future_max_hp_delta
    value = _utility(
        hp_loss_fraction=total_hp_loss_fraction,
        death_probability=death,
        potion_spent=potion_spent,
        max_hp_delta=max_hp_delta,
        objective=objective,
    )
    return SearchOutcome(
        value=value,
        death_probability=death,
        end_hp=max(0.0, current_hp - predicted_hp_loss),
        potion_spent=potion_spent,
        max_hp_delta=max_hp_delta,
        terminal=False,
        leaf_source=leaf_source,
        determinization_id=determinization_id,
        depth=depth,
        max_hp_delta_raw=actual_max_hp_delta + weighted["max_hp_delta"],
        max_hp_growth_cap=float(growth["positive_growth_cap"]),
    )


def _select_edge(node: TreeNode, *, c_puct: float) -> str:
    parent_visits = sum(edge.visits for edge in node.edges.values())
    return max(
        node.edges,
        key=lambda action_id: (
            puct_score(
                mean_value=node.edges[action_id].mean_value,
                prior=node.edges[action_id].prior,
                parent_visits=parent_visits,
                child_visits=node.edges[action_id].visits,
                c_puct=c_puct,
            ),
            node.edges[action_id].prior,
            action_id,
        ),
    )


def _select_root_action(
    action_rows: list[dict[str, Any]], *, selection_rule: str
) -> dict[str, Any]:
    """Select a root action without trusting tiny-sample Q outliers.

    High-budget Spire Pilot-style search can rank sufficiently sampled actions
    by risk-adjusted value. Our current low-budget search instead defaults to
    visit count, allowing the human-pretrained prior and repeatedly confirmed
    search value to determine the result. Risk score remains the tie-breaker.
    """

    if selection_rule == "risk_adjusted_value":
        key = lambda row: (
            bool(row["selection_eligible"]),
            float(row["selection_score"]),
            int(row["visits"]),
            float(row["prior"]),
        )
    elif selection_rule == "visit_count_then_risk":
        key = lambda row: (
            bool(row["selection_eligible"]),
            int(row["visits"]),
            float(row["selection_score"]),
            float(row["prior"]),
        )
    else:
        raise EngineError(f"unsupported root selection rule: {selection_rule}")
    action_rows.sort(key=key, reverse=True)
    return next(row for row in action_rows if row["selection_eligible"])


def _policy_root_action(action_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        action_rows,
        key=lambda row: (
            float(row["prior"]),
            -int(row["candidate"]["candidate_index"]),
        ),
    )


def _paired_root_metrics(
    row: dict[str, Any],
    identities: set[str],
    *,
    root_config: dict[str, Any],
) -> dict[str, float]:
    values_by_world = row["determinization_values"]
    values = [float(values_by_world[identity]["mean_value"]) for identity in sorted(identities)]
    deaths = [
        float(values_by_world[identity]["death_probability"])
        for identity in sorted(identities)
    ]
    mean_value = statistics.fmean(values)
    tail = lower_tail_cvar(values, float(root_config["cvar_alpha"]))
    return {
        "mean_value": mean_value,
        "lower_tail_cvar": tail,
        "selection_score": risk_adjusted_root_score(
            mean_value=mean_value,
            lower_tail_value=tail,
            mean_weight=float(root_config["mean_value_weight"]),
            cvar_weight=float(root_config["cvar_weight"]),
        ),
        "death_probability": statistics.fmean(deaths),
    }


def _apply_policy_fallback(
    *,
    search_choice: dict[str, Any],
    policy_choice: dict[str, Any],
    root_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = root_config.get("policy_fallback") or {}
    report: dict[str, Any] = {
        "enabled": bool(gate.get("enabled", False)),
        "applied": False,
        "reason": "disabled",
        "search_candidate_id": search_choice["candidate"]["candidate_id"],
        "policy_candidate_id": policy_choice["candidate"]["candidate_id"],
    }
    if not report["enabled"]:
        return search_choice, report
    if search_choice["candidate"]["candidate_id"] == policy_choice["candidate"]["candidate_id"]:
        report["reason"] = "search_matches_policy"
        return search_choice, report

    search_worlds = set(search_choice["determinization_values"])
    policy_worlds = set(policy_choice["determinization_values"])
    shared_worlds = search_worlds & policy_worlds
    report["shared_determinizations"] = len(shared_worlds)
    minimum_worlds = int(gate["minimum_shared_determinizations"])
    if len(shared_worlds) < minimum_worlds:
        report.update(applied=True, reason="insufficient_shared_determinizations")
        return policy_choice, report

    search_metrics = _paired_root_metrics(
        search_choice, shared_worlds, root_config=root_config
    )
    policy_metrics = _paired_root_metrics(
        policy_choice, shared_worlds, root_config=root_config
    )
    report["search_paired_metrics"] = search_metrics
    report["policy_paired_metrics"] = policy_metrics
    if search_metrics["death_probability"] > float(gate["maximum_death_probability"]):
        report.update(applied=True, reason="search_death_probability_above_limit")
        return policy_choice, report
    if search_metrics["death_probability"] > (
        policy_metrics["death_probability"]
        + float(gate["maximum_death_probability_increase"])
    ):
        report.update(applied=True, reason="search_death_probability_worse_than_policy")
        return policy_choice, report
    advantage = search_metrics["selection_score"] - policy_metrics["selection_score"]
    report["paired_selection_score_advantage"] = advantage
    if advantage < float(gate["minimum_risk_adjusted_advantage"]):
        report.update(applied=True, reason="insufficient_risk_adjusted_advantage")
        return policy_choice, report
    report["reason"] = "search_evidence_accepted"
    return search_choice, report


def search_current_root(
    *,
    worker: EngineProcess,
    entrance_save: Path,
    enter_command: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    root_state: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    config: dict[str, Any],
    budget: int,
    max_depth: int,
    search_seed: int,
) -> dict[str, Any]:
    """Search one real combat decision and return a fully auditable root report."""

    puct_config = config["puct"]
    root_config = config["root_selection"]
    leaf_config = config["leaf_evaluation"]
    reuse_entity_encoding = bool(
        config.get("model_inference", {}).get("reuse_entity_encoding", True)
    )
    reuse_precomputed_sample = bool(
        config.get("model_inference", {}).get("reuse_precomputed_sample", True)
    )
    transition_cache_config = config.get("engine_transition_cache", {})
    transition_cache_enabled = bool(transition_cache_config.get("enabled", False))
    transition_cache_maximum_entries = int(
        transition_cache_config.get("maximum_entries", 10_000)
    )
    if transition_cache_maximum_entries < 1:
        raise ValueError("engine transition cache maximum_entries must be positive")
    root_node = _expand_search_node(
        root_state,
        sample=None,
        model=model,
        tensorizer=tensorizer,
        device=device,
        node_index=0,
        reuse_entity_encoding=reuse_entity_encoding,
    )
    tree: dict[str, TreeNode] = {root_node.key: root_node}
    root_action_order = sorted(
        root_node.edges,
        key=lambda action_id: (
            -root_node.edges[action_id].prior,
            root_node.candidates[action_id]["candidate_index"],
        ),
    )
    minimum_root_visits = int(puct_config["minimum_root_visits_per_legal_action"])
    effective_budget, forced_coverage_simulations = root_coverage_budget(
        int(budget),
        legal_action_count=len(root_action_order),
        minimum_visits_per_action=minimum_root_visits,
    )
    root_determinization_count = int(puct_config["root_determinization_count"])
    visible_draw = _visible_draw_multiset(root_state)
    simulation_ms: list[float] = []
    restore_ms: list[float] = []
    engine_action_ms: list[float] = []
    engine_action_ms_by_type: dict[str, list[float]] = {}
    engine_actions = 0
    engine_action_ipc_count = 0
    replay_actions = 0
    batched_suffix_replay_actions = 0
    transition_cache_hits = 0
    transition_cache_misses = 0
    transition_cache_full_path_replays = 0
    transition_cache_skipped_insertions = 0
    transition_state_cache: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    max_depth_reached = 0
    inference_ms = root_node.inference_ms
    restore_mode = str(config.get("engine_restore", {}).get("mode", "legacy"))
    cache_setup_ms = 0.0
    if restore_mode in {
        "cached_batch",
        "cached_batch_compact",
        "cached_batch_auto",
        "cached_batch_auto_prepared",
    }:
        cache_result, cache_setup_ms = worker.send({
            "cmd": "cache_save",
            "name": _cache_key(entrance_save),
            "path": str(entrance_save),
        })
        if cache_result.get("type") != "ok":
            raise EngineError(f"failed to cache entrance save: {cache_result!r}")
    started_search = time.perf_counter()

    for simulation in range(effective_budget):
        simulation_started = time.perf_counter()
        world_index = paired_root_determinization_index(
            simulation,
            legal_action_count=len(root_action_order),
            forced_coverage_simulations=forced_coverage_simulations,
            determinization_count=root_determinization_count,
        )
        order, determinization_id = _determinization(
            visible_draw, search_seed=search_seed, simulation=world_index
        )
        state = root_state
        current_restore_ms = 0.0
        engine_ready = False
        selected_action_ids: list[str] = []
        selected_commands: list[dict[str, Any]] = []
        if not transition_cache_enabled:
            state, current_restore_ms = _restore_search_root(
                worker=worker,
                entrance_save=entrance_save,
                enter_command=enter_command,
                root_prefix=root_prefix,
                draw_order=order,
                mode=restore_mode,
            )
            replay_actions += len(root_prefix)
            engine_ready = True
            if state != root_state:
                raise EngineError(
                    "combat prefix replay did not reproduce the current real root"
                )
        path: list[SearchEdgeStats] = []
        depth = 0
        leaf_node: TreeNode | None = None
        while state.get("decision") in SEARCH_DECISIONS and depth < max_depth:
            decision = str(state.get("decision"))
            sample = headless_state_to_model_sample(
                state,
                transition_id=f"mcts:sim:{simulation}:depth:{depth}",
                combat_id="mcts",
            ) if decision == "combat_play" else None
            key = _search_information_key(state, sample)
            node = tree.get(key)
            if node is None:
                node = _expand_search_node(
                    state,
                    sample=sample if reuse_precomputed_sample else None,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    node_index=len(tree),
                    reuse_entity_encoding=reuse_entity_encoding,
                )
                tree[key] = node
                inference_ms += node.inference_ms
                if decision == "combat_play":
                    leaf_node = node
                    break

            current_candidates = (
                {row["candidate_id"]: row for row in sample["candidates"]}
                if sample is not None
                else node.candidates
            )
            if set(current_candidates) != set(node.edges):
                raise EngineError("same information key produced a different legal action set")
            forced_index = forced_root_action_index(
                simulation,
                legal_action_count=len(root_action_order),
                forced_coverage_simulations=forced_coverage_simulations,
            ) if depth == 0 else None
            if forced_index is not None:
                action_id = root_action_order[forced_index]
            else:
                action_id = _select_edge(node, c_puct=float(puct_config["c_puct"]))
            edge = node.edges[action_id]
            path.append(edge)
            action_command = _candidate_command(current_candidates[action_id])
            engine_actions += 1
            next_action_ids = (*selected_action_ids, action_id)
            cache_key = (world_index, next_action_ids)
            cached_state = (
                transition_state_cache.get(cache_key)
                if transition_cache_enabled and not engine_ready
                else None
            )
            if cached_state is not None:
                transition_cache_hits += 1
                selected_action_ids.append(action_id)
                selected_commands.append(action_command)
                state = cached_state
                depth += 1
                continue

            if transition_cache_enabled:
                transition_cache_misses += 1
            if not engine_ready:
                restored_state, current_restore_ms = _restore_search_root(
                    worker=worker,
                    entrance_save=entrance_save,
                    enter_command=enter_command,
                    root_prefix=root_prefix,
                    draw_order=order,
                    mode=restore_mode,
                    suffix_commands=selected_commands,
                )
                replay_actions += len(root_prefix)
                batched_suffix_replay_actions += len(selected_commands)
                if restored_state != state:
                    raise EngineError(json.dumps({
                        "error": "cached transition prefix did not match compact engine replay",
                        "world_index": world_index,
                        "determinization_id": determinization_id,
                        "action_ids": selected_action_ids,
                        "differences": _state_difference_paths(state, restored_state),
                        "expected": _state_debug_signature(state),
                        "actual": _state_debug_signature(restored_state),
                    }, ensure_ascii=False))
                engine_ready = True

            state, current_action_ms = worker.send(action_command)
            engine_action_ipc_count += 1
            engine_action_ms.append(current_action_ms)
            action_type = str(action_command.get("action") or "unknown")
            engine_action_ms_by_type.setdefault(action_type, []).append(current_action_ms)
            if transition_cache_enabled:
                if len(transition_state_cache) < transition_cache_maximum_entries:
                    transition_state_cache[cache_key] = state
                else:
                    transition_cache_skipped_insertions += 1
            selected_action_ids.append(action_id)
            selected_commands.append(action_command)
            depth += 1

        max_depth_reached = max(max_depth_reached, depth)
        if transition_cache_enabled and not engine_ready:
            restored_state, current_restore_ms = _restore_search_root(
                worker=worker,
                entrance_save=entrance_save,
                enter_command=enter_command,
                root_prefix=root_prefix,
                draw_order=order,
                mode=restore_mode,
                suffix_commands=selected_commands,
            )
            replay_actions += len(root_prefix)
            batched_suffix_replay_actions += len(selected_commands)
            if restored_state != state:
                raise EngineError(json.dumps({
                    "error": "fully cached transition path did not match compact engine replay",
                    "world_index": world_index,
                    "determinization_id": determinization_id,
                    "action_ids": selected_action_ids,
                    "differences": _state_difference_paths(state, restored_state),
                    "expected": _state_debug_signature(state),
                    "actual": _state_debug_signature(restored_state),
                }, ensure_ascii=False))
            state = restored_state
            engine_ready = True
            transition_cache_full_path_replays += 1
        if state.get("decision") == "combat_play" and leaf_node is None:
            sample = headless_state_to_model_sample(
                state,
                transition_id=f"mcts:sim:{simulation}:leaf",
                combat_id="mcts",
            )
            leaf_node = tree.get(_information_key(sample))
            if leaf_node is None:
                leaf_node = _expand_search_node(
                    state,
                    sample=sample if reuse_precomputed_sample else None,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    node_index=len(tree),
                    reuse_entity_encoding=reuse_entity_encoding,
                )
                tree[leaf_node.key] = leaf_node
                inference_ms += leaf_node.inference_ms
        outcome = _leaf_outcome(
            state,
            root_state=root_state,
            node=leaf_node,
            objective=objective,
            determinization_id=determinization_id,
            unsupported_penalty=float(leaf_config.get("unsupported_subdecision_penalty", 1.0)),
            depth=depth,
        )
        for edge in path:
            edge.update(outcome)
        restore_ms.append(current_restore_ms)
        simulation_ms.append((time.perf_counter() - simulation_started) * 1000.0)

    action_rows = []
    alpha = float(root_config["cvar_alpha"])
    for action_id, edge in root_node.edges.items():
        stats = edge.summary(cvar_alpha=alpha)
        selection_score = risk_adjusted_root_score(
            mean_value=float(stats["determinization_mean_value"]),
            lower_tail_value=float(stats["determinization_lower_tail_cvar"]),
            mean_weight=float(root_config["mean_value_weight"]),
            cvar_weight=float(root_config["cvar_weight"]),
        )
        action_rows.append({
            "candidate": root_node.candidates[action_id],
            **stats,
            "selection_score": selection_score,
        })
    visit_threshold = root_visit_threshold(
        [int(row["visits"]) for row in action_rows],
        minimum_visits=int(root_config["minimum_visits"]),
        minimum_fraction_of_max=float(root_config["minimum_fraction_of_max_visits"]),
    )
    for row in action_rows:
        row["selection_eligible"] = int(row["visits"]) >= visit_threshold
    selection_rule = str(
        root_config.get("selection_rule", "risk_adjusted_value")
    )
    search_choice = _select_root_action(action_rows, selection_rule=selection_rule)
    policy_choice = _policy_root_action(action_rows)
    chosen, policy_fallback = _apply_policy_fallback(
        search_choice=search_choice,
        policy_choice=policy_choice,
        root_config=root_config,
    )
    total_visits = sum(int(row["visits"]) for row in action_rows)
    for row in action_rows:
        row["visit_policy_probability"] = (
            int(row["visits"]) / total_visits if total_visits else 0.0
        )
    elapsed_ms = (time.perf_counter() - started_search) * 1000.0
    report = {
        "requested_budget": int(budget),
        "effective_budget": effective_budget,
        "minimum_root_visits_per_legal_action": minimum_root_visits,
        "root_determinization_count": root_determinization_count,
        "forced_root_coverage_simulations": forced_coverage_simulations,
        "tree_node_count": len(tree),
        "engine_action_count": engine_actions,
        "engine_action_ipc_count": engine_action_ipc_count,
        "prefix_replay_action_count": replay_actions,
        "batched_suffix_replay_action_count": batched_suffix_replay_actions,
        "root_prefix_length": len(root_prefix),
        "max_depth_reached": max_depth_reached,
        "search_wall_ms": round(elapsed_ms, 3),
        "simulations_per_second": round(1000.0 * effective_budget / elapsed_ms, 3),
        "engine_restore_mode": restore_mode,
        "cache_setup_ms": round(cache_setup_ms, 3),
        "root_restore_latency_ms": {
            "mean": round(statistics.fmean(restore_ms), 3),
            "p50": round(statistics.median(restore_ms), 3),
            "max": round(max(restore_ms), 3),
        },
        "simulation_latency_ms": {
            "mean": round(statistics.fmean(simulation_ms), 3),
            "p50": round(statistics.median(simulation_ms), 3),
            "max": round(max(simulation_ms), 3),
        },
        "model_inference_total_ms": round(inference_ms, 3),
        "reuse_entity_encoding": reuse_entity_encoding,
        "reuse_precomputed_sample": reuse_precomputed_sample,
        "engine_transition_cache": {
            "enabled": transition_cache_enabled,
            "maximum_entries": transition_cache_maximum_entries,
            "entry_count": len(transition_state_cache),
            "hit_count": transition_cache_hits,
            "miss_count": transition_cache_misses,
            "full_path_replay_simulation_count": transition_cache_full_path_replays,
            "skipped_insertion_count": transition_cache_skipped_insertions,
        },
        "engine_action_latency_ms": _latency_summary(engine_action_ms),
        "engine_action_latency_by_type_ms": {
            action_type: _latency_summary(values)
            for action_type, values in sorted(engine_action_ms_by_type.items())
        },
        "root_selection_visit_threshold": visit_threshold,
        "root_selection_rule": selection_rule,
        "search_selected_candidate": search_choice["candidate"],
        "policy_candidate": policy_choice["candidate"],
        "policy_fallback": policy_fallback,
        "chosen_candidate": chosen["candidate"],
        "chosen_selection_score": chosen["selection_score"],
        "actions": action_rows,
    }
    report["teacher_record"] = build_search_teacher_record(
        root_node.sample,
        report,
        search_seed=search_seed,
        model_version=str(getattr(model, "checkpoint_model_version", "unknown")),
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config.resolve())
    if getattr(args, "restore_mode", None) is not None:
        config.setdefault("engine_restore", {})["mode"] = args.restore_mode
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if model.resource_value_head is None:
        raise EngineError("MCTS V0 requires the combat_policy_value_v1 checkpoint")
    objective = CombatObjective.from_config(model.config)
    puct_config = config["puct"]
    root_config = config["root_selection"]
    leaf_config = config["leaf_evaluation"]
    reuse_entity_encoding = bool(
        config.get("model_inference", {}).get("reuse_entity_encoding", True)
    )
    reuse_precomputed_sample = bool(
        config.get("model_inference", {}).get("reuse_precomputed_sample", True)
    )
    transition_cache_enabled = bool(
        config.get("engine_transition_cache", {}).get("enabled", False)
    )
    budget = int(args.budget)
    max_depth = int(args.max_depth or puct_config["maximum_player_decision_depth"])
    game_data_dir = _game_data_dir(args.game_dir)

    with tempfile.TemporaryDirectory(prefix="sts2_mcts_v0_") as temp_dir:
        entrance_save = Path(temp_dir) / "entrance.save"
        with _engine(args, game_data_dir) as source:
            map_state, _ = source.send({
                "cmd": "start_run",
                "character": "Ironclad",
                "ascension": args.ascension,
                "seed": args.seed,
                "lang": "en",
            })
            map_state, _ = _advance_initial_event(source, map_state)
            choice = _monster_choice(map_state)
            save_result, _ = source.send({"cmd": "write_continue_save", "path": str(entrance_save)})
            if not save_result.get("success"):
                raise EngineError(f"failed to write entrance save: {save_result!r}")
            root_state, _ = source.send(_enter_command(choice))
            if root_state.get("decision") != "combat_play":
                raise EngineError(f"Monster node did not enter combat: {root_state!r}")

        if transition_cache_enabled:
            with _engine(args, game_data_dir) as worker:
                warm_state, _ = worker.send({
                    "cmd": "load_save",
                    "path": str(entrance_save),
                    "lang": "en",
                })
                if warm_state.get("decision") != "map_select":
                    raise EngineError(f"worker warmup restore failed: {warm_state!r}")
                restored_root_state, _ = worker.send(_enter_command(choice))
                if restored_root_state.get("decision") != "combat_play":
                    raise EngineError(
                        f"worker did not restore the combat root: {restored_root_state!r}"
                    )
                # The reload path is authoritative for search.  In particular,
                # continue-save floor counters can differ from the source
                # process that wrote the save before selecting the map node.
                root_state = restored_root_state
                result = search_current_root(
                    worker=worker,
                    entrance_save=entrance_save,
                    enter_command=_enter_command(choice),
                    root_prefix=[],
                    root_state=root_state,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    objective=objective,
                    config=config,
                    budget=budget,
                    max_depth=max_depth,
                    search_seed=args.search_seed,
                )
            return {
                **result,
                "schema_version": "policy-guided-mcts-0.3.0",
                "search_version": SEARCH_VERSION,
                "generated_at": utc_now(),
                "status": "pass",
                "seed": args.seed,
                "search_seed": args.search_seed,
                "ascension": args.ascension,
                "enemy_ids": [
                    row.get("id") for row in root_state.get("enemies") or []
                ],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "device": device,
                "information_boundary": config["information_boundary"],
                "worker_count": 1,
                "leaf_evaluation": {
                    **leaf_config,
                    "active_nonterminal_source": (
                        "provisional_policy_weighted_resource_leaf"
                    ),
                    "immediate_resource_source": "exact_engine_root_to_leaf_delta",
                },
                "root_selection": root_config,
                "known_limitations": [
                    "nonterminal leaves use point estimates from an action-conditioned head, not a state value distribution",
                    "combat card-selection subdecisions are not yet searched and receive an explicit penalty",
                    "only the currently visible draw-pile multiset is determinized; later reshuffle RNG remains engine-controlled",
                    "this gate uses one persistent worker",
                ],
            }

        root_node = _expand_search_node(
            root_state,
            sample=None,
            model=model,
            tensorizer=tensorizer,
            device=device,
            node_index=0,
            reuse_entity_encoding=reuse_entity_encoding,
        )
        tree: dict[str, TreeNode] = {root_node.key: root_node}
        root_action_order = sorted(
            root_node.edges,
            key=lambda action_id: (
                -root_node.edges[action_id].prior,
                root_node.candidates[action_id]["candidate_index"],
            ),
        )
        minimum_root_visits = int(puct_config["minimum_root_visits_per_legal_action"])
        effective_budget, forced_coverage_simulations = root_coverage_budget(
            budget,
            legal_action_count=len(root_action_order),
            minimum_visits_per_action=minimum_root_visits,
        )
        root_determinization_count = int(puct_config["root_determinization_count"])
        visible_draw = _visible_draw_multiset(root_state)
        simulation_ms: list[float] = []
        restore_ms: list[float] = []
        engine_action_ms: list[float] = []
        engine_action_ms_by_type: dict[str, list[float]] = {}
        engine_actions = 0
        max_depth_reached = 0
        inference_ms = root_node.inference_ms
        started_search = time.perf_counter()

        with _engine(args, game_data_dir) as worker:
            warm_state, _ = worker.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
            if warm_state.get("decision") != "map_select":
                raise EngineError(f"worker warmup restore failed: {warm_state!r}")
            restore_mode = str(config.get("engine_restore", {}).get("mode", "legacy"))
            cache_setup_ms = 0.0
            if restore_mode in {
                "cached_batch",
                "cached_batch_compact",
                "cached_batch_auto",
                "cached_batch_auto_prepared",
            }:
                cache_result, cache_setup_ms = worker.send({
                    "cmd": "cache_save",
                    "name": _cache_key(entrance_save),
                    "path": str(entrance_save),
                })
                if cache_result.get("type") != "ok":
                    raise EngineError(f"failed to cache entrance save: {cache_result!r}")

            for simulation in range(effective_budget):
                simulation_started = time.perf_counter()
                world_index = paired_root_determinization_index(
                    simulation,
                    legal_action_count=len(root_action_order),
                    forced_coverage_simulations=forced_coverage_simulations,
                    determinization_count=root_determinization_count,
                )
                order, determinization_id = _determinization(
                    visible_draw, search_seed=args.search_seed, simulation=world_index
                )
                state, current_restore_ms = _restore_search_root(
                    worker=worker,
                    entrance_save=entrance_save,
                    enter_command=_enter_command(choice),
                    root_prefix=[],
                    draw_order=order,
                    mode=restore_mode,
                )
                restore_ms.append(current_restore_ms)
                if state != root_state:
                    raise EngineError("restored combat root did not match source root")

                path: list[SearchEdgeStats] = []
                depth = 0
                leaf_node: TreeNode | None = None
                while state.get("decision") in SEARCH_DECISIONS and depth < max_depth:
                    decision = str(state.get("decision"))
                    sample = headless_state_to_model_sample(
                        state,
                        transition_id=f"mcts:sim:{simulation}:depth:{depth}",
                        combat_id="mcts",
                    ) if decision == "combat_play" else None
                    key = _search_information_key(state, sample)
                    node = tree.get(key)
                    if node is None:
                        node = _expand_search_node(
                            state,
                            sample=sample if reuse_precomputed_sample else None,
                            model=model,
                            tensorizer=tensorizer,
                            device=device,
                            node_index=len(tree),
                            reuse_entity_encoding=reuse_entity_encoding,
                        )
                        tree[key] = node
                        inference_ms += node.inference_ms
                        if decision == "combat_play":
                            leaf_node = node
                            break

                    current_candidates = (
                        {row["candidate_id"]: row for row in sample["candidates"]}
                        if sample is not None
                        else node.candidates
                    )
                    if set(current_candidates) != set(node.edges):
                        raise EngineError("same information key produced a different legal action set")
                    forced_index = forced_root_action_index(
                        simulation,
                        legal_action_count=len(root_action_order),
                        forced_coverage_simulations=forced_coverage_simulations,
                    ) if depth == 0 else None
                    if forced_index is not None:
                        action_id = root_action_order[forced_index]
                    else:
                        action_id = _select_edge(node, c_puct=float(puct_config["c_puct"]))
                    edge = node.edges[action_id]
                    path.append(edge)
                    action_command = _candidate_command(current_candidates[action_id])
                    state, current_action_ms = worker.send(action_command)
                    engine_action_ms.append(current_action_ms)
                    action_type = str(action_command.get("action") or "unknown")
                    engine_action_ms_by_type.setdefault(action_type, []).append(
                        current_action_ms
                    )
                    engine_actions += 1
                    depth += 1

                max_depth_reached = max(max_depth_reached, depth)
                if state.get("decision") == "combat_play" and leaf_node is None:
                    sample = headless_state_to_model_sample(
                        state,
                        transition_id=f"mcts:sim:{simulation}:leaf",
                        combat_id="mcts",
                    )
                    leaf_node = tree.get(_information_key(sample))
                    if leaf_node is None:
                        leaf_node = _expand_search_node(
                            state,
                            sample=sample if reuse_precomputed_sample else None,
                            model=model,
                            tensorizer=tensorizer,
                            device=device,
                            node_index=len(tree),
                            reuse_entity_encoding=reuse_entity_encoding,
                        )
                        tree[leaf_node.key] = leaf_node
                        inference_ms += leaf_node.inference_ms
                outcome = _leaf_outcome(
                    state,
                    root_state=root_state,
                    node=leaf_node,
                    objective=objective,
                    determinization_id=determinization_id,
                    unsupported_penalty=float(leaf_config.get("unsupported_subdecision_penalty", 1.0)),
                    depth=depth,
                )
                for edge in path:
                    edge.update(outcome)
                simulation_ms.append((time.perf_counter() - simulation_started) * 1000.0)

        action_rows = []
        alpha = float(root_config["cvar_alpha"])
        for action_id, edge in root_node.edges.items():
            stats = edge.summary(cvar_alpha=alpha)
            selection_score = risk_adjusted_root_score(
                mean_value=float(stats["determinization_mean_value"]),
                lower_tail_value=float(stats["determinization_lower_tail_cvar"]),
                mean_weight=float(root_config["mean_value_weight"]),
                cvar_weight=float(root_config["cvar_weight"]),
            )
            action_rows.append({
                "candidate": root_node.candidates[action_id],
                **stats,
                "selection_score": selection_score,
            })
        visit_threshold = root_visit_threshold(
            [int(row["visits"]) for row in action_rows],
            minimum_visits=int(root_config["minimum_visits"]),
            minimum_fraction_of_max=float(root_config["minimum_fraction_of_max_visits"]),
        )
        for row in action_rows:
            row["selection_eligible"] = int(row["visits"]) >= visit_threshold
        selection_rule = str(
            root_config.get("selection_rule", "risk_adjusted_value")
        )
        search_choice = _select_root_action(action_rows, selection_rule=selection_rule)
        policy_choice = _policy_root_action(action_rows)
        chosen, policy_fallback = _apply_policy_fallback(
            search_choice=search_choice,
            policy_choice=policy_choice,
            root_config=root_config,
        )
        total_visits = sum(int(row["visits"]) for row in action_rows)
        for row in action_rows:
            row["visit_policy_probability"] = (
                int(row["visits"]) / total_visits if total_visits else 0.0
            )

        elapsed_ms = (time.perf_counter() - started_search) * 1000.0
        return {
            "schema_version": "policy-guided-mcts-0.2.0",
            "search_version": SEARCH_VERSION,
            "generated_at": utc_now(),
            "status": "pass",
            "seed": args.seed,
            "search_seed": args.search_seed,
            "ascension": args.ascension,
            "enemy_ids": [row.get("id") for row in root_state.get("enemies") or []],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device": device,
            "requested_budget": budget,
            "effective_budget": effective_budget,
            "minimum_root_visits_per_legal_action": minimum_root_visits,
            "root_determinization_count": root_determinization_count,
            "forced_root_coverage_simulations": forced_coverage_simulations,
            "tree_node_count": len(tree),
            "engine_action_count": engine_actions,
            "max_depth_reached": max_depth_reached,
            "search_wall_ms": round(elapsed_ms, 3),
            "simulations_per_second": round(1000.0 * effective_budget / elapsed_ms, 3),
            "engine_restore_mode": restore_mode,
            "cache_setup_ms": round(cache_setup_ms, 3),
            "root_restore_latency_ms": {
                "mean": round(statistics.fmean(restore_ms), 3),
                "p50": round(statistics.median(restore_ms), 3),
                "max": round(max(restore_ms), 3),
            },
            "simulation_latency_ms": {
                "mean": round(statistics.fmean(simulation_ms), 3),
                "p50": round(statistics.median(simulation_ms), 3),
                "max": round(max(simulation_ms), 3),
            },
            "model_inference_total_ms": round(inference_ms, 3),
            "reuse_entity_encoding": reuse_entity_encoding,
            "reuse_precomputed_sample": reuse_precomputed_sample,
            "engine_action_latency_ms": _latency_summary(engine_action_ms),
            "engine_action_latency_by_type_ms": {
                action_type: _latency_summary(values)
                for action_type, values in sorted(engine_action_ms_by_type.items())
            },
            "information_boundary": config["information_boundary"],
            "worker_count": 1,
            "leaf_evaluation": {
                **leaf_config,
                "active_nonterminal_source": "provisional_policy_weighted_resource_leaf",
                "immediate_resource_source": "exact_engine_root_to_leaf_delta",
            },
            "root_selection": root_config,
            "root_selection_visit_threshold": visit_threshold,
            "root_selection_rule": selection_rule,
            "search_selected_candidate": search_choice["candidate"],
            "policy_candidate": policy_choice["candidate"],
            "policy_fallback": policy_fallback,
            "known_limitations": [
                "nonterminal leaves use point estimates from an action-conditioned head, not a state value distribution",
                "combat card-selection subdecisions are not yet searched and receive an explicit penalty",
                "only the currently visible draw-pile multiset is determinized; later reshuffle RNG remains engine-controlled",
                "this gate uses one persistent worker",
            ],
            "chosen_candidate": chosen["candidate"],
            "chosen_selection_score": chosen["selection_score"],
            "actions": action_rows,
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
    parser.add_argument("--seed", default="policy-guided-mcts-v0")
    parser.add_argument("--search-seed", type=int, default=20260815)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-depth", type=int)
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
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        key: report[key]
        for key in (
            "status", "seed", "enemy_ids", "requested_budget", "effective_budget",
            "tree_node_count", "engine_action_count", "search_wall_ms",
            "simulations_per_second", "chosen_candidate",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
