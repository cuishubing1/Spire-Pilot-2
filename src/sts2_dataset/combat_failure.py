from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from .util import load_json, sha256_file, utc_now, write_json_atomic


FAILURE_RATCHET_SCHEMA_VERSION = "combat-failure-ratchet-0.1.0"
NUMERIC_FEATURES = (
    "round",
    "hp_ratio",
    "block",
    "energy",
    "hand_size",
    "enemy_count",
    "enemy_hp_ratio",
    "incoming_damage",
    "legal_count",
)


def encounter_pool(row: dict[str, Any]) -> str:
    room_type = str(row.get("room_type") or "").lower()
    if room_type == "elite":
        return "elite"
    if room_type == "boss":
        return "boss"
    if room_type == "monster":
        return "weak" if str(row.get("encounter") or "").endswith("_WEAK") else "strong"
    return "unknown"


def classify_failure(
    row: dict[str, Any],
    *,
    high_regret_threshold: float = 20.0,
    search_regression_threshold: float = 15.0,
) -> list[str]:
    if "p2_policy" not in row or "p2_one_step" not in row:
        return []
    policy = row["p2_policy"]
    one_step = row["p2_one_step"]
    human = row["human"]
    flags: list[str] = []
    if policy.get("status") == "death":
        flags.append("policy_death")
    if one_step.get("status") == "death":
        flags.append("one_step_death")
    if policy.get("status") != "death" and one_step.get("status") == "death":
        flags.append("search_introduced_death")
    if float(one_step["hp_loss"]) - float(human["hp_loss"]) >= high_regret_threshold:
        flags.append("high_human_regret")
    if float(one_step["hp_loss"]) - float(policy["hp_loss"]) >= search_regression_threshold:
        flags.append("search_regression")
    return flags


def _incoming_damage(enemies: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for enemy in enemies:
        intents = enemy.get("intent") or enemy.get("intents") or []
        for intent in intents:
            is_attack = bool(intent.get("is_attack")) or str(intent.get("type") or "").lower() == "attack"
            if not is_attack:
                continue
            damage = float(intent.get("damage") or 0.0)
            hits = float(intent.get("hits") or intent.get("repeats") or 1.0)
            total += float(intent.get("total_damage") or damage * hits)
    return total


def state_features(
    state: dict[str, Any],
    *,
    max_hp: float | None = None,
    legal_count: float | None = None,
) -> dict[str, float]:
    global_state = state.get("global") or {}
    enemies = state.get("enemies") or []
    hp = float(state.get("player_hp", global_state.get("hp", 0.0)) or 0.0)
    observed_max_hp = float(max_hp or global_state.get("max_hp", 0.0) or max(hp, 1.0))
    enemy_hp = sum(float(enemy.get("hp") or 0.0) for enemy in enemies)
    enemy_max_hp = sum(float(enemy.get("max_hp") or 0.0) for enemy in enemies)
    return {
        "round": float(state.get("round", global_state.get("round", 1.0)) or 1.0),
        "hp_ratio": hp / max(observed_max_hp, 1.0),
        "block": float(state.get("player_block", global_state.get("block", 0.0)) or 0.0),
        "energy": float(state.get("energy", global_state.get("energy", 0.0)) or 0.0),
        "hand_size": float(len(state.get("hand") or [])),
        "enemy_count": float(len(enemies)),
        "enemy_hp_ratio": enemy_hp / max(enemy_max_hp, 1.0),
        "incoming_damage": _incoming_damage(enemies),
        "legal_count": float(legal_count) if legal_count is not None else math.nan,
    }


def _action_key(candidate: dict[str, Any]) -> tuple[str, str | None]:
    return str(candidate.get("action_type") or ""), candidate.get("source_id")


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": candidate.get("action_type"),
        "source_id": candidate.get("source_id"),
        "target_id": candidate.get("target_id"),
        "target_kind": candidate.get("target_kind"),
        "search_category": candidate.get("search_category"),
        "search_equivalence_key": candidate.get("search_equivalence_key"),
    }


def first_search_divergence(row: dict[str, Any]) -> dict[str, Any] | None:
    policy_steps = {
        json.dumps(step.get("before"), sort_keys=True, ensure_ascii=False): step
        for step in row.get("p2_policy", {}).get("steps", [])
        if step.get("before", {}).get("decision") == "combat_play"
    }
    for search_step in row.get("p2_one_step", {}).get("steps", []):
        before = search_step.get("before") or {}
        if before.get("decision") != "combat_play":
            continue
        policy_step = policy_steps.get(json.dumps(before, sort_keys=True, ensure_ascii=False))
        if policy_step is None:
            continue
        policy_candidate = policy_step.get("chosen_candidate") or {}
        search_candidate = search_step.get("chosen_candidate") or {}
        if policy_candidate.get("search_equivalence_key") == search_candidate.get("search_equivalence_key"):
            continue
        lookahead = search_step.get("lookahead") or {}
        evaluations = lookahead.get("evaluations") or []

        def evaluation_for(candidate: dict[str, Any]) -> dict[str, Any] | None:
            key = candidate.get("search_equivalence_key")
            return next(
                (
                    evaluation
                    for evaluation in evaluations
                    if (evaluation.get("candidate") or {}).get("search_equivalence_key") == key
                ),
                None,
            )

        policy_evaluation = evaluation_for(policy_candidate)
        search_evaluation = evaluation_for(search_candidate)
        score_margin = None
        if policy_evaluation is not None and search_evaluation is not None:
            score_margin = float(search_evaluation.get("selection_score") or 0.0) - float(
                policy_evaluation.get("selection_score") or 0.0
            )
        return {
            "step": search_step.get("step"),
            "round": before.get("round"),
            "player_hp": before.get("player_hp"),
            "hand": before.get("hand"),
            "policy_entropy": lookahead.get("root_policy_entropy"),
            "policy_action": _candidate_summary(policy_candidate),
            "search_action": _candidate_summary(search_candidate),
            "policy_probability": policy_step.get("policy_probability"),
            "search_policy_probability": (
                search_evaluation.get("policy_probability") if search_evaluation else None
            ),
            "selection_score_margin": round(score_margin, 6) if score_margin is not None else None,
            "policy_selection_score": (
                policy_evaluation.get("selection_score") if policy_evaluation else None
            ),
            "search_selection_score": (
                search_evaluation.get("selection_score") if search_evaluation else None
            ),
        }
    return None


def _quantiles(rows: list[dict[str, float]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for feature in NUMERIC_FEATURES:
        values = [row[feature] for row in rows if math.isfinite(row[feature])]
        if values:
            result[feature] = [
                round(float(value), 6)
                for value in np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
            ]
    return result


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return round(sum(values) / len(values), 6) if values else None


def _normalize_card_id(value: str) -> str:
    return value if value.startswith("CARD.") else f"CARD.{value}"


def _load_training_reference(samples_path: Path) -> dict[str, Any]:
    table = pq.read_table(
        samples_path,
        columns=[
            "split",
            "combat_id",
            "encounter_signature",
            "act",
            "observation_v0_json",
            "candidates_json",
            "candidate_count",
            "label_index",
        ],
    )
    encounter_combats: dict[str, set[str]] = defaultdict(set)
    encounter_transitions: Counter[str] = Counter()
    card_state_count: Counter[str] = Counter()
    action_label_count: Counter[tuple[str, str | None]] = Counter()
    profiles: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    encounter_profiles: dict[str, list[dict[str, float]]] = defaultdict(list)
    state_count = 0
    for row in table.to_pylist():
        if row["split"] != "train":
            continue
        state_count += 1
        observation = json.loads(row["observation_v0_json"])
        room_type = str(observation.get("global", {}).get("room_type") or "Monster")
        features = state_features(observation, legal_count=float(row["candidate_count"]))
        profiles[(int(row["act"]), room_type)].append(features)
        signature = str(row["encounter_signature"])
        encounter_profiles[signature].append(features)
        encounter_combats[signature].add(str(row["combat_id"]))
        encounter_transitions[signature] += 1
        for card in observation.get("hand") or []:
            card_state_count[str(card.get("id"))] += 1
        candidates = json.loads(row["candidates_json"])
        candidate = candidates[int(row["label_index"])]
        action_label_count[_action_key(candidate)] += 1
    return {
        "state_count": state_count,
        "encounter_combats": encounter_combats,
        "encounter_transitions": encounter_transitions,
        "card_state_count": card_state_count,
        "action_label_count": action_label_count,
        "profile_quantiles": {key: _quantiles(values) for key, values in profiles.items()},
        "encounter_profile_quantiles": {
            key: _quantiles(values) for key, values in encounter_profiles.items()
        },
    }


def _trajectory_states(row: dict[str, Any]) -> list[dict[str, Any]]:
    steps = row["p2_one_step"].get("steps") or []
    result: list[dict[str, Any]] = []
    combat_steps = [step for step in steps if step.get("before", {}).get("decision") == "combat_play"]
    for index, step in enumerate(combat_steps):
        lookahead = step.get("lookahead") or {}
        features = state_features(
            step["before"],
            max_hp=float(row["snapshot"]["max_hp"]),
            legal_count=lookahead.get("raw_candidate_count"),
        )
        features["progress"] = index / max(len(combat_steps) - 1, 1)
        features["chosen_candidate"] = step.get("chosen_candidate") or {}
        features["policy_entropy"] = lookahead.get("root_policy_entropy")
        policy_candidate = lookahead.get("policy_candidate") or {}
        features["search_changed_policy"] = bool(
            lookahead
            and policy_candidate.get("search_equivalence_key")
            != (step.get("chosen_candidate") or {}).get("search_equivalence_key")
        )
        result.append(features)
    return result


def _trajectory_distribution_summary(
    rows: list[dict[str, Any]],
    training: dict[str, Any],
) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    action_label_counts: list[float] = []
    search_changes = 0
    for row in rows:
        key = (int(row["act"]), str(row["room_type"]))
        quantiles = training["profile_quantiles"].get(key, {})
        for state in _trajectory_states(row):
            finite = [feature for feature in NUMERIC_FEATURES if math.isfinite(state[feature])]
            tail_count = sum(
                state[feature] < quantiles[feature][1]
                or state[feature] > quantiles[feature][3]
                for feature in finite
                if feature in quantiles
            )
            comparable_count = sum(feature in quantiles for feature in finite)
            state["tail_feature_rate"] = tail_count / max(comparable_count, 1)
            candidate = state["chosen_candidate"]
            action_label_counts.append(
                float(training["action_label_count"][_action_key(candidate)])
            )
            search_changes += int(state["search_changed_policy"])
            states.append(state)

    def phase_summary(phase_states: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "states": len(phase_states),
            "mean_tail_feature_rate": _mean(
                state["tail_feature_rate"] for state in phase_states
            ),
            "mean_hp_ratio": _mean(state["hp_ratio"] for state in phase_states),
            "mean_round": _mean(state["round"] for state in phase_states),
            "mean_incoming_damage": _mean(
                state["incoming_damage"] for state in phase_states
            ),
            "mean_legal_count": _mean(state["legal_count"] for state in phase_states),
        }

    early = [state for state in states if state["progress"] <= 1.0 / 3.0]
    late = [state for state in states if state["progress"] >= 2.0 / 3.0]
    return {
        "combats": len(rows),
        "all": phase_summary(states),
        "early": phase_summary(early),
        "late": phase_summary(late),
        "search_override_rate": round(search_changes / len(states), 6) if states else None,
        "chosen_action_train_label_count_mean": _mean(action_label_counts),
        "chosen_action_train_label_count_median": (
            round(float(np.median(action_label_counts)), 6) if action_label_counts else None
        ),
        "chosen_action_unseen_rate": (
            round(sum(count == 0 for count in action_label_counts) / len(action_label_counts), 6)
            if action_label_counts
            else None
        ),
        "chosen_action_rare_rate_lt20": (
            round(sum(count < 20 for count in action_label_counts) / len(action_label_counts), 6)
            if action_label_counts
            else None
        ),
    }


def _combat_support(row: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    signature = str(row["encounter_signature"])
    deck_ids = [_normalize_card_id(str(card["id"])) for card in row["snapshot"]["deck"]]
    card_counts = training["card_state_count"]
    encounter_states = _trajectory_states(row)
    exact_quantiles = training["encounter_profile_quantiles"].get(signature, {})
    exact_round_quantiles = exact_quantiles.get("round")
    exact_hp_quantiles = exact_quantiles.get("hp_ratio")
    return {
        "train_exact_encounter_combats": len(training["encounter_combats"].get(signature, set())),
        "train_exact_encounter_transitions": int(training["encounter_transitions"][signature]),
        "deck_cards": len(deck_ids),
        "unseen_deck_card_fraction": round(
            sum(card_counts[card_id] == 0 for card_id in deck_ids) / max(len(deck_ids), 1), 6
        ),
        "rare_deck_card_fraction_lt20": round(
            sum(card_counts[card_id] < 20 for card_id in deck_ids) / max(len(deck_ids), 1), 6
        ),
        "online_max_round": max((state["round"] for state in encounter_states), default=None),
        "exact_encounter_train_round_p95": (
            exact_round_quantiles[3] if exact_round_quantiles else None
        ),
        "online_state_fraction_above_exact_round_p95": (
            round(
                sum(state["round"] > exact_round_quantiles[3] for state in encounter_states)
                / len(encounter_states),
                6,
            )
            if exact_round_quantiles and encounter_states
            else None
        ),
        "online_state_fraction_below_exact_hp_p05": (
            round(
                sum(state["hp_ratio"] < exact_hp_quantiles[1] for state in encounter_states)
                / len(encounter_states),
                6,
            )
            if exact_hp_quantiles and encounter_states
            else None
        ),
    }


def _static_support_summary(
    rows: list[dict[str, Any]], training: dict[str, Any]
) -> dict[str, Any]:
    supports = [_combat_support(row, training) for row in rows]
    encounter_counts = [support["train_exact_encounter_combats"] for support in supports]
    return {
        "combats": len(rows),
        "exact_encounter_train_combats_mean": _mean(encounter_counts),
        "exact_encounter_train_combats_median": (
            round(float(np.median(encounter_counts)), 6) if encounter_counts else None
        ),
        "exact_encounter_train_combats_min": min(encounter_counts) if encounter_counts else None,
        "rare_exact_encounter_count_lt3": sum(count < 3 for count in encounter_counts),
        "mean_unseen_deck_card_fraction": _mean(
            support["unseen_deck_card_fraction"] for support in supports
        ),
        "mean_rare_deck_card_fraction_lt20": _mean(
            support["rare_deck_card_fraction_lt20"] for support in supports
        ),
        "mean_online_state_fraction_above_exact_round_p95": _mean(
            support["online_state_fraction_above_exact_round_p95"]
            for support in supports
            if support["online_state_fraction_above_exact_round_p95"] is not None
        ),
        "mean_online_state_fraction_below_exact_hp_p05": _mean(
            support["online_state_fraction_below_exact_hp_p05"]
            for support in supports
            if support["online_state_fraction_below_exact_hp_p05"] is not None
        ),
    }


def build_failure_ratchet(
    evaluation_path: Path,
    samples_path: Path,
    output_path: Path,
    *,
    high_regret_threshold: float = 20.0,
    search_regression_threshold: float = 15.0,
) -> dict[str, Any]:
    evaluation_path = evaluation_path.resolve()
    samples_path = samples_path.resolve()
    output_path = output_path.resolve()
    evaluation = load_json(evaluation_path)
    training = _load_training_reference(samples_path)
    evaluable_rows = [
        row
        for row in evaluation["combat_rows"]
        if "p2_policy" in row and "p2_one_step" in row
    ]
    selected: list[tuple[dict[str, Any], list[str]]] = []
    for row in evaluable_rows:
        flags = classify_failure(
            row,
            high_regret_threshold=high_regret_threshold,
            search_regression_threshold=search_regression_threshold,
        )
        if flags:
            selected.append((row, flags))
    selected_ids = {str(row["scenario_id"]) for row, _ in selected}
    control_rows = [row for row in evaluable_rows if str(row["scenario_id"]) not in selected_ids]
    flag_counts = Counter(flag for _, flags in selected for flag in flags)
    by_act_pool = Counter(
        f"act{row['act']}:{encounter_pool(row)}" for row, _ in selected
    )
    failure_rows: list[dict[str, Any]] = []
    for row, flags in selected:
        failure_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "source_combat_id": row["source_combat_id"],
                "source_transition_id": row["source_transition_id"],
                "run_id": row["run_id"],
                "act": row["act"],
                "floor": row["floor"],
                "ascension": row["ascension"],
                "room_type": row["room_type"],
                "encounter_pool": encounter_pool(row),
                "encounter": row["encounter"],
                "encounter_signature": row["encounter_signature"],
                "flags": flags,
                "human": row["human"],
                "snapshot": row["snapshot"],
                "root": row["root"],
                "reconstruction_seed": row["reconstruction_seed"],
                "support": _combat_support(row, training),
                "first_search_divergence": first_search_divergence(row),
                "p2_policy": row["p2_policy"],
                "p2_one_step": row["p2_one_step"],
            }
        )
    introduced = [
        row for row, flags in selected if "search_introduced_death" in flags
    ]
    avoided = [
        row
        for row in evaluable_rows
        if row["p2_policy"].get("status") == "death"
        and row["p2_one_step"].get("status") != "death"
    ]
    result = {
        "schema_version": FAILURE_RATCHET_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "pass",
        "source": {
            "evaluation_path": str(evaluation_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "evaluation_schema_version": evaluation.get("schema_version"),
            "samples_path": str(samples_path),
            "samples_sha256": sha256_file(samples_path),
        },
        "criteria": {
            "high_human_regret_hp": high_regret_threshold,
            "search_regression_hp": search_regression_threshold,
            "always_include_policy_death": True,
            "always_include_one_step_death": True,
        },
        "summary": {
            "evaluable_combats": len(evaluable_rows),
            "failure_combats": len(selected),
            "control_combats": len(control_rows),
            "flag_counts": dict(sorted(flag_counts.items())),
            "by_act_encounter_pool": dict(sorted(by_act_pool.items())),
            "train_states": training["state_count"],
        },
        "distribution_shift": {
            "static_input_support": {
                "failure": _static_support_summary(
                    [row for row, _ in selected], training
                ),
                "control": _static_support_summary(control_rows, training),
            },
            "on_policy_trajectory": {
                "failure": _trajectory_distribution_summary(
                    [row for row, _ in selected], training
                ),
                "control": _trajectory_distribution_summary(control_rows, training),
            },
            "interpretation_boundary": (
                "This is an interpretable support audit over common visible features, not a "
                "calibrated density model or proof that distribution shift caused the failure."
            ),
        },
        "search_takeover": {
            "introduced_deaths": [
                {
                    "scenario_id": row["scenario_id"],
                    "encounter": row["encounter"],
                    "policy_hp_loss": row["p2_policy"]["hp_loss"],
                    "one_step_hp_loss": row["p2_one_step"]["hp_loss"],
                    "first_divergence": first_search_divergence(row),
                }
                for row in introduced
            ],
            "avoided_deaths": [
                {
                    "scenario_id": row["scenario_id"],
                    "encounter": row["encounter"],
                    "policy_hp_loss": row["p2_policy"]["hp_loss"],
                    "one_step_hp_loss": row["p2_one_step"]["hp_loss"],
                    "first_divergence": first_search_divergence(row),
                }
                for row in avoided
            ],
        },
        "failure_combats": failure_rows,
    }
    write_json_atomic(output_path, result)
    return result
