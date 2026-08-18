from __future__ import annotations

import itertools
import math
import statistics
from typing import Any, Iterable

from .combat_search import lower_tail_cvar
from .util import canonical_json, sha256_bytes


COUNTERFACTUAL_TEACHER_VERSION = "combat-counterfactual-teacher-0.2.0"
COUNTERFACTUAL_DATASET_VERSION = "combat-counterfactual-dataset-0.1.0"


def counterfactual_gate_variants(reasons: Iterable[str]) -> dict[str, bool]:
    values = set(reasons)
    risk = {
        "low_hp",
        "late_round_ood",
        "high_visible_incoming_loss",
        "rare_chosen_action",
    }
    uncertainty = {
        "high_policy_entropy",
        "low_policy_margin",
        "rare_chosen_action",
    }
    has_uncertainty = bool(values & uncertainty)
    return {
        "any_trigger": bool(values),
        "two_signals": len(values) >= 2,
        "risk_and_uncertainty": bool(values & risk) and has_uncertainty,
        "strict": bool(
            "rare_chosen_action" in values
            or "late_round_ood" in values
            or ("low_hp" in values and has_uncertainty)
            or (
                "high_visible_incoming_loss" in values
                and "high_policy_entropy" in values
                and "low_policy_margin" in values
            )
        ),
    }


def on_policy_trigger_reasons(
    *,
    hp_ratio: float,
    round_number: int,
    exact_encounter_round_p95: float | None,
    incoming_hp_loss: float,
    policy_entropy: float,
    policy_margin: float,
    chosen_action_train_count: int,
    low_hp_threshold: float = 0.40,
    incoming_hp_loss_threshold: float = 10.0,
    high_entropy_threshold: float = 0.55,
    low_margin_threshold: float = 0.15,
    rare_action_threshold: int = 20,
) -> list[str]:
    reasons: list[str] = []
    if float(hp_ratio) <= float(low_hp_threshold):
        reasons.append("low_hp")
    if (
        exact_encounter_round_p95 is not None
        and int(round_number) > float(exact_encounter_round_p95)
    ):
        reasons.append("late_round_ood")
    if float(incoming_hp_loss) >= float(incoming_hp_loss_threshold):
        reasons.append("high_visible_incoming_loss")
    if float(policy_entropy) >= float(high_entropy_threshold):
        reasons.append("high_policy_entropy")
    if float(policy_margin) <= float(low_margin_threshold):
        reasons.append("low_policy_margin")
    if int(chosen_action_train_count) < int(rare_action_threshold):
        reasons.append("rare_chosen_action")
    return reasons


def root_priority_score(root: dict[str, Any], *, combat_failure: bool) -> float:
    reason_weights = {
        "low_hp": 4.0,
        "late_round_ood": 3.0,
        "high_visible_incoming_loss": 2.0,
        "high_policy_entropy": 1.0,
        "low_policy_margin": 1.0,
        "rare_chosen_action": 2.0,
    }
    score = sum(reason_weights.get(reason, 0.0) for reason in root.get("trigger_reasons") or [])
    if combat_failure:
        score += 5.0
    score += 2.0 * (1.0 - float(root.get("hp_ratio") or 0.0))
    score += min(2.0, float(root.get("round") or 1.0) / 5.0)
    return round(score, 6)


def select_counterfactual_roots(
    roots: Iterable[dict[str, Any]],
    *,
    combat_failure: bool,
    limit: int,
    strategy: str = "diverse",
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("counterfactual root limit must be positive")
    candidates = []
    for root in roots:
        if not root.get("trigger_reasons") and not combat_failure:
            continue
        value = dict(root)
        value["priority_score"] = root_priority_score(
            value, combat_failure=combat_failure
        )
        candidates.append(value)
    candidates.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            int(row.get("step") or 0),
            str(row.get("root_fingerprint") or ""),
        )
    )
    if strategy not in {"diverse", "earliest"}:
        raise ValueError(f"unsupported counterfactual root strategy: {strategy}")
    if strategy == "earliest":
        return sorted(
            candidates,
            key=lambda row: (
                int(row.get("step") or 0),
                -float(row["priority_score"]),
                str(row.get("root_fingerprint") or ""),
            ),
        )[:limit]
    if len(candidates) <= 1 or limit == 1:
        return candidates[:limit]

    # A risk-only ranking tends to concentrate labels near terminal states,
    # where all shortlisted actions may already be forced wins or deaths.
    # Keep the highest-risk state, then deliberately cover the earliest point
    # at which the trajectory exhibited an on-policy trigger.
    selected = [candidates[0]]
    remaining = candidates[1:]
    early_pool = [row for row in remaining if row.get("trigger_reasons")]
    if not early_pool:
        early_pool = remaining
    if early_pool:
        earliest = min(
            early_pool,
            key=lambda row: (
                int(row.get("step") or 0),
                str(row.get("root_fingerprint") or ""),
            ),
        )
        selected.append(earliest)
        remaining = [row for row in remaining if row is not earliest]
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected[:limit]


def summarize_counterfactual_root(
    actions: list[dict[str, Any]],
    *,
    utility_epsilon: float = 1e-6,
) -> dict[str, Any]:
    teacher_eligible = bool(actions) and all(
        bool(action.get("teacher_eligible")) for action in actions
    )
    eligible = [
        action
        for action in actions
        if action.get("teacher_eligible") and action.get("mean_utility") is not None
    ]
    if not teacher_eligible or not eligible:
        return {
            "teacher_eligible": teacher_eligible,
            "informative": False,
            "utility_range": None,
            "policy_candidate_id": None,
            "best_candidate_id": None,
            "policy_suboptimal": None,
            "policy_utility_regret": None,
        }

    policy_action = max(
        eligible,
        key=lambda row: (
            float(row.get("policy_probability") or 0.0),
            str(row["candidate"]["candidate_id"]),
        ),
    )
    utilities = [float(row["mean_utility"]) for row in eligible]
    best_utility = max(utilities)
    tied_best = [
        row
        for row in eligible
        if math.isclose(
            float(row["mean_utility"]),
            best_utility,
            abs_tol=utility_epsilon,
        )
    ]
    best_action = max(
        tied_best,
        key=lambda row: (
            float(row.get("policy_probability") or 0.0),
            str(row["candidate"]["candidate_id"]),
        ),
    )
    policy_utility = float(policy_action["mean_utility"])
    regret = max(0.0, best_utility - policy_utility)
    utility_range = max(utilities) - min(utilities)
    return {
        "teacher_eligible": True,
        "informative": utility_range > utility_epsilon,
        "utility_range": utility_range,
        "policy_candidate_id": str(policy_action["candidate"]["candidate_id"]),
        "best_candidate_id": str(best_action["candidate"]["candidate_id"]),
        "policy_suboptimal": regret > utility_epsilon,
        "policy_utility_regret": regret,
    }


def summarize_counterfactual_action(
    candidate: dict[str, Any],
    worlds: list[dict[str, Any]],
    *,
    cvar_alpha: float,
) -> dict[str, Any]:
    utilities = [float(world["utility"]) for world in worlds]
    terminal_worlds = [world for world in worlds if bool(world.get("terminal"))]
    return {
        "candidate": candidate,
        "world_count": len(worlds),
        "terminal_world_count": len(terminal_worlds),
        "terminal_fraction": len(terminal_worlds) / len(worlds) if worlds else 0.0,
        "teacher_eligible": bool(worlds) and len(terminal_worlds) == len(worlds),
        "mean_utility": statistics.fmean(utilities) if utilities else None,
        "lower_tail_cvar": lower_tail_cvar(utilities, cvar_alpha) if utilities else None,
        "death_probability": (
            statistics.fmean(float(world["death"]) for world in worlds)
            if worlds
            else None
        ),
        "mean_terminal_hp": (
            statistics.fmean(float(world["terminal_hp"]) for world in terminal_worlds)
            if terminal_worlds
            else None
        ),
        "mean_hp_loss": (
            statistics.fmean(float(world["hp_loss"]) for world in terminal_worlds)
            if terminal_worlds
            else None
        ),
        "mean_potion_spent": (
            statistics.fmean(float(world["potion_spent"]) for world in terminal_worlds)
            if terminal_worlds
            else None
        ),
        "worlds": worlds,
    }


def build_pairwise_labels(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    eligible = [row for row in actions if row.get("teacher_eligible")]
    for left, right in itertools.combinations(eligible, 2):
        left_worlds = {
            str(world["determinization_id"]): world for world in left["worlds"]
        }
        right_worlds = {
            str(world["determinization_id"]): world for world in right["worlds"]
        }
        shared = sorted(set(left_worlds) & set(right_worlds))
        if not shared:
            continue
        deltas = [
            float(left_worlds[identity]["utility"])
            - float(right_worlds[identity]["utility"])
            for identity in shared
        ]
        mean_delta = statistics.fmean(deltas)
        left_id = str(left["candidate"]["candidate_id"])
        right_id = str(right["candidate"]["candidate_id"])
        winner = left_id if mean_delta > 0 else right_id if mean_delta < 0 else None
        labels.append(
            {
                "left_candidate_id": left_id,
                "right_candidate_id": right_id,
                "shared_determinization_count": len(shared),
                "mean_utility_delta_left_minus_right": mean_delta,
                "lower_tail_utility_delta_left_minus_right": lower_tail_cvar(
                    deltas, 0.5
                ),
                "winner_candidate_id": winner,
                "tie": math.isclose(mean_delta, 0.0, abs_tol=1e-12),
            }
        )
    return labels


def build_counterfactual_training_examples(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    if report.get("dataset_split") != "train":
        raise ValueError("counterfactual training examples require a train-split report")
    examples: list[dict[str, Any]] = []
    for combat in report.get("combats") or []:
        for root in combat.get("teacher_roots") or []:
            if not root.get("teacher_eligible") or not root.get("informative"):
                continue
            sample = root["root_sample"]
            candidate_indices = {
                str(candidate["candidate_id"]): int(candidate["candidate_index"])
                for candidate in sample["candidates"]
            }
            actions = []
            for action in root["actions"]:
                candidate_id = str(action["candidate"]["candidate_id"])
                if candidate_id not in candidate_indices:
                    raise ValueError(
                        f"counterfactual candidate is absent from root sample: {candidate_id}"
                    )
                actions.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_index": candidate_indices[candidate_id],
                        "policy_probability": float(action["policy_probability"]),
                        "mean_utility": float(action["mean_utility"]),
                        "lower_tail_cvar": float(action["lower_tail_cvar"]),
                        "death_probability": float(action["death_probability"]),
                        "mean_hp_loss": float(action["mean_hp_loss"]),
                        "mean_potion_spent": float(action["mean_potion_spent"]),
                    }
                )
            pairs = []
            for pair in root.get("pairwise_labels") or []:
                winner = pair.get("winner_candidate_id")
                if pair.get("tie") or winner is None:
                    continue
                left_id = str(pair["left_candidate_id"])
                right_id = str(pair["right_candidate_id"])
                if left_id not in candidate_indices or right_id not in candidate_indices:
                    raise ValueError("pairwise candidate is absent from root sample")
                pairs.append(
                    {
                        "left_candidate_id": left_id,
                        "left_candidate_index": candidate_indices[left_id],
                        "right_candidate_id": right_id,
                        "right_candidate_index": candidate_indices[right_id],
                        "winner_candidate_id": str(winner),
                        "winner_candidate_index": candidate_indices[str(winner)],
                        "mean_utility_delta_left_minus_right": float(
                            pair["mean_utility_delta_left_minus_right"]
                        ),
                        "shared_determinization_count": int(
                            pair["shared_determinization_count"]
                        ),
                    }
                )
            payload = {
                "schema_version": COUNTERFACTUAL_DATASET_VERSION,
                "dataset_split": "train",
                "scenario_id": str(root["scenario_id"]),
                "root_fingerprint": str(root["root_fingerprint"]),
                "step": int(root["step"]),
                "act": int(combat["act"]),
                "ascension": int(combat["ascension"]),
                "encounter": str(combat["encounter"]),
                "trigger_reasons": list(root.get("trigger_reasons") or []),
                "determinization_count": int(root["determinization_count"]),
                "continuation_policy": str(root["continuation_policy"]),
                "policy_candidate_id": str(root["policy_candidate_id"]),
                "best_candidate_id": str(root["best_candidate_id"]),
                "policy_suboptimal": bool(root["policy_suboptimal"]),
                "policy_utility_regret": float(root["policy_utility_regret"]),
                "utility_range": float(root["utility_range"]),
                "sample": sample,
                "actions": actions,
                "pairwise_labels": pairs,
            }
            payload["example_id"] = "counterfactual-" + sha256_bytes(
                canonical_json(payload).encode("utf-8")
            )[:20].lower()
            examples.append(payload)
    examples.sort(
        key=lambda row: (
            int(row["act"]),
            str(row["scenario_id"]),
            int(row["step"]),
            str(row["root_fingerprint"]),
        )
    )
    return examples
