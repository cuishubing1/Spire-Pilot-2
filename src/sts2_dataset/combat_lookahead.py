from __future__ import annotations

import math
from typing import Any, Sequence

from .combat_online import visible_intent_end_turn_hp_loss


COMBAT_ONE_STEP_VERSION = "combat-one-step-0.4.0"


def one_step_takeover_ineligibility(
    evaluation: dict[str, Any],
    *,
    policy_candidate_id: str,
    minimum_potion_policy_probability: float,
) -> tuple[str, ...]:
    """Return conservative reasons that forbid a one-step policy override.

    P1 remains free to choose every legal action. These checks only stop a
    shallow, differently calibrated successor value from introducing a risky
    action that the behavior prior did not support. A future upper-level
    directive can lower the potion probability floor explicitly.
    """

    if not 0.0 <= minimum_potion_policy_probability <= 1.0:
        raise ValueError("minimum potion policy probability must be in [0, 1]")
    candidate = evaluation["candidate"]
    if str(candidate.get("candidate_id") or "") == policy_candidate_id:
        return ()

    reasons: list[str] = []
    action_type = str(candidate.get("action_type") or "")
    if action_type == "discard_potion":
        reasons.append("search_cannot_introduce_potion_discard")
    if (
        action_type == "use_potion"
        and float(evaluation.get("policy_probability") or 0.0)
        < minimum_potion_policy_probability
    ):
        reasons.append("search_potion_below_policy_support_floor")
    if (
        action_type == "use_potion"
        and str(candidate.get("source_id") or "") == "POTION.BLOCK_POTION"
        and float(evaluation.get("root_visible_end_turn_hp_loss") or 0.0) <= 0.0
        and not bool(evaluation.get("root_retains_block"))
    ):
        reasons.append("transient_block_potion_without_visible_attack")
    if action_type == "end_turn" and any(
        float((world.get("exact_transition") or {}).get("hp_loss") or 0.0) > 0.0
        for world in evaluation.get("worlds") or []
        if isinstance(world, dict)
    ):
        reasons.append("one_step_end_turn_has_exact_hp_loss")
    return tuple(reasons)


def apply_exact_terminal_death_veto(
    evaluations: Sequence[dict[str, Any]],
) -> int:
    """Veto certainly lethal engine branches when a certain survivor exists."""

    def exact_death(world: dict[str, Any]) -> bool:
        outcome = world.get("outcome") or {}
        return bool(outcome.get("terminal")) and float(
            outcome.get("death_probability") or 0.0
        ) >= 1.0

    safe_exists = any(
        evaluation.get("worlds")
        and all(not exact_death(world) for world in evaluation["worlds"])
        for evaluation in evaluations
        if bool(evaluation.get("selection_eligible", True))
    )
    if not safe_exists:
        return 0
    vetoed = 0
    for evaluation in evaluations:
        worlds = evaluation.get("worlds") or []
        if worlds and all(exact_death(world) for world in worlds):
            evaluation["selection_eligible"] = False
            reasons = list(evaluation.get("selection_ineligible_reasons") or [])
            reasons.append("exact_terminal_death_veto")
            evaluation["selection_ineligible_reasons"] = reasons
            vetoed += 1
    return vetoed


def policy_top_k(
    ranked_actions: Sequence[dict[str, Any]],
    top_k: int,
    *,
    required_categories: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return a semantic, category-aware shortlist for exact branch execution.

    Exact duplicate card instances share one executable representative and their
    policy mass is summed.  The raw policy argmax remains first so that search
    fallback retains P1 semantics rather than silently changing the base policy.
    """

    if top_k < 1:
        raise ValueError("one-step top_k must be positive")
    ordered = sorted(
        ranked_actions,
        key=lambda row: (
            -float(row["policy_probability"]),
            int(row["candidate"]["candidate_index"]),
        ),
    )
    if not ordered:
        return []

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        candidate = row["candidate"]
        key = str(
            candidate.get("search_equivalence_key")
            or candidate.get("candidate_id")
        )
        groups.setdefault(key, []).append(row)

    grouped: dict[str, dict[str, Any]] = {}
    for key, members in groups.items():
        representative = members[0]
        candidate = dict(representative["candidate"])
        candidate["equivalent_candidate_count"] = len(members)
        candidate["equivalent_candidate_ids"] = [
            str(member["candidate"]["candidate_id"]) for member in members
        ]
        row = dict(representative)
        row["candidate"] = candidate
        row["policy_probability"] = sum(
            float(member["policy_probability"]) for member in members
        )
        grouped[key] = row

    policy_key = str(
        ordered[0]["candidate"].get("search_equivalence_key")
        or ordered[0]["candidate"].get("candidate_id")
    )
    semantic_order = sorted(
        grouped.values(),
        key=lambda row: (
            -float(row["policy_probability"]),
            int(row["candidate"]["candidate_index"]),
        ),
    )
    policy_group = grouped[policy_key]
    semantic_order = [policy_group] + [
        row for row in semantic_order if row is not policy_group
    ]

    limit = min(top_k, len(semantic_order))
    selected = semantic_order[:limit]
    if limit < 2:
        return selected

    selected_keys = {
        str(row["candidate"].get("search_equivalence_key") or row["candidate"]["candidate_id"])
        for row in selected
    }
    for required in dict.fromkeys(str(value) for value in required_categories):
        if any(row["candidate"].get("search_category") == required for row in selected):
            continue
        replacement = next(
            (
                row
                for row in semantic_order
                if row["candidate"].get("search_category") == required
                and str(
                    row["candidate"].get("search_equivalence_key")
                    or row["candidate"]["candidate_id"]
                ) not in selected_keys
            ),
            None,
        )
        if replacement is None:
            continue
        replace_index = next(
            (
                index
                for index in range(len(selected) - 1, 0, -1)
                if selected[index]["candidate"].get("search_category")
                not in required_categories
            ),
            len(selected) - 1,
        )
        removed = selected[replace_index]
        selected_keys.discard(
            str(
                removed["candidate"].get("search_equivalence_key")
                or removed["candidate"]["candidate_id"]
            )
        )
        selected[replace_index] = replacement
        selected_keys.add(
            str(
                replacement["candidate"].get("search_equivalence_key")
                or replacement["candidate"]["candidate_id"]
            )
        )
    return selected


def required_search_categories(observation: dict[str, Any]) -> tuple[str, ...]:
    """Reserve a defensive branch when visible attacks exceed current block."""

    estimate = visible_intent_end_turn_hp_loss(observation)
    if estimate is not None and float(estimate["hp_loss"]) > 0.0:
        return ("card_block",)
    return ()


def regularized_one_step_score(
    *, value: float, policy_probability: float, policy_log_weight: float
) -> float:
    """Combine successor value with a conservative human-policy prior."""

    if policy_log_weight < 0.0:
        raise ValueError("policy log weight must be non-negative")
    probability = max(float(policy_probability), 1e-8)
    return float(value) + float(policy_log_weight) * math.log(probability)


def choose_one_step_candidate(
    evaluations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in evaluations if bool(row.get("selection_eligible", True))]
    if not eligible:
        raise ValueError("one-step evaluation produced no eligible candidate")
    return max(
        eligible,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["policy_probability"]),
            -int(row["candidate"]["candidate_index"]),
        ),
    )


def apply_policy_advantage_gate(
    *,
    search_choice: dict[str, Any],
    policy_choice: dict[str, Any],
    minimum_advantage: float,
    minimum_end_turn_advantage: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Require a material successor-value gain before overriding Policy."""

    if minimum_advantage < 0.0:
        raise ValueError("minimum one-step advantage must be non-negative")
    if minimum_end_turn_advantage is not None and minimum_end_turn_advantage < 0.0:
        raise ValueError("minimum end-turn advantage must be non-negative")
    if (
        search_choice["candidate"]["candidate_id"]
        == policy_choice["candidate"]["candidate_id"]
    ):
        return search_choice, None
    if not bool(policy_choice.get("selection_eligible", True)):
        return search_choice, {
            "reason": "policy_candidate_ineligible",
            "observed_advantage": None,
            "minimum_advantage": float(minimum_advantage),
        }
    advantage = float(search_choice["selection_score"]) - float(
        policy_choice["selection_score"]
    )
    required = float(minimum_advantage)
    if (
        search_choice["candidate"].get("action_type") == "end_turn"
        and policy_choice["candidate"].get("action_type") != "end_turn"
        and minimum_end_turn_advantage is not None
    ):
        required = max(required, float(minimum_end_turn_advantage))
    if advantage < required:
        return policy_choice, {
            "reason": "insufficient_value_advantage",
            "observed_advantage": advantage,
            "minimum_advantage": required,
        }
    return search_choice, None
