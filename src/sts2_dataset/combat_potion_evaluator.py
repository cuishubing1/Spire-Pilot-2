from __future__ import annotations

import statistics
from typing import Any

from .combat_potions import POTION_SPECS_BY_ID


POTION_PROPOSAL_VERSION = "combat-potion-proposal-0.1.0"


def _eligible_evaluations(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("evaluations") or []
        if bool(row.get("selection_eligible"))
    ]


def _best_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no eligible counterfactual evaluation")
    return max(
        rows,
        key=lambda row: (
            float(row.get("risk_adjusted_value") or 0.0),
            float(row.get("mean_value") or 0.0),
            float(row.get("policy_probability") or 0.0),
            str((row.get("candidate") or {}).get("candidate_id") or ""),
        ),
    )


def _worlds_by_id(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(world["determinization_id"]): world
        for world in evaluation.get("worlds") or []
    }


def _mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    if len(values) != len(rows):
        raise ValueError(f"counterfactual worlds lack numeric field: {'.'.join(path)}")
    return statistics.fmean(values) if values else 0.0


def _delta(
    use_worlds: list[dict[str, Any]],
    hold_worlds: list[dict[str, Any]],
    path: tuple[str, ...],
) -> float:
    return _mean(use_worlds, path) - _mean(hold_worlds, path)


def _optional_delta(
    use_worlds: list[dict[str, Any]],
    hold_worlds: list[dict[str, Any]],
    path: tuple[str, ...],
    observed_flag: str,
) -> float | None:
    if not all(
        bool((world.get("exact_transition") or {}).get(observed_flag, True))
        for world in [*use_worlds, *hold_worlds]
    ):
        return None
    return _delta(use_worlds, hold_worlds, path)


def _direction(values: list[float | None], *, epsilon: float = 1e-8) -> str:
    usable = [float(value) for value in values if value is not None]
    positive = any(value > epsilon for value in usable)
    negative = any(value < -epsilon for value in usable)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "neutral"


def build_paired_potion_proposal(
    *,
    potion_id: str,
    use_report: dict[str, Any],
    hold_report: dict[str, Any],
    state_fingerprint: str,
    target_index: int | None = None,
) -> dict[str, Any]:
    """Compare use-now and hold-this-turn branches under common RNG worlds.

    The output is deliberately evidence, not a calibrated probability.  It
    measures the tactical consequence of consuming one potion with the upper
    run-level resource price excluded from the search objective.  A future
    upper-level agent may combine this evidence with inventory pressure,
    encounter importance and replacement expectations.
    """

    try:
        spec = POTION_SPECS_BY_ID[potion_id]
    except KeyError as exc:
        raise ValueError(f"unknown Ironclad-applicable potion: {potion_id}") from exc
    if spec.evaluator != "paired_turn_boundary":
        raise ValueError(
            f"{potion_id} requires evaluator={spec.evaluator}, not paired_turn_boundary"
        )

    use_rows = [
        row
        for row in _eligible_evaluations(use_report)
        if (row.get("candidate") or {}).get("action_type") == "use_potion"
        and str((row.get("candidate") or {}).get("source_id") or "") == potion_id
        and (
            target_index is None
            or (row.get("candidate") or {}).get("target_index") == target_index
        )
    ]
    use = _best_evaluation(use_rows)
    hold = _best_evaluation(
        [
            row
            for row in _eligible_evaluations(hold_report)
            if (row.get("candidate") or {}).get("action_type") != "use_potion"
        ]
    )

    use_by_id = _worlds_by_id(use)
    hold_by_id = _worlds_by_id(hold)
    if not use_by_id or set(use_by_id) != set(hold_by_id):
        raise ValueError("use and hold reports do not share identical determinizations")
    world_ids = sorted(use_by_id)
    use_worlds = [use_by_id[world_id] for world_id in world_ids]
    hold_worlds = [hold_by_id[world_id] for world_id in world_ids]

    use_death = _mean(use_worlds, ("outcome", "death_probability"))
    hold_death = _mean(hold_worlds, ("outcome", "death_probability"))
    use_end_hp = _mean(use_worlds, ("outcome", "end_hp"))
    hold_end_hp = _mean(hold_worlds, ("outcome", "end_hp"))

    combat_value_gain = float(use["risk_adjusted_value"]) - float(hold["risk_adjusted_value"])
    exact_hp_saved = -_delta(use_worlds, hold_worlds, ("exact_transition", "hp_loss"))
    exact_enemy_hp_loss_gain = _optional_delta(
        use_worlds, hold_worlds, ("exact_transition", "enemy_hp_loss"),
        "enemy_state_observed",
    )
    exact_enemies_killed_gain = _optional_delta(
        use_worlds, hold_worlds, ("exact_transition", "enemies_killed"),
        "enemy_state_observed",
    )
    engine_direction = _direction(
        [exact_hp_saved, exact_enemy_hp_loss_gain, exact_enemies_killed_gain]
    )
    learned_direction = _direction([combat_value_gain])
    comparable_directions = {engine_direction, learned_direction} <= {"positive", "negative"}
    direction_agreement = (
        engine_direction == learned_direction if comparable_directions else None
    )

    return {
        "schema_version": POTION_PROPOSAL_VERSION,
        "state_fingerprint": state_fingerprint,
        "potion_id": potion_id,
        "potion_title_zh": spec.title_zh,
        "requested_target_index": target_index,
        "evaluator": spec.evaluator,
        "horizon": "next_player_turn_or_combat_terminal",
        "information_boundary": use_report.get("information_boundary"),
        "calibration_status": "uncalibrated_evidence_only",
        "tactical_necessity": None,
        "estimate_confidence": None,
        "use_candidate": use["candidate"],
        "hold_candidate": hold["candidate"],
        "paired_world_count": len(world_ids),
        "paired_determinization_ids": world_ids,
        "tactical_evidence": {
            "combat_value_gain": combat_value_gain,
            "mean_value_gain": float(use["mean_value"]) - float(hold["mean_value"]),
            "lower_tail_value_gain": float(use["lower_tail_cvar"])
            - float(hold["lower_tail_cvar"]),
            "death_risk_reduction": hold_death - use_death,
            "predicted_end_hp_gain": use_end_hp - hold_end_hp,
            "exact_boundary_hp_saved": exact_hp_saved,
            "exact_enemy_hp_loss_gain": exact_enemy_hp_loss_gain,
            "exact_enemies_killed_gain": exact_enemies_killed_gain,
            "exact_block_delta_gain": _optional_delta(
                use_worlds, hold_worlds, ("exact_transition", "block_delta"),
                "block_observed",
            ),
            "exact_energy_delta_gain": _optional_delta(
                use_worlds, hold_worlds, ("exact_transition", "energy_delta"),
                "energy_observed",
            ),
            "exact_hand_count_delta_gain": _optional_delta(
                use_worlds, hold_worlds, ("exact_transition", "hand_count_delta"),
                "hand_observed",
            ),
            "exact_potion_count_delta": _mean(
                use_worlds, ("exact_transition", "potion_count_delta")
            ),
        },
        "evidence_diagnostics": {
            "engine_short_horizon_direction": engine_direction,
            "learned_leaf_value_direction": learned_direction,
            "direction_agreement": direction_agreement,
            "requires_mechanic_or_deeper_search_review": direction_agreement is False,
        },
        "upper_agent_fields_not_applied": [
            "potion_shadow_price",
            "inventory_pressure",
            "encounter_importance",
            "replacement_expectation",
        ],
    }
