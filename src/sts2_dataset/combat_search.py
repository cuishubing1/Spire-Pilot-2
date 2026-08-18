from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from .util import canonical_json, sha256_bytes


SEARCH_VERSION = "combat-search-0.2.0"
SEARCH_TEACHER_VERSION = "combat-search-teacher-0.1.0"


@dataclass(frozen=True)
class SearchBudgetInputs:
    policy_probabilities: Sequence[float]
    player_hp: float
    player_max_hp: float
    predicted_death_probability: float = 0.0


@dataclass(frozen=True)
class SearchOutcome:
    """One engine rollout or provisional leaf evaluation.

    ``value`` is always expressed from the root combat state's perspective.
    The remaining fields stay separate so later fusion/distillation work does
    not have to reverse engineer a single scalar Q value.
    """

    value: float
    death_probability: float
    end_hp: float
    potion_spent: float
    max_hp_delta: float
    terminal: bool
    leaf_source: str
    determinization_id: str
    depth: int = 0
    max_hp_delta_raw: float = 0.0
    max_hp_growth_cap: float = 0.0


@dataclass
class SearchEdgeStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    outcomes: list[SearchOutcome] | None = None

    def __post_init__(self) -> None:
        if self.outcomes is None:
            self.outcomes = []

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def update(self, outcome: SearchOutcome) -> None:
        self.visits += 1
        self.value_sum += float(outcome.value)
        assert self.outcomes is not None
        self.outcomes.append(outcome)

    def summary(self, *, cvar_alpha: float) -> dict[str, object]:
        rows = self.outcomes or []
        values = [float(row.value) for row in rows]
        end_hp = [float(row.end_hp) for row in rows]
        leaf_sources: dict[str, int] = {}
        by_determinization: dict[str, list[float]] = {}
        for row in rows:
            leaf_sources[row.leaf_source] = leaf_sources.get(row.leaf_source, 0) + 1
            by_determinization.setdefault(row.determinization_id, []).append(float(row.value))
        determinization_means = [statistics.fmean(group) for group in by_determinization.values()]
        determinization_deaths = {
            identity: statistics.fmean(
                float(row.death_probability)
                for row in rows
                if row.determinization_id == identity
            )
            for identity in by_determinization
        }
        determinization_rows = {
            identity: [row for row in rows if row.determinization_id == identity]
            for identity in by_determinization
        }
        return {
            "visits": self.visits,
            "prior": float(self.prior),
            "mean_value": self.mean_value,
            "value_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "lower_tail_cvar": lower_tail_cvar(values, cvar_alpha),
            "determinization_mean_value": statistics.fmean(determinization_means)
            if determinization_means else 0.0,
            "determinization_lower_tail_cvar": lower_tail_cvar(
                determinization_means, cvar_alpha
            ),
            "determinization_death_probability": statistics.fmean(
                determinization_deaths.values()
            ) if determinization_deaths else 0.0,
            "determinization_values": {
                identity: {
                    "visits": len(by_determinization[identity]),
                    "mean_value": statistics.fmean(by_determinization[identity]),
                    "death_probability": determinization_deaths[identity],
                    "end_hp_mean": statistics.fmean(
                        float(row.end_hp) for row in determinization_rows[identity]
                    ),
                    "terminal_fraction": sum(
                        bool(row.terminal) for row in determinization_rows[identity]
                    ) / len(determinization_rows[identity]),
                    "leaf_sources": dict(sorted(Counter(
                        row.leaf_source for row in determinization_rows[identity]
                    ).items())),
                }
                for identity in sorted(by_determinization)
            },
            "death_probability": statistics.fmean(
                float(row.death_probability) for row in rows
            ) if rows else 0.0,
            "end_hp_mean": statistics.fmean(end_hp) if end_hp else 0.0,
            "end_hp_std": statistics.pstdev(end_hp) if len(end_hp) > 1 else 0.0,
            "potion_spent_mean": statistics.fmean(
                float(row.potion_spent) for row in rows
            ) if rows else 0.0,
            "max_hp_delta_mean": statistics.fmean(
                float(row.max_hp_delta) for row in rows
            ) if rows else 0.0,
            "max_hp_delta_raw_mean": statistics.fmean(
                float(row.max_hp_delta_raw) for row in rows
            ) if rows else 0.0,
            "max_hp_growth_cap_mean": statistics.fmean(
                float(row.max_hp_growth_cap) for row in rows
            ) if rows else 0.0,
            "terminal_fraction": sum(bool(row.terminal) for row in rows) / len(rows)
            if rows else 0.0,
            "determinization_count": len(by_determinization),
            "determinization_value_std": statistics.pstdev(determinization_means)
            if len(determinization_means) > 1 else 0.0,
            "mean_search_depth": statistics.fmean(float(row.depth) for row in rows)
            if rows else 0.0,
            "max_search_depth": max((int(row.depth) for row in rows), default=0),
            "leaf_sources": dict(sorted(leaf_sources.items())),
        }


def lower_tail_cvar(values: Sequence[float], alpha: float) -> float:
    """Mean of the worst ``alpha`` fraction for a reward-like value."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("CVaR alpha must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    count = max(1, math.ceil(len(ordered) * float(alpha)))
    return statistics.fmean(ordered[:count])


def risk_adjusted_root_score(
    *, mean_value: float, lower_tail_value: float, mean_weight: float, cvar_weight: float
) -> float:
    if mean_weight < 0.0 or cvar_weight < 0.0 or mean_weight + cvar_weight <= 0.0:
        raise ValueError("root score weights must be non-negative and not both zero")
    total = float(mean_weight) + float(cvar_weight)
    return (
        float(mean_weight) * float(mean_value)
        + float(cvar_weight) * float(lower_tail_value)
    ) / total


def build_search_teacher_record(
    root_sample: dict[str, Any],
    search_report: dict[str, Any],
    *,
    search_seed: int,
    model_version: str,
) -> dict[str, Any]:
    """Freeze an auditable search root without declaring it a good label.

    The record deliberately preserves visit, value, tail-risk and per-world
    evidence separately. A later dataset builder can apply stricter quality
    gates without rerunning the engine or reverse engineering a scalar Q.
    """

    root_candidates = root_sample.get("candidates") or []
    root_ids = [str(row["candidate_id"]) for row in root_candidates]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("search teacher root contains duplicate candidate ids")
    action_rows = search_report.get("actions") or []
    action_ids = [str(row["candidate"]["candidate_id"]) for row in action_rows]
    if set(action_ids) != set(root_ids) or len(action_ids) != len(root_ids):
        raise ValueError("search teacher actions do not match the root candidates")

    root_payload = {
        "observation_version": root_sample.get("observation_version"),
        "action_version": root_sample.get("action_version"),
        "act": int(root_sample.get("act") or 0),
        "floor": int(root_sample.get("floor") or 0),
        "observation": root_sample.get("observation"),
        "candidates": root_candidates,
    }
    evidence = []
    for row in action_rows:
        evidence.append({
            "candidate": row["candidate"],
            "prior": float(row["prior"]),
            "visits": int(row["visits"]),
            "visit_policy_probability": float(row.get("visit_policy_probability", 0.0)),
            "mean_value": float(row["mean_value"]),
            "value_std": float(row["value_std"]),
            "lower_tail_cvar": float(row["lower_tail_cvar"]),
            "determinization_mean_value": float(row["determinization_mean_value"]),
            "determinization_lower_tail_cvar": float(
                row["determinization_lower_tail_cvar"]
            ),
            "death_probability": float(row["death_probability"]),
            "end_hp_mean": float(row["end_hp_mean"]),
            "end_hp_std": float(row["end_hp_std"]),
            "potion_spent_mean": float(row["potion_spent_mean"]),
            "max_hp_delta_mean": float(row["max_hp_delta_mean"]),
            "terminal_fraction": float(row["terminal_fraction"]),
            "mean_search_depth": float(row["mean_search_depth"]),
            "max_search_depth": int(row["max_search_depth"]),
            "leaf_sources": row["leaf_sources"],
            "determinization_count": int(row["determinization_count"]),
            "determinization_values": row["determinization_values"],
            "selection_score": float(row["selection_score"]),
            "selection_eligible": bool(row["selection_eligible"]),
        })
    evidence.sort(key=lambda row: int(row["candidate"]["candidate_index"]))
    return {
        "schema_version": SEARCH_TEACHER_VERSION,
        "search_version": SEARCH_VERSION,
        "root_fingerprint": sha256_bytes(canonical_json(root_payload).encode("utf-8")),
        "root": root_payload,
        "search_seed": int(search_seed),
        "model_version": str(model_version),
        "requested_budget": int(search_report["requested_budget"]),
        "effective_budget": int(search_report["effective_budget"]),
        "root_determinization_count": int(search_report["root_determinization_count"]),
        "root_selection_rule": str(search_report["root_selection_rule"]),
        "policy_candidate_id": str(search_report["policy_candidate"]["candidate_id"]),
        "search_candidate_id": str(
            search_report["search_selected_candidate"]["candidate_id"]
        ),
        "chosen_candidate_id": str(search_report["chosen_candidate"]["candidate_id"]),
        "policy_fallback": search_report["policy_fallback"],
        "actions": evidence,
    }


def root_visit_threshold(
    visits: Sequence[int], *, minimum_visits: int, minimum_fraction_of_max: float
) -> int:
    """Require evidence before trusting a root action's risk estimate.

    The threshold never exceeds the most visited action.  This preserves a
    usable fallback when a tiny smoke-test budget only visits each action once.
    """

    if minimum_visits < 1:
        raise ValueError("minimum root selection visits must be positive")
    if not 0.0 <= minimum_fraction_of_max <= 1.0:
        raise ValueError("minimum root visit fraction must be in [0, 1]")
    maximum = max((max(0, int(value)) for value in visits), default=0)
    if maximum == 0:
        return 0
    requested = max(
        int(minimum_visits), math.ceil(maximum * float(minimum_fraction_of_max))
    )
    return min(maximum, requested)


def root_coverage_budget(
    requested_budget: int,
    *,
    legal_action_count: int,
    minimum_visits_per_action: int,
) -> tuple[int, int]:
    """Return effective budget and forced root-coverage simulations.

    A requested budget smaller than the legal action count is a breadth smoke
    test, not meaningful search.  Repeating every root action under distinct
    determinizations before PUCT takes over prevents one-sample leaf outliers
    from winning solely because the requested budget was too small.
    """

    if requested_budget < 1:
        raise ValueError("requested search budget must be positive")
    if legal_action_count < 1:
        raise ValueError("legal action count must be positive")
    if minimum_visits_per_action < 1:
        raise ValueError("minimum root visits per action must be positive")
    forced = int(legal_action_count) * int(minimum_visits_per_action)
    return max(int(requested_budget), forced), forced


def forced_root_action_index(
    simulation: int,
    *,
    legal_action_count: int,
    forced_coverage_simulations: int,
) -> int | None:
    """Round-robin root action index during the forced coverage prefix."""

    if simulation < 0:
        raise ValueError("simulation index must be non-negative")
    if legal_action_count < 1:
        raise ValueError("legal action count must be positive")
    if forced_coverage_simulations < legal_action_count:
        raise ValueError("forced coverage must include every legal action")
    if simulation >= forced_coverage_simulations:
        return None
    return int(simulation) % int(legal_action_count)


def paired_root_determinization_index(
    simulation: int,
    *,
    legal_action_count: int,
    forced_coverage_simulations: int,
    determinization_count: int,
) -> int:
    """Map simulations onto a shared root-world schedule.

    During forced root coverage every action is evaluated in world 0, then
    every action in world 1, and so on. Later PUCT simulations cycle through
    the same bounded world set instead of continually introducing worlds that
    only one root action may see.
    """

    if simulation < 0:
        raise ValueError("simulation index must be non-negative")
    if legal_action_count < 1:
        raise ValueError("legal action count must be positive")
    if forced_coverage_simulations < legal_action_count:
        raise ValueError("forced coverage must include every legal action")
    if forced_coverage_simulations % legal_action_count != 0:
        raise ValueError("forced coverage must contain complete root-action rounds")
    if determinization_count < 1:
        raise ValueError("determinization count must be positive")
    forced_worlds = forced_coverage_simulations // legal_action_count
    if forced_worlds > determinization_count:
        raise ValueError("determinization count must cover every forced root round")
    if simulation < forced_coverage_simulations:
        return simulation // legal_action_count
    return (forced_worlds + simulation - forced_coverage_simulations) % determinization_count


def normalized_policy_entropy(probabilities: Sequence[float]) -> float:
    values = [max(0.0, float(value)) for value in probabilities]
    total = sum(values)
    if len(values) <= 1 or total <= 0.0:
        return 0.0
    normalized = [value / total for value in values if value > 0.0]
    entropy = -sum(value * math.log(value) for value in normalized)
    return entropy / math.log(len(values))


def adaptive_search_budget(inputs: SearchBudgetInputs, config: dict) -> tuple[int, dict[str, float | str]]:
    budgets = config["budgets"]
    triggers = config["adaptive_triggers"]
    probabilities = sorted(
        (max(0.0, float(value)) for value in inputs.policy_probabilities), reverse=True
    )
    entropy = normalized_policy_entropy(probabilities)
    top_gap = probabilities[0] - probabilities[1] if len(probabilities) > 1 else 1.0
    hp_fraction = (
        max(0.0, inputs.player_hp) / inputs.player_max_hp
        if inputs.player_max_hp > 0 else 0.0
    )
    death_probability = max(0.0, min(1.0, inputs.predicted_death_probability))

    if hp_fraction <= float(triggers["critical_hp_fraction"]):
        level = "critical"
    elif (
        death_probability >= float(triggers["high_death_probability"])
        or hp_fraction <= float(triggers["low_hp_fraction"])
        or entropy >= float(triggers["high_entropy_min"])
        or top_gap <= float(triggers["small_top_probability_gap"])
    ):
        level = "high"
    elif entropy <= float(triggers["low_entropy_max"]):
        level = "low"
    else:
        level = "default"
    return int(budgets[level]), {
        "level": level,
        "normalized_entropy": round(entropy, 6),
        "top_probability_gap": round(top_gap, 6),
        "hp_fraction": round(hp_fraction, 6),
        "predicted_death_probability": round(death_probability, 6),
    }


def puct_score(
    *,
    mean_value: float,
    prior: float,
    parent_visits: int,
    child_visits: int,
    c_puct: float,
) -> float:
    exploration = (
        float(c_puct)
        * max(0.0, float(prior))
        * math.sqrt(max(1, int(parent_visits)))
        / (1 + max(0, int(child_visits)))
    )
    return float(mean_value) + exploration
