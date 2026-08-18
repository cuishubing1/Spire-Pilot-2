import json

import pytest
from jsonschema import Draft202012Validator

from sts2_dataset.constants import ROOT
from sts2_dataset.combat_search import (
    SearchBudgetInputs,
    SearchEdgeStats,
    SearchOutcome,
    adaptive_search_budget,
    build_search_teacher_record,
    forced_root_action_index,
    lower_tail_cvar,
    normalized_policy_entropy,
    paired_root_determinization_index,
    puct_score,
    risk_adjusted_root_score,
    root_coverage_budget,
    root_visit_threshold,
)


CONFIG = {
    "budgets": {"low": 32, "default": 64, "high": 128, "critical": 256},
    "adaptive_triggers": {
        "low_entropy_max": 0.35,
        "high_entropy_min": 0.65,
        "small_top_probability_gap": 0.15,
        "low_hp_fraction": 0.35,
        "critical_hp_fraction": 0.18,
        "high_death_probability": 0.12,
    },
}


def test_adaptive_budget_uses_small_search_for_confident_healthy_state():
    budget, details = adaptive_search_budget(
        SearchBudgetInputs([0.98, 0.01, 0.01], player_hp=70, player_max_hp=80), CONFIG
    )
    assert budget == 32
    assert details["level"] == "low"


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (SearchBudgetInputs([0.5, 0.5], 80, 80), 128),
        (SearchBudgetInputs([0.9, 0.1], 20, 80), 128),
        (SearchBudgetInputs([0.9, 0.1], 10, 80), 256),
        (SearchBudgetInputs([0.9, 0.1], 80, 80, 0.2), 128),
    ],
)
def test_adaptive_budget_escalates_uncertain_or_risky_states(inputs, expected):
    budget, _ = adaptive_search_budget(inputs, CONFIG)
    assert budget == expected


def test_normalized_entropy_and_puct_prior_behavior():
    assert normalized_policy_entropy([1.0, 0.0]) == 0.0
    assert normalized_policy_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert puct_score(
        mean_value=0.2, prior=0.8, parent_visits=16, child_visits=1, c_puct=1.5
    ) > puct_score(
        mean_value=0.2, prior=0.2, parent_visits=16, child_visits=1, c_puct=1.5
    )


def test_search_edge_keeps_risk_and_resource_statistics_separate_from_q():
    edge = SearchEdgeStats(prior=0.4)
    edge.update(SearchOutcome(-0.2, 0.0, 60, 0, 0, False, "learned", "d1"))
    edge.update(SearchOutcome(-1.0, 0.5, 20, 1, 3, True, "terminal", "d2"))
    summary = edge.summary(cvar_alpha=0.5)
    assert summary["visits"] == 2
    assert summary["mean_value"] == pytest.approx(-0.6)
    assert summary["lower_tail_cvar"] == pytest.approx(-1.0)
    assert summary["death_probability"] == pytest.approx(0.25)
    assert summary["end_hp_mean"] == pytest.approx(40.0)
    assert summary["determinization_count"] == 2
    assert summary["leaf_sources"] == {"learned": 1, "terminal": 1}


def test_lower_tail_cvar_and_root_score_are_risk_averse():
    assert lower_tail_cvar([1.0, 0.8, 0.6, -1.0], 0.25) == -1.0
    score = risk_adjusted_root_score(
        mean_value=0.5, lower_tail_value=-0.5, mean_weight=0.4, cvar_weight=0.6
    )
    assert score == pytest.approx(-0.1)


def test_root_visit_threshold_rejects_one_sample_outliers_but_keeps_smoke_fallback():
    assert root_visit_threshold([11, 9, 1], minimum_visits=2, minimum_fraction_of_max=0.2) == 3
    assert root_visit_threshold([1, 1, 1], minimum_visits=2, minimum_fraction_of_max=0.2) == 1


def test_root_coverage_budget_repeats_every_action_before_puct():
    effective, forced = root_coverage_budget(
        4, legal_action_count=6, minimum_visits_per_action=2
    )
    assert effective == 12
    assert forced == 12
    assert [
        forced_root_action_index(
            simulation,
            legal_action_count=6,
            forced_coverage_simulations=forced,
        )
        for simulation in range(13)
    ] == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, None]


def test_root_coverage_preserves_larger_requested_budget():
    assert root_coverage_budget(
        32, legal_action_count=6, minimum_visits_per_action=2
    ) == (32, 12)


def test_forced_root_coverage_pairs_every_action_in_the_same_worlds():
    assert [
        paired_root_determinization_index(
            simulation,
            legal_action_count=3,
            forced_coverage_simulations=6,
            determinization_count=4,
        )
        for simulation in range(12)
    ] == [0, 0, 0, 1, 1, 1, 2, 3, 0, 1, 2, 3]


def test_edge_summary_balances_worlds_before_root_risk_scoring():
    edge = SearchEdgeStats(prior=0.5)
    for _ in range(3):
        edge.update(SearchOutcome(1.0, 0.0, 70, 0, 0, False, "learned", "common"))
    edge.update(SearchOutcome(-1.0, 1.0, 0, 0, 0, True, "terminal", "rare"))
    summary = edge.summary(cvar_alpha=0.5)
    assert summary["mean_value"] == pytest.approx(0.5)
    assert summary["determinization_mean_value"] == pytest.approx(0.0)
    assert summary["determinization_lower_tail_cvar"] == pytest.approx(-1.0)
    assert summary["determinization_death_probability"] == pytest.approx(0.5)
    assert summary["determinization_values"]["common"]["terminal_fraction"] == 0.0
    assert summary["determinization_values"]["rare"]["leaf_sources"] == {"terminal": 1}


def test_search_teacher_record_preserves_root_and_per_world_evidence():
    candidate = {"candidate_id": "end", "candidate_index": 0, "action_type": "end_turn"}
    report = {
        "requested_budget": 32,
        "effective_budget": 32,
        "root_determinization_count": 8,
        "root_selection_rule": "visit_count_then_risk",
        "policy_candidate": candidate,
        "search_selected_candidate": candidate,
        "chosen_candidate": candidate,
        "policy_fallback": {"enabled": True, "applied": False, "reason": "search_matches_policy"},
        "actions": [{
            "candidate": candidate,
            "prior": 1.0,
            "visits": 32,
            "visit_policy_probability": 1.0,
            "mean_value": -0.25,
            "value_std": 0.1,
            "lower_tail_cvar": -0.4,
            "determinization_mean_value": -0.25,
            "determinization_lower_tail_cvar": -0.4,
            "death_probability": 0.0,
            "end_hp_mean": 60.0,
            "end_hp_std": 2.0,
            "potion_spent_mean": 0.0,
            "max_hp_delta_mean": 0.0,
            "terminal_fraction": 0.25,
            "mean_search_depth": 3.0,
            "max_search_depth": 5,
            "leaf_sources": {"independent_state_value_leaf": 24, "exact_engine_terminal": 8},
            "determinization_count": 8,
            "determinization_values": {"world-0": {"visits": 4, "mean_value": -0.2}},
            "selection_score": -0.3,
            "selection_eligible": True,
        }],
    }
    root = {
        "observation_version": "combat-observation-0.1.0",
        "action_version": "combat-action-0.1.0",
        "act": 2,
        "floor": 21,
        "observation": {"global": {"hp": 70}},
        "candidates": [candidate],
    }
    record = build_search_teacher_record(
        root, report, search_seed=7, model_version="combat-policy-transformer-0.2.0"
    )
    schema = json.loads(
        (ROOT / "schemas" / "combat_search_teacher_v0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(record)
    assert record["schema_version"] == "combat-search-teacher-0.1.0"
    assert len(record["root_fingerprint"]) == 64
    assert record["actions"][0]["determinization_count"] == 8
    assert record["actions"][0]["determinization_values"]["world-0"]["visits"] == 4
