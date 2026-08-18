import json

import pytest
from jsonschema import Draft202012Validator

from sts2_dataset.combat_directive import (
    CandidateMechanicFactV0,
    CombatDirectiveV0,
    CombatSearchPolicyV0,
)
from sts2_dataset.constants import ROOT


def test_default_directive_round_trips_and_matches_schema():
    value = CombatDirectiveV0.default().to_dict()
    schema = json.loads(
        (ROOT / "schemas" / "combat_directive_v0.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(value)
    assert CombatDirectiveV0.from_dict(value).to_dict() == value


def test_directive_accepts_typed_objectives_constraints_and_biases():
    directive = CombatDirectiveV0.from_dict({
        "schema_version": "combat-directive-0.2.0",
        "objective": {"death_penalty": 12.0, "potion_cost": 0.05},
        "resource_policy": {
            "max_potion_uses": 1,
            "potion_uses_so_far": 0,
            "preserve_potion_ids": ["POTION.FAIRY"],
            "acceptable_hp_loss_fraction": 0.1,
        },
        "action_preferences": {
            "target_biases": {"enemy:boss": 1.25},
            "action_type_biases": {"use_potion": 0.5},
        },
        "search_policy": {
            "mode": "turn_boundary",
            "budget_class": "high",
            "max_wall_ms": 3000,
            "determinizations": 4,
            "allow_policy_override": True,
            "mechanic_plan_id": "terror-eel-burst-v0",
        },
    })
    assert directive.objective_overrides["death_penalty"] == 12.0
    assert directive.target_biases == {"enemy:boss": 1.25}
    assert directive.search_policy.mode == "turn_boundary"
    assert directive.search_policy.mechanic_plan_id == "terror-eel-burst-v0"


def test_directive_rejects_open_ended_or_untrusted_fields():
    with pytest.raises(ValueError, match="unknown combat objective"):
        CombatDirectiveV0.from_dict({"objective": {"be_careful": 1.0}})
    with pytest.raises(ValueError, match="engine_rule"):
        CandidateMechanicFactV0.from_dict({
            "candidate_id": "a0",
            "mechanic_id": "split_threshold",
            "source": "llm",
        })


def test_search_policy_is_bounded_and_legacy_directive_defaults_to_policy_only():
    legacy = CombatDirectiveV0.from_dict({"schema_version": "combat-directive-0.1.0"})
    assert legacy.search_policy == CombatSearchPolicyV0()
    assert legacy.to_dict()["schema_version"] == "combat-directive-0.2.0"
    with pytest.raises(ValueError, match="max_wall_ms"):
        CombatDirectiveV0.from_dict({"search_policy": {"max_wall_ms": 50_000}})
    with pytest.raises(ValueError, match="unsupported search mode"):
        CombatDirectiveV0.from_dict({"search_policy": {"mode": "unbounded_mcts"}})
