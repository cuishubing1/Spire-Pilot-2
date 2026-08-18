import json

import pytest
from jsonschema import Draft202012Validator


torch = pytest.importorskip("torch")

from sts2_dataset.combat_directive import CombatDirectiveV0
from sts2_dataset.combat_model import CombatPolicyConfig, CombatPolicyTransformer
from sts2_dataset.combat_tensorizer import CombatTensorizerV0
from sts2_dataset.combat_tool import CombatToolV0
from sts2_dataset.constants import ROOT


VOCAB = {
    "tensorizer_version": "combat-tensorizer-0.2.0",
    "numeric_feature_dim": 64,
    "categorical_feature_dim": 64,
    "entity_types": [
        "<PAD>", "global", "hand", "draw", "discard", "exhaust",
        "enemy", "relic", "potion", "power", "orb",
    ],
    "action_types": ["<PAD>", "play_card", "use_potion", "discard_potion", "end_turn"],
    "target_kinds": ["<PAD>", "none", "self", "enemy", "all_enemies"],
    "entity_identity": [
        "<PAD>", "<UNK>", "global", "card:CARD.STRIKE",
        "enemy:MONSTER.A", "enemy:MONSTER.B", "potion:POTION.FIRE",
    ],
}


def _sample():
    return {
        "transition_id": "tool:0",
        "combat_id": "tool-combat",
        "split": "online",
        "act": 2,
        "floor": 20,
        "label_index": 0,
        "label_action_type": "play_card",
        "observation_version": "combat-observation-0.1.0",
        "action_version": "combat-action-0.1.0",
        "observation": {
            "global": {"hp": 60, "max_hp": 80, "block": 0, "energy": 3, "room_type": "Elite"},
            "hand": [{
                "id": "CARD.STRIKE", "entity_ref": "card:0", "cost": 1,
                "stats": {"damage": 6},
                "damage_by_target": [
                    {"target_index": 0, "damage": 6},
                    {"target_index": 1, "damage": 6},
                ],
            }],
            "piles": {"draw": [], "discard": [], "exhaust": []},
            "enemies": [
                {"id": "MONSTER.A", "entity_ref": "enemy:0", "index": 0, "hp": 20, "max_hp": 20},
                {"id": "MONSTER.B", "entity_ref": "enemy:1", "index": 1, "hp": 30, "max_hp": 30},
            ],
            "relics": [],
            "potions": [{"id": "POTION.FIRE", "entity_ref": "potion:0"}],
            "player_powers": [],
            "orbs": [],
        },
        "candidates": [
            {
                "candidate_id": "attack-a", "candidate_index": 0,
                "action_type": "play_card", "source_type": "card", "source_id": "CARD.STRIKE",
                "source_ref": "card:0", "target_kind": "enemy", "target_ref": "enemy:0",
                "target_index": 0,
            },
            {
                "candidate_id": "attack-b", "candidate_index": 1,
                "action_type": "play_card", "source_type": "card", "source_id": "CARD.STRIKE",
                "source_ref": "card:0", "target_kind": "enemy", "target_ref": "enemy:1",
                "target_index": 1,
            },
            {
                "candidate_id": "potion-a", "candidate_index": 2,
                "action_type": "use_potion", "source_type": "potion", "source_id": "POTION.FIRE",
                "source_ref": "potion:0", "target_kind": "enemy", "target_ref": "enemy:0",
                "target_index": 0,
            },
            {
                "candidate_id": "end", "candidate_index": 3,
                "action_type": "end_turn", "source_type": None, "source_id": None,
                "source_ref": None, "target_kind": "none", "target_ref": None,
                "target_index": None,
            },
        ],
    }


def _tool():
    config = CombatPolicyConfig.from_vocabulary(
        VOCAB,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        resource_value_heads=True,
        decision_value_scale=0.0,
    )
    model = CombatPolicyTransformer(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.eval()
    return CombatToolV0(model, CombatTensorizerV0(VOCAB), device="cpu")


def _state_value_tool():
    config = CombatPolicyConfig.from_vocabulary(
        VOCAB,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        resource_value_heads=False,
        state_value_head=True,
        decision_value_scale=0.0,
    )
    model = CombatPolicyTransformer(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.eval()
    return CombatToolV0(model, CombatTensorizerV0(VOCAB), device="cpu")


def test_tool_default_response_matches_schema_and_has_score_breakdown():
    response = _tool().decide(_sample())
    schema = json.loads(
        (ROOT / "schemas" / "combat_tool_v0.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(response)
    assert response["status"] == "ok"
    assert response["model_provenance"]["checkpoint_sha256"] is None
    assert response["chosen_action"]["candidate_id"] == "attack-a"
    assert response["chosen"]["score_breakdown"]["policy_logit"] == 0.0
    assert response["chosen"]["engine_preview"]["total_damage"] == 6.0
    assert response["chosen"]["resource_prediction"]["max_hp_delta"] == 0.0
    assert response["chosen"]["resource_prediction"]["max_hp_delta_source"] == (
        "engine_public_growth_cap"
    )
    assert response["predicted_risk"]["source"] == "candidate_resource_head"
    assert response["capabilities"]["candidate_resource_prediction"] == "diagnostic_on_policy"
    assert response["capabilities"]["objective_reranking"] == "inactive"
    assert response["directive_effects"]["objective_reranking_applied"] is False
    assert response["search_request"]["mode"] == "policy_only"
    assert response["search_request"]["search_executed"] is False


def test_state_value_only_checkpoint_still_exposes_replan_risk():
    sample = _sample()
    sample["observation"]["enemies"][0]["intent"] = [
        {"is_attack": True, "damage": 12, "hits": 1}
    ]
    response = _state_value_tool().decide(sample)
    schema = json.loads(
        (ROOT / "schemas" / "combat_tool_v0.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(response)
    assert response["chosen"]["resource_prediction"] is None
    assert response["predicted_risk"]["source"] == "state_value_head"
    assert response["predicted_risk"]["death_probability"] == pytest.approx(0.5)
    assert response["predicted_risk"]["hp_loss_fraction"] == pytest.approx(0.5)
    assert response["predicted_risk"]["immediate_hp_loss_fraction"] == pytest.approx(0.15)
    assert response["predicted_risk"]["immediate_hp_loss_source"] == "visible_intent_rule"
    assert response["capabilities"]["visible_end_turn_damage"] == "exact_rule"
    assert "predicted_death_risk" in response["request_replan"]["reasons"]


def test_target_bias_changes_action_without_changing_policy_prior():
    directive = CombatDirectiveV0.from_dict({
        "action_preferences": {"target_biases": {"enemy:1": 2.0}},
        "replan_policy": {"normalized_entropy": 1.0, "top_probability_gap": 0.0},
    })
    response = _tool().decide(_sample(), directive=directive)
    assert response["chosen_action"]["candidate_id"] == "attack-b"
    assert response["chosen"]["policy_probability"] == 0.25
    assert response["chosen"]["score_breakdown"]["target_bias"] == 2.0


def test_inactive_objective_override_is_reported_instead_of_silently_ignored():
    response = _tool().decide(
        _sample(),
        directive={"objective": {"hp_loss_weight": 9.0}},
    )
    assert response["directive_effects"] == {
        "objective_overrides_requested": ["hp_loss_weight"],
        "objective_reranking_applied": False,
        "ignored_objective_overrides": ["hp_loss_weight"],
        "ignored_objective_reason": "decision_value_scale_zero",
    }


def test_explicit_positive_value_scale_marks_reranking_experimental():
    response = _tool().decide(
        _sample(),
        directive={
            "objective": {
                "decision_value_scale": 0.25,
                "hp_loss_weight": 9.0,
            }
        },
    )
    assert response["capabilities"]["objective_reranking"] == "experimental_on_policy"
    assert response["directive_effects"]["objective_reranking_applied"] is True
    assert response["directive_effects"]["ignored_objective_overrides"] == []


def test_resource_budget_and_trusted_mechanic_fact_filter_candidates():
    directive = CombatDirectiveV0.from_dict({
        "resource_policy": {
            "max_potion_uses": 0,
            "potion_uses_so_far": 0,
            "preserve_potion_ids": [],
        },
        "action_preferences": {"candidate_biases": {"attack-b": 3.0}},
        "replan_policy": {"normalized_entropy": 1.0, "top_probability_gap": 0.0},
    })
    response = _tool().decide(
        _sample(),
        directive=directive,
        mechanic_facts=[{
            "candidate_id": "attack-b",
            "mechanic_id": "would_trigger_split",
            "hard_forbidden": True,
            "source": "engine_rule",
        }],
    )
    by_id = {row["candidate"]["candidate_id"]: row for row in response["ranked_actions"]}
    assert not by_id["potion-a"]["eligible"]
    assert "directive_potion_budget_exhausted" in by_id["potion-a"]["exclusion_reasons"]
    assert not by_id["attack-b"]["eligible"]
    assert by_id["attack-b"]["mechanic_ids"] == ["would_trigger_split"]
    assert response["chosen_action"]["candidate_id"] == "attack-a"


def test_all_actions_forbidden_falls_back_and_requests_replan():
    directive = CombatDirectiveV0.from_dict({
        "action_preferences": {
            "forbidden_candidate_ids": ["attack-a", "attack-b", "potion-a", "end"]
        }
    })
    response = _tool().decide(_sample(), directive=directive)
    assert response["status"] == "directive_conflict_fallback"
    assert response["request_replan"]["required"]
    assert "directive_conflict_fallback" in response["request_replan"]["reasons"]
    assert response["search_request"]["recommended_mode"] == "one_step"


def test_typed_search_request_is_exposed_without_claiming_execution():
    directive = CombatDirectiveV0.from_dict({
        "search_policy": {
            "mode": "turn_boundary",
            "budget_class": "medium",
            "max_wall_ms": 1500,
            "determinizations": 3,
            "allow_policy_override": False,
            "mechanic_plan_id": "boss-plan-v0",
        }
    })
    response = _tool().decide(_sample(), directive=directive)
    assert response["search_request"] == {
        "mode": "turn_boundary",
        "budget_class": "medium",
        "max_wall_ms": 1500,
        "determinizations": 3,
        "allow_policy_override": False,
        "mechanic_plan_id": "boss-plan-v0",
        "recommended_mode": "turn_boundary",
        "execute_search": True,
        "search_executed": False,
        "trigger_reasons": response["request_replan"]["reasons"],
    }
