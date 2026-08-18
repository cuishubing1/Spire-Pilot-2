from sts2_dataset.combat_engine_features import (
    CANDIDATE_ENGINE_FEATURE_NAMES,
    candidate_engine_feature_vector,
    candidate_preview_features,
    combat_future_max_hp_growth_cap,
    exact_transition_features,
    ground_future_max_hp_delta,
)


def test_candidate_preview_features_uses_target_specific_engine_damage():
    observation = {
        "global": {"energy": 3, "max_hp": 80, "block": 0},
        "hand": [{
            "entity_ref": "hand:0",
            "cost": 1,
            "energy_cost": {"current": 1, "costs_x": False},
            "stats": {"damage": 6},
            "damage_by_target": [{
                "target_combat_id": "enemy:0", "damage": 9, "hits": 2, "total_damage": 18,
            }],
        }],
        "enemies": [{"entity_ref": "enemy:0", "index": 0, "hp": 17, "block": 0}],
    }
    features = candidate_preview_features(observation, {
        "action_type": "play_card", "source_ref": "hand:0", "target_ref": "enemy:0",
    })
    assert features["total_damage"] == 18
    assert features["hit_count"] == 2
    assert features["preview_lethal"] is True


def test_candidate_engine_vector_exposes_exact_damage_block_and_visible_loss():
    observation = {
        "global": {"energy": 3, "max_energy": 3, "max_hp": 80, "block": 2},
        "hand": [{
            "entity_ref": "hand:0",
            "cost": 1,
            "energy_cost": {"current": 1, "costs_x": False},
            "stats": {"block": 8},
        }],
        "enemies": [{
            "entity_ref": "enemy:0",
            "hp": 20,
            "max_hp": 40,
            "block": 0,
            "intent": [{"damage": 12, "hits": 1, "total_damage": 12, "is_attack": True}],
        }],
    }
    vector = candidate_engine_feature_vector(observation, {
        "action_type": "play_card", "source_ref": "hand:0", "target_ref": None,
    })
    values = dict(zip(CANDIDATE_ENGINE_FEATURE_NAMES, vector, strict=True))
    assert values["energy_cost_fraction"] == 1 / 3
    assert values["block_gain_fraction"] == 0.1
    assert values["visible_end_turn_hp_loss_fraction"] == 0.125
    assert values["block_adjusted_end_turn_hp_loss_fraction"] == 0.025


def test_exact_transition_features_measures_engine_result():
    before = {
        "decision": "combat_play", "round": 1, "energy": 3,
        "player": {"hp": 70, "max_hp": 80, "block": 0, "potions": [{"id": "P"}]},
        "hand": [{"id": "CARD.FEED"}],
        "enemies": [{"hp": 10}],
    }
    after = {
        "decision": "combat_play", "round": 1, "energy": 2,
        "player": {"hp": 70, "max_hp": 83, "block": 0, "potions": [{"id": "P"}]},
        "hand": [],
        "enemies": [],
    }
    features = exact_transition_features(before, after, {"action_type": "play_card"})
    assert features["max_hp_delta"] == 3
    assert features["enemy_hp_loss"] == 10
    assert features["enemies_killed"] == 1
    assert features["energy_delta"] == -1


def test_exact_transition_features_does_not_treat_missing_death_state_as_enemy_kill():
    before = {
        "decision": "combat_play",
        "round": 2,
        "energy": 3,
        "hand": [{"id": "CARD.STRIKE_IRONCLAD"}],
        "player": {"hp": 5, "max_hp": 80, "block": 0, "potions": []},
        "enemies": [{"hp": 140, "max_hp": 140}],
    }
    after = {
        "decision": "game_over",
        "victory": False,
        "player": {"hp": 0, "max_hp": 80, "potions": []},
    }

    features = exact_transition_features(before, after, {"action_type": "end_turn"})

    assert features["hp_loss"] == 5.0
    assert features["enemy_state_observed"] is False
    assert features["enemy_hp_loss"] == 0.0
    assert features["enemies_killed"] == 0.0
    assert features["energy_observed"] is False
    assert features["hand_observed"] is False
    assert features["energy_delta"] == 0.0
    assert features["hand_count_delta"] == 0.0


def test_future_max_hp_growth_is_zero_without_a_combat_growth_source():
    observation = {"hand": [], "piles": {"draw": [], "discard": []}, "relics": []}
    result = ground_future_max_hp_delta(0.24, observation)
    assert result["positive_growth_cap"] == 0.0
    assert result["grounded_prediction"] == 0.0
    assert ground_future_max_hp_delta(-0.3, observation)["grounded_prediction"] == 0.0


def test_feed_and_chosen_cheese_define_public_growth_upper_bound():
    observation = {
        "hand": [{"id": "CARD.FEED", "stats": {"maxhp": 3}}],
        "piles": {"draw": [], "discard": []},
        "relics": [
            {
                "id": "RELIC.CHOSEN_CHEESE",
                "dynamic_vars": {"maxhp": 1},
                "visible_state": {"is_used_up": False, "is_melted": False},
            },
            {"id": "RELIC.STRAWBERRY", "dynamic_vars": {"maxhp": 7}},
        ],
    }
    capability = combat_future_max_hp_growth_cap(observation)
    assert capability["positive_growth_cap"] == 4.0
    assert {row["id"] for row in capability["sources"]} == {
        "CARD.FEED", "RELIC.CHOSEN_CHEESE"
    }
    assert ground_future_max_hp_delta(8.0, observation)["grounded_prediction"] == 4.0
