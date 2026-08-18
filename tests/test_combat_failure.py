from sts2_dataset.combat_failure import (
    classify_failure,
    first_search_divergence,
    state_features,
)


def test_classify_failure_includes_death_regret_and_search_regression():
    row = {
        "human": {"hp_loss": 2},
        "p2_policy": {"hp_loss": 5, "status": "combat_won"},
        "p2_one_step": {"hp_loss": 30, "status": "death"},
    }
    assert classify_failure(row) == [
        "one_step_death",
        "search_introduced_death",
        "high_human_regret",
        "search_regression",
    ]


def test_state_features_accepts_training_and_online_intent_shapes():
    features = state_features(
        {
            "player_hp": 30,
            "player_block": 4,
            "energy": 2,
            "round": 3,
            "hand": ["CARD.A", "CARD.B"],
            "enemies": [
                {
                    "hp": 20,
                    "max_hp": 40,
                    "intents": [{"type": "Attack", "damage": 5, "hits": 2}],
                }
            ],
        },
        max_hp=60,
        legal_count=7,
    )
    assert features["hp_ratio"] == 0.5
    assert features["enemy_hp_ratio"] == 0.5
    assert features["incoming_damage"] == 10
    assert features["legal_count"] == 7


def test_first_search_divergence_reports_score_margin():
    before = {"decision": "combat_play", "round": 1, "player_hp": 40, "hand": ["CARD.A"]}
    policy_candidate = {
        "action_type": "play_card",
        "source_id": "CARD.A",
        "search_equivalence_key": "policy",
    }
    search_candidate = {
        "action_type": "end_turn",
        "source_id": None,
        "search_equivalence_key": "search",
    }
    row = {
        "p2_policy": {
            "steps": [
                {
                    "before": before,
                    "chosen_candidate": policy_candidate,
                    "policy_probability": 0.8,
                }
            ]
        },
        "p2_one_step": {
            "steps": [
                {
                    "step": 0,
                    "before": before,
                    "chosen_candidate": search_candidate,
                    "lookahead": {
                        "root_policy_entropy": 0.3,
                        "evaluations": [
                            {
                                "candidate": policy_candidate,
                                "selection_score": 0.1,
                                "policy_probability": 0.8,
                            },
                            {
                                "candidate": search_candidate,
                                "selection_score": 0.25,
                                "policy_probability": 0.2,
                            },
                        ],
                    },
                }
            ]
        },
    }
    divergence = first_search_divergence(row)
    assert divergence is not None
    assert divergence["policy_action"]["source_id"] == "CARD.A"
    assert divergence["search_action"]["action_type"] == "end_turn"
    assert divergence["selection_score_margin"] == 0.15
