import pytest

from sts2_dataset.normalize import normalize_observation, outcome_delta


CONFIG = {
    "dataset_version": "test",
    "game_version": "v0.107.1",
    "steam_build_id": "23811903",
    "sts2_dll_sha256": "A" * 64,
    "sts2_cli_commit": "b" * 40,
    "sts2_cli_protocol": "0.2.0",
}


def player(hp=80, gold=99):
    return {
        "id": "CHARACTER.IRONCLAD",
        "name": "Ironclad",
        "hp": hp,
        "max_hp": 80,
        "block": 0,
        "gold": gold,
        "deck": [{"id": "CARD.STRIKE", "name": "Strike"}],
        "relics": [{"id": "RELIC.BURNING_BLOOD", "name": "Burning Blood"}],
        "potions": [],
    }


def test_normalization_uses_visible_allow_list_and_stable_ids():
    raw = {
        "type": "decision",
        "decision": "map_select",
        "context": {"act": 1, "floor": 0, "room_type": "Map"},
        "player": player(),
        "choices": [{"col": 1, "row": 0, "type": "Monster"}],
        "secret_rng_state": "must not pass",
    }
    result = normalize_observation(raw, config=CONFIG, run_id="r", step_id=0, audit_ref=None)
    assert "secret_rng_state" not in str(result.agent_observation)
    assert result.agent_observation["player"]["display_name"] == "Ironclad"
    assert result.agent_observation["player"]["deck"][0]["instance_id"].startswith("deck/")
    assert result.legal_actions[0]["action"] == "select_map_node"


def test_unknown_phase_fails_closed():
    with pytest.raises(ValueError, match="Unknown decision phase"):
        normalize_observation(
            {"type": "decision", "decision": "new_screen", "player": player()},
            config=CONFIG,
            run_id="r",
            step_id=0,
            audit_ref=None,
        )


def test_outcome_is_fact_vector_not_scalar_reward():
    raw_a = {
        "type": "decision", "decision": "map_select", "context": {"act": 1, "floor": 1},
        "player": player(hp=70, gold=100), "choices": [{"col": 1, "row": 2, "type": "Monster"}],
    }
    raw_b = {
        "type": "decision", "decision": "map_select", "context": {"act": 1, "floor": 2},
        "player": player(hp=65, gold=115), "choices": [{"col": 1, "row": 3, "type": "Monster"}],
    }
    a = normalize_observation(raw_a, config=CONFIG, run_id="r", step_id=0, audit_ref=None)
    b = normalize_observation(raw_b, config=CONFIG, run_id="r", step_id=1, audit_ref=None)
    delta = outcome_delta(a, b)
    assert delta["hp_delta"] == -5
    assert delta["gold_delta"] == 15
    assert "reward" not in delta

