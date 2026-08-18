from copy import deepcopy

from sts2_dataset.combat_online import (
    candidate_to_headless_command,
    first_card_select_candidate,
    headless_state_to_model_sample,
    visible_intent_end_turn_hp_loss,
)


def _state():
    return {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1, "ascension": 3, "room_type": "Monster"},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 1,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "piles": {"draw": [{"id": "CARD.DEFEND_IRONCLAD", "count": 1}], "discard": [], "exhaust": []},
        "hand": [{
            "index": 0, "id": "CARD.STRIKE_IRONCLAD", "cost": 1, "type": "Attack",
            "rarity": "Basic", "can_play": True, "target_type": "AnyEnemy", "stats": {"damage": 6},
            "damage_by_target": [{"target_index": 0, "damage": 6}],
        }],
        "enemies": [{
            "index": 0, "id": "MONSTER.SHRINKER_BEETLE", "combat_id": "0", "hp": 39,
            "max_hp": 39, "block": 0, "intents": [{"type": "Attack", "damage": 7}],
            "intends_attack": True,
        }],
        "player": {
            "id": "CHARACTER.IRONCLAD", "hp": 80, "max_hp": 80, "block": 0, "gold": 99,
            "relics": [{"index": 0, "id": "RELIC.BURNING_BLOOD", "vars": {"Heal": 6}}],
            "potions": [],
        },
    }


def test_headless_state_projects_to_existing_model_contract():
    sample = headless_state_to_model_sample(_state(), transition_id="online:0", combat_id="online")
    assert sample["encounter_signature"] == "encounter:MONSTER.SHRINKER_BEETLE"
    assert sample["observation"]["global"]["ascension"] == 3
    assert sample["observation"]["enemies"][0]["id"] == "MONSTER.SHRINKER_BEETLE"
    assert [candidate["action_type"] for candidate in sample["candidates"]] == ["play_card", "end_turn"]
    assert candidate_to_headless_command(sample["candidates"][0]) == {
        "cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0},
    }


def test_online_encounter_signature_uses_cached_combat_start_identity():
    state = _state()
    state["context"]["encounter_signature"] = (
        "encounter:MONSTER.FUZZY_WURM_CRAWLER+MONSTER.SHRINKER_BEETLE"
    )
    sample = headless_state_to_model_sample(
        state, transition_id="online:cached", combat_id="online"
    )
    assert sample["encounter_signature"] == state["context"]["encounter_signature"]


def test_online_encounter_signature_accepts_explicit_frozen_identity():
    state = _state()
    sample = headless_state_to_model_sample(
        state,
        transition_id="online:explicit",
        combat_id="online",
        encounter_signature="encounter:MONSTER.ORIGINAL_BOSS",
    )
    assert sample["encounter_signature"] == "encounter:MONSTER.ORIGINAL_BOSS"


def test_headless_search_metadata_merges_only_equivalent_card_instances():
    state = _state()
    duplicate_strike = deepcopy(state["hand"][0])
    duplicate_strike["index"] = 1
    discounted_strike = deepcopy(state["hand"][0])
    discounted_strike["index"] = 3
    discounted_strike["cost"] = 0
    defend = {
        "index": 2,
        "id": "CARD.DEFEND_IRONCLAD",
        "cost": 1,
        "type": "Skill",
        "rarity": "Basic",
        "can_play": True,
        "target_type": "Self",
        "stats": {"block": 5},
    }
    state["hand"] = [
        state["hand"][0],
        duplicate_strike,
        defend,
        discounted_strike,
    ]
    second_enemy = deepcopy(state["enemies"][0])
    second_enemy["index"] = 1
    second_enemy["combat_id"] = "1"
    state["enemies"].append(second_enemy)

    candidates = headless_state_to_model_sample(
        state, transition_id="online:duplicates", combat_id="online"
    )["candidates"]
    strikes = [
        candidate
        for candidate in candidates
        if candidate.get("source_id") == "CARD.STRIKE_IRONCLAD"
    ]
    block = next(
        candidate
        for candidate in candidates
        if candidate.get("source_id") == "CARD.DEFEND_IRONCLAD"
    )

    assert len(strikes) == 6
    assert strikes[0]["candidate_id"] != strikes[1]["candidate_id"]
    key_counts: dict[str, int] = {}
    for strike in strikes:
        key = strike["search_equivalence_key"]
        key_counts[key] = key_counts.get(key, 0) + 1
        assert strike["search_category"] == "card_attack"
    assert sorted(key_counts.values()) == [1, 1, 2, 2]
    assert block["search_category"] == "card_block"
    assert block["search_equivalence_key"] not in key_counts


def test_visible_intent_end_turn_hp_loss_uses_total_damage_and_block():
    observation = headless_state_to_model_sample(
        _state(), transition_id="online:0", combat_id="online"
    )["observation"]
    observation["global"]["block"] = 3
    observation["enemies"].append({
        "id": "MONSTER.SECOND",
        "hp": 20,
        "intent": [{"type": "Attack", "damage": 2, "hits": 3, "total_damage": 6}],
    })
    estimate = visible_intent_end_turn_hp_loss(observation)
    assert estimate == {
        "incoming_damage": 13.0,
        "block": 3.0,
        "hp_loss": 10.0,
        "hp_loss_fraction": 0.125,
    }


def test_visible_intent_end_turn_hp_loss_rejects_unknown_attack_damage():
    observation = headless_state_to_model_sample(
        _state(), transition_id="online:0", combat_id="online"
    )["observation"]
    observation["enemies"][0]["intent"] = [{"type": "Attack", "is_attack": True}]
    assert visible_intent_end_turn_hp_loss(observation) is None


def test_card_select_fallback_explicitly_chooses_first_card():
    state = {
        "decision": "card_select",
        "min_select": 0,
        "max_select": 1,
        "cards": [
            {"index": 0, "id": "CARD.DISINTEGRATION"},
            {"index": 1, "id": "CARD.MIND_ROT"},
        ],
    }
    candidate = first_card_select_candidate(state)
    assert candidate["source_id"] == "CARD.DISINTEGRATION"
    assert candidate["source_type"] == "card_selection_first"
    assert candidate_to_headless_command(candidate) == {
        "cmd": "action",
        "action": "select_cards",
        "args": {"indices": "0"},
    }


def test_card_select_fallback_prefers_recorded_human_choice():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "cards": [
            {"index": 0, "id": "CARD.DISINTEGRATION"},
            {"index": 1, "id": "CARD.MIND_ROT"},
        ],
    }
    candidate = first_card_select_candidate(
        state, preferred_card_ids=("CARD.MIND_ROT",)
    )
    assert candidate["source_id"] == "CARD.MIND_ROT"
    assert candidate["source_type"] == "card_selection_human_match"
    assert candidate_to_headless_command(candidate)["args"] == {"indices": "1"}
