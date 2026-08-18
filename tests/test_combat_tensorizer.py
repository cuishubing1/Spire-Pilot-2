import numpy as np
import pytest

from sts2_dataset.combat_tensorizer import CombatTensorizerV0, collate_combat_numpy


VOCAB = {
    "tensorizer_version": "combat-tensorizer-0.2.0",
    "numeric_feature_dim": 64,
    "categorical_feature_dim": 64,
    "entity_types": ["<PAD>", "global", "hand", "draw", "discard", "exhaust", "enemy", "relic", "potion", "power", "orb"],
    "action_types": ["<PAD>", "play_card", "use_potion", "discard_potion", "end_turn"],
    "target_kinds": ["<PAD>", "none", "self", "enemy", "all_enemies"],
    "entity_identity": ["<PAD>", "<UNK>", "global", "card:CARD.STRIKE", "enemy:MONSTER.CULTIST"],
    "encounter_identity": ["<PAD>", "<UNK>", "encounter:MONSTER.CULTIST"],
}


def _sample(transition_id="t0"):
    return {
        "transition_id": transition_id, "combat_id": "combat-0", "split": "train", "label_index": 0,
        "encounter_signature": "encounter:MONSTER.CULTIST",
        "act": 1, "floor": 1, "label_action_type": "play_card",
        "observation": {
            "global": {"hp": 50, "energy": 3, "room_type": "Monster"},
            "hand": [{"id": "CARD.STRIKE", "entity_ref": "card-0", "cost": 1}],
            "piles": {"draw": [], "discard": [], "exhaust": []},
            "enemies": [{"id": "MONSTER.CULTIST", "entity_ref": "1", "hp": 40}],
            "relics": [], "potions": [], "player_powers": [], "orbs": [],
        },
        "candidates": [{
            "action_type": "play_card", "source_type": "card", "source_ref": "card-0",
            "target_kind": "enemy", "target_ref": "1",
        }],
    }


def test_tensorizer_binds_dynamic_action_sources_and_targets():
    row = CombatTensorizerV0(VOCAB).tensorize(_sample())
    assert row["entity_type"].shape == (3,)
    assert row["entity_numeric"].shape == (3, 64)
    assert row["entity_categorical"].shape == (3, 64)
    assert np.any(row["entity_categorical"][0] != 0)
    assert row["action_source"].tolist() == [1]
    assert row["action_target"].tolist() == [2]
    assert row["candidate_engine_numeric"].shape == (1, 18)
    assert row["encounter_identity"] == 2
    assert row["label"] == 0


def test_numpy_collate_pads_entities_and_candidates_with_masks():
    tensorizer = CombatTensorizerV0(VOCAB)
    first = tensorizer.tensorize(_sample("t0"))
    second_sample = _sample("t1")
    second_sample["candidates"].append({
        "action_type": "end_turn", "source_type": None, "source_ref": None,
        "target_kind": "none", "target_ref": None,
    })
    second = tensorizer.tensorize(second_sample)
    batch = collate_combat_numpy([first, second])
    assert batch["action_type"].shape == (2, 2)
    assert batch["entity_numeric"].dtype == np.float32
    assert batch["entity_categorical"].dtype == np.float32
    assert batch["action_mask"].tolist() == [[True, False], [True, True]]
    assert batch["candidate_engine_numeric"].shape == (2, 2, 18)
    assert batch["encounter_identity"].tolist() == [2, 2]


def test_tensorizer_derives_terminal_hp_fraction_from_existing_value_target():
    sample = _sample()
    sample["value_target"] = {
        "terminal_hp": 30,
        "terminal_max_hp": 75,
        "death": False,
    }
    row = CombatTensorizerV0(VOCAB).tensorize(sample)
    batch = collate_combat_numpy([row])
    assert row["target_terminal_hp_fraction"] == 0.4
    assert batch["target_terminal_hp_fraction"].tolist() == [pytest.approx(0.4)]
