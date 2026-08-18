import json

import pyarrow as pa
import pyarrow.parquet as pq

import sts2_dataset.combat_contract as contract


def _observation():
    return {
        "phase": "combat_play",
        "run": {"act": 1, "total_floor": 4, "ascension": 10, "room_type": "Monster"},
        "player": {
            "character_id": "CHARACTER.IRONCLAD", "hp": 50, "max_hp": 80, "block": 0, "gold": 99,
            "relics": [],
            "potions": [{
                "id": "POTION.FLEX", "instance_id": "potion:0:p", "target_type": "Self", "vars": {},
                "source_assembly": "sts2",
            }],
        },
        "combat": {
            "turn": 1, "round": 1, "turn_phase": "Play", "energy": 3, "max_energy": 3, "stars": 0,
            "hand": [{
                "id": "CARD.STRIKE", "instance_id": "hand:0:c", "lineage_id": "card:c",
                "target_type": "AnyEnemy", "type": "Attack", "cost": 1,
                "engine_object_ref": "private-engine-ref", "display_name": "Strike",
            }],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [], "player_powers": [], "orbs": [],
            "enemies": [{
                "id": "MONSTER.CULTIST", "combat_id": "1", "index": 0, "hp": 40, "max_hp": 40,
                "block": 0, "intends_attack": True,
                "intent": [{"type": "Attack", "damage": 6, "hits": 1, "total_damage": 6}],
                "powers": [],
            }],
        },
        "legal_actions": [
            {"action_id": "play_card", "args": {"card_instance_id": "hand:0:c", "target_index": 0}},
            {"action_id": "use_potion", "args": {"potion_instance_id": "potion:0:p"}},
            {"action_id": "discard_potion", "args": {"potion_instance_id": "potion:0:p"}},
            {"action_id": "end_turn", "args": {}},
        ],
    }


def test_combat_v0_resolves_legacy_self_target_potion_and_strips_metadata():
    observation = _observation()
    actual = {"action_id": "use_potion", "args": {
        "potion_instance_id": "potion:0:p", "target_combat_id": "0",
        "target_id": "CHARACTER.IRONCLAD",
    }}
    candidates, label = contract.build_action_candidates_v0(observation, actual)
    assert label == 1
    assert candidates[label]["target_kind"] == "self"
    projected = contract.project_observation_v0(observation)
    assert projected["hand"][0]["entity_ref"] == "hand:0:c"
    assert "engine_object_ref" not in str(projected)
    assert "source_assembly" not in str(projected)
    assert "display_name" not in str(projected)


def test_combat_v0_resolves_enemy_target_by_stable_combat_id():
    observation = _observation()
    actual = {"action_id": "play_card", "args": {
        "card_instance_id": "hand:0:c", "target_combat_id": "1", "target_id": "MONSTER.CULTIST",
    }}
    candidates, label = contract.build_action_candidates_v0(observation, actual)
    assert label == 0
    assert candidates[label]["target_index"] == 0
    assert candidates[label]["target_ref"] == "1"


def test_combat_model_example_builder_appends_only_new_transitions(tmp_path, monkeypatch):
    combat_root = tmp_path / "combat_v1"
    model_root = combat_root / "model_v0"
    combat_root.mkdir()
    monkeypatch.setattr(contract, "COMBAT_DATASET_ROOT", combat_root)
    monkeypatch.setattr(contract, "COMBAT_MODEL_ROOT", model_root)
    monkeypatch.setattr(contract, "validate_combat_dataset", lambda config: {"status": "PASS"})

    def row(index):
        observation = _observation()
        return {
            "transition_id": f"run:{index}", "combat_id": f"combat-{index}", "split": "train",
            "act": 1, "floor": index + 1,
            "observation_json": json.dumps(observation),
            "action_json": json.dumps({"action_id": "end_turn", "args": {}}),
            "source_transition_sha256": f"sha-{index}",
        }

    def combat_row(index):
        return {
            "combat_id": f"combat-{index}",
            "encounter_signature": "encounter:MONSTER.CULTIST",
        }

    source_path = combat_root / "transitions.parquet"
    combat_path = combat_root / "combats.parquet"
    pq.write_table(pa.Table.from_pylist([row(0)]), source_path)
    pq.write_table(pa.Table.from_pylist([combat_row(0)]), combat_path)
    (combat_root / "manifest.json").write_text('{"version":1}', encoding="utf-8")
    first = contract.build_combat_model_examples()
    assert first["sample_count"] == 1
    pq.write_table(pa.Table.from_pylist([row(0), row(1)]), source_path)
    pq.write_table(
        pa.Table.from_pylist([combat_row(0), combat_row(1)]), combat_path
    )
    (combat_root / "manifest.json").write_text('{"version":2}', encoding="utf-8")
    second = contract.build_combat_model_examples()
    assert second["sample_count"] == 2
    assert second["new_sample_count"] == 1
    decoded = list(contract.iter_combat_model_samples("train"))
    assert len(decoded) == 2
    assert decoded[0]["candidates"][decoded[0]["label_index"]]["action_type"] == "end_turn"
