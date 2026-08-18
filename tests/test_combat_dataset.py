import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import sts2_dataset.combat_dataset as combat
from sts2_dataset.combat_difficulty import combat_difficulty_tier
from sts2_dataset.human import HumanRecordingError
from sts2_dataset.human import HUMAN_EPISODE_SCHEMA, HUMAN_TRANSITION_SCHEMA


def _episode(run_id, *, ascension=10):
    return {
        "run_id": run_id,
        "actor_id": "anon-test",
        "policy_id": "human_v4",
        "character_ids_json": '["CHARACTER.IRONCLAD"]',
        "seed": f"seed-{run_id}",
        "seed_numeric": "1",
        "ascension": ascension,
        "run_context_quality": "complete",
        "terminal_reason": "game_ended",
        "victory": True,
        "max_act": 3,
        "max_floor": 48,
        "transitions": 1,
        "partial_transitions": 0,
        "training_eligible_transitions": 1,
        "strict_vanilla_eligible_transitions": 1,
        "rollback_count": 0,
        "unmatched_resume_count": 0,
    }


def _transition(run_id, act, floor, index):
    observation = {
        "phase": "combat_play",
        "run": {
            "act": act,
            "total_floor": floor,
            "room_type": "Monster",
            "act_id": f"ACT.{act}",
            "map_coord": {"col": floor % 3, "row": floor},
            "room_model_id": "0",
        },
        "legal_actions": [{"action_id": "end_turn", "args": {}}],
    }
    return {
        "transition_id": f"{run_id}:{index}",
        "run_id": run_id,
        "step_id": index,
        "record_sequence": index + 2,
        "attempt_id": 0,
        "phase": "combat_play",
        "act": act,
        "floor": floor,
        "room_type": "Monster",
        "capture_quality": "complete",
        "commit_status": "method_returned",
        "is_canonical": True,
        "sl_contaminated": False,
        "boundary_status": "resolved",
        "termination": "continued",
        "environment_scope": "base_game",
        "content_scope": "base_game",
        "is_training_eligible": True,
        "strict_vanilla_eligible": True,
        "observation_json": json.dumps(observation, sort_keys=True),
        "legal_actions_json": '[{"action_id":"end_turn","args":{}}]',
        "action_json": '{"action_id":"end_turn","args":{}}',
        "next_observation_json": "null",
        "done": False,
        "policy_id": "human_v4",
    }


def _write_human_snapshot(root, combat_counts):
    root.mkdir(parents=True, exist_ok=True)
    episodes = []
    transitions = []
    for act, count in combat_counts.items():
        for offset in range(count):
            run_id = f"act-{act}-combat-{offset}"
            episodes.append(_episode(run_id))
            transitions.append(_transition(run_id, act, offset + 1, 0))
            transitions.append(_transition(run_id, act, offset + 1, 1))
    episodes.append(_episode("heldout-run", ascension=0))
    for act in (1, 2, 3):
        transitions.append(_transition("heldout-run", act, act * 10, act * 2))
        transitions.append(_transition("heldout-run", act, act * 10, act * 2 + 1))
    pq.write_table(pa.Table.from_pylist(episodes, schema=HUMAN_EPISODE_SCHEMA), root / "episodes.parquet")
    pq.write_table(pa.Table.from_pylist(transitions, schema=HUMAN_TRANSITION_SCHEMA), root / "transitions.parquet")
    (root / "manifest.json").write_text(
        json.dumps({"episodes": len(episodes), "transitions": len(transitions)}), encoding="utf-8"
    )


def test_test_is_grouped_by_run_while_train_validation_are_grouped_by_combat_and_incremental(
    tmp_path, monkeypatch
):
    human_root = tmp_path / "human" / "dataset"
    combat_root = tmp_path / "human" / "combat_v1"
    config_path = tmp_path / "combat.json"
    config_path.write_text(json.dumps({
        "schema_version": "combat-dataset-1.1.0",
        "character_ids": ["CHARACTER.IRONCLAD"],
        "acts": [1, 2, 3],
        "eligibility_column": "is_training_eligible",
        "split_seed": "test-seed",
        "test_run_ids": ["heldout-run"],
        "test_run_constraints": {
            "require_complete_context": True,
            "require_victory": True,
            "minimum_max_act": 3,
            "minimum_max_floor": 48,
            "required_ascensions": [0],
        },
        "splits": {"train": 0.9, "validation": 0.1},
    }), encoding="utf-8")
    monkeypatch.setattr(combat, "HUMAN_DATASET_ROOT", human_root)
    monkeypatch.setattr(combat, "COMBAT_DATASET_ROOT", combat_root)
    monkeypatch.setattr(combat, "validate_human_dataset", lambda: {"status": "PASS"})

    _write_human_snapshot(human_root, {1: 10, 2: 10, 3: 10})
    first = combat.build_combat_dataset(config_path)
    assert first["combat_count"] == 33
    assert first["test_run_ids"] == ["heldout-run"]
    assert first["test_run_leak_count"] == 0
    for act in ("1", "2", "3"):
        assert first["act_split_counts"][act]["train"]["combats"] == 9
        assert first["act_split_counts"][act]["validation"]["combats"] == 1
        assert first["act_split_counts"][act]["test"]["combats"] == 1
    before = {
        row["combat_id"]: row["split"]
        for row in pq.read_table(combat_root / "combats.parquet").to_pylist()
    }

    _write_human_snapshot(human_root, {1: 13, 2: 13, 3: 13})
    second = combat.build_combat_dataset(config_path)
    after_rows = pq.read_table(combat_root / "combats.parquet").to_pylist()
    after = {row["combat_id"]: row["split"] for row in after_rows}
    assert second["combat_count"] == 42
    assert second["new_combat_count"] == 9
    assert all(after[combat_id] == split for combat_id, split in before.items())
    assert len({(row["combat_id"], row["split"]) for row in after_rows}) == 42
    assert {
        row["split"] for row in after_rows if row["run_id"] == "heldout-run"
    } == {"test"}
    assert all(
        row["split"] != "test" for row in after_rows if row["run_id"] != "heldout-run"
    )
    validation = combat.validate_combat_dataset(config_path)
    assert validation["status"] == "PASS"
    assert validation["cross_split_combats"] == 0
    assert validation["cross_split_runs"] == 0
    assert validation["test_runs"] == 1
    assert validation["test_run_leaks"] == 0


def test_combat_difficulty_tiers_follow_enemy_stat_changes():
    assert {combat_difficulty_tier(value) for value in range(8)} == {"base_a0_a7"}
    assert combat_difficulty_tier(8) == "tough_a8"
    assert combat_difficulty_tier(9) == "deadly_a9_a10"
    assert combat_difficulty_tier(10) == "deadly_a9_a10"
    with pytest.raises(HumanRecordingError):
        combat_difficulty_tier(11)


def test_train_validation_are_stratified_by_act_and_combat_difficulty(
    tmp_path, monkeypatch
):
    human_root = tmp_path / "human" / "dataset"
    combat_root = tmp_path / "human" / "combat_v1"
    config_path = tmp_path / "combat.json"
    config_path.write_text(json.dumps({
        "schema_version": "combat-dataset-1.2.0",
        "character_ids": ["CHARACTER.IRONCLAD"],
        "acts": [1, 2, 3],
        "eligibility_column": "is_training_eligible",
        "split_seed": "difficulty-test-seed",
        "test_run_ids": ["heldout-run"],
        "test_run_constraints": {
            "require_complete_context": True,
            "require_victory": True,
            "minimum_max_act": 3,
            "minimum_max_floor": 48,
            "required_ascensions": [0],
        },
        "required_combat_difficulty_tiers": [
            "base_a0_a7", "tough_a8", "deadly_a9_a10"
        ],
        "minimum_stratum_combats_for_split_coverage": 6,
        "splits": {"train": 0.9, "validation": 0.1},
    }), encoding="utf-8")
    monkeypatch.setattr(combat, "HUMAN_DATASET_ROOT", human_root)
    monkeypatch.setattr(combat, "COMBAT_DATASET_ROOT", combat_root)
    monkeypatch.setattr(combat, "validate_human_dataset", lambda: {"status": "PASS"})

    episodes = [_episode("heldout-run", ascension=0)]
    transitions = [
        _transition("heldout-run", act, act * 10, act * 2 + offset)
        for act in (1, 2, 3)
        for offset in (0, 1)
    ]
    for ascension in (0, 8, 10):
        for act in (1, 2, 3):
            for offset in range(10):
                run_id = f"a{ascension}-act{act}-{offset}"
                episodes.append(_episode(run_id, ascension=ascension))
                transitions.append(_transition(run_id, act, offset + 1, 0))
    human_root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(episodes, schema=HUMAN_EPISODE_SCHEMA),
        human_root / "episodes.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(transitions, schema=HUMAN_TRANSITION_SCHEMA),
        human_root / "transitions.parquet",
    )
    (human_root / "manifest.json").write_text("{}", encoding="utf-8")

    manifest = combat.build_combat_dataset(config_path, rebuild=True)
    for act in ("1", "2", "3"):
        for tier in ("base_a0_a7", "tough_a8", "deadly_a9_a10"):
            counts = manifest["act_difficulty_split_counts"][act][tier]
            assert counts["train"]["combats"] == 9
            assert counts["validation"]["combats"] == 1
    validation = combat.validate_combat_dataset(config_path)
    assert validation["status"] == "PASS"
    assert validation["missing_stratum_splits"] == 0
