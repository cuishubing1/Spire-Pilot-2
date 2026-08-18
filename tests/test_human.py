import hashlib
import json

import pytest

import sts2_dataset.human as human
from sts2_dataset.human import HumanRecordingError, audit_human_recording, read_and_verify_recording, recover_recording


def _append(path, record_type, payload, sequence, previous):
    record = {
        "payload": payload,
        "prev_record_sha256": previous,
        "record_type": record_type,
        "sequence": sequence,
        "timestamp_utc": "2026-08-04T00:00:00Z",
    }
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
    record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return record["record_sha256"]


def _partial(path, schema="human-live-0.1.0"):
    previous = _append(path, "recorder_start", {
        "schema_version": schema, "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": "a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52"},
    }, 0, "0" * 64)
    _append(path, "run_start", {"run_id": "human-test"}, 1, previous)


def test_human_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "run.jsonl.partial"
    _partial(path)
    text = path.read_text(encoding="utf-8").replace("human-test", "human-hack")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(HumanRecordingError, match="hash mismatch"):
        read_and_verify_recording(path, require_complete=False)


def test_recover_human_creates_sealed_copy_without_mutating_source(tmp_path):
    source = tmp_path / "run.jsonl.partial"
    _partial(source)
    before = source.read_bytes()
    destination = tmp_path / "run.jsonl.recovered"
    result = recover_recording(source, destination)
    assert result["status"] == "PASS"
    assert source.read_bytes() == before
    assert read_and_verify_recording(destination)[-1]["record_type"] == "run_end"


def test_import_and_validate_complete_human_recording(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()
    path = source / "run.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.1.0", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "human-test"}, 1, previous)
    observation = {
        "phase": "map_select", "capture_quality": "complete",
        "run": {"act": 1, "total_floor": 0, "room_type": "Map"},
        "legal_actions": [{"action_id": "select_map_node", "args": {"coord": {"col": 1, "row": 0}}}],
    }
    previous = _append(path, "decision", {
        "run_id": "human-test", "step_id": 0, "phase": "map_select",
        "observation": observation,
        "action": {"action_id": "select_map_node", "args": {"coord": {"col": 1, "row": 0}}},
        "capture_quality": "complete", "policy_id": "human_v1",
    }, 2, previous)
    _append(path, "run_end", {
        "run_id": "human-test", "reason": "game_ended", "won": False,
        "observation": {"phase": "game_over", "run": {"act": 1, "total_floor": 1}},
    }, 3, previous)
    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    manifest = human.import_human_recordings(source)
    assert manifest["episode_count"] == 1
    assert manifest["transition_count"] == 1
    assert human.validate_human_dataset()["status"] == "PASS"


def test_human_import_is_incremental_and_idempotent(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()

    def write_run(run_id, floor):
        path = source / f"{run_id}.jsonl"
        previous = _append(path, "recorder_start", {
            "schema_version": "human-live-0.1.0", "actor_id": "anon-test",
            "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                     "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
        }, 0, "0" * 64)
        previous = _append(path, "run_start", {"run_id": run_id}, 1, previous)
        observation = {
            "phase": "combat_play", "capture_quality": "complete",
            "run": {"act": 1, "total_floor": floor, "room_type": "Monster"},
            "legal_actions": [{"action_id": "end_turn", "args": {}}],
        }
        previous = _append(path, "decision", {
            "run_id": run_id, "phase": "combat_play", "observation": observation,
            "action": {"action_id": "end_turn", "args": {}},
            "capture_quality": "complete", "content_scope": "base_game",
        }, 2, previous)
        _append(path, "run_end", {"run_id": run_id, "reason": "abandoned", "won": False}, 3, previous)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    write_run("incremental-1", 1)
    first = human.import_human_recordings(source)
    assert first["new_episode_count"] == 1
    write_run("incremental-2", 2)
    second = human.import_human_recordings(source)
    assert second["episode_count"] == 2
    assert second["transition_count"] == 2
    assert second["new_episode_count"] == 1
    assert second["skipped_episode_count"] == 1
    third = human.import_human_recordings(source)
    assert third["episode_count"] == 2
    assert third["new_episode_count"] == 0
    assert third["skipped_episode_count"] == 2


def test_v02_rollback_is_not_exported_as_normal_transition(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()
    path = source / "sl-run.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.2.0", "recorder_version": "0.2.0", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "sl-test", "attempt_id": 0}, 1, previous)
    obs0 = {"phase": "combat_play", "capture_quality": "complete",
            "run": {"act": 1, "total_floor": 5, "room_type": "Monster"},
            "legal_actions": [{"action_id": "end_turn", "args": {}}]}
    obs1 = {**obs0, "combat": {"round": 2}}
    previous = _append(path, "decision", {"run_id": "sl-test", "step_id": 2, "attempt_id": 0,
        "phase": "combat_play", "observation": obs0, "action": {"action_id": "end_turn", "args": {}},
        "capture_quality": "complete", "policy_id": "human_v2"}, 2, previous)
    previous = _append(path, "decision", {"run_id": "sl-test", "step_id": 3, "attempt_id": 0,
        "phase": "combat_play", "observation": obs1, "action": {"action_id": "end_turn", "args": {}},
        "capture_quality": "complete", "policy_id": "human_v2"}, 3, previous)
    previous = _append(path, "resume", {"run_id": "sl-test", "from_attempt_id": 0, "to_attempt_id": 1,
        "from_decision_sequence": 3}, 4, previous)
    previous = _append(path, "rollback", {"run_id": "sl-test", "rollback_id": 1,
        "from_attempt_id": 0, "to_attempt_id": 1, "from_decision_sequence": 3,
        "rollback_target_sequence": 2, "rollback_target_attempt_id": 0,
        "discarded_decision_range": [2, 3], "match_quality": "exact", "match_confidence": 1.0,
        "canonical_boundary": "resolved", "room_key": "1:5:Monster::"}, 5, previous)
    previous = _append(path, "decision", {"run_id": "sl-test", "step_id": 6, "attempt_id": 1,
        "phase": "combat_play", "observation": obs0, "action": {"action_id": "end_turn", "args": {}},
        "capture_quality": "complete", "policy_id": "human_v2"}, 6, previous)
    _append(path, "run_end", {"run_id": "sl-test", "reason": "game_ended", "won": False,
        "attempt_id": 1, "rollback_count": 1, "observation": {"phase": "game_over"}}, 7, previous)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    manifest = human.import_human_recordings(source)
    assert manifest["rollback_count"] == 1
    rows = human.pq.read_table(human.HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()
    assert [row["is_canonical"] for row in rows] == [False, False, True]
    assert rows[1]["termination"] == "rollback"
    assert json.loads(rows[1]["next_observation_json"]) is None
    assert human.validate_human_dataset()["bad_rollback_links"] == 0


def test_v02_unmatched_resume_quarantines_both_attempts(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()
    path = source / "unmatched.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.2.0", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "unmatched-test", "attempt_id": 0}, 1, previous)
    obs = {"phase": "map_select", "capture_quality": "complete",
           "run": {"act": 1, "total_floor": 2, "room_type": "Map"},
           "legal_actions": [{"action_id": "select_map_node", "args": {}}]}
    previous = _append(path, "decision", {"run_id": "unmatched-test", "attempt_id": 0,
        "phase": "map_select", "observation": obs,
        "action": {"action_id": "select_map_node", "args": {}}, "capture_quality": "complete"}, 2, previous)
    previous = _append(path, "resume", {"run_id": "unmatched-test", "from_attempt_id": 0,
        "to_attempt_id": 1, "from_decision_sequence": 2}, 3, previous)
    previous = _append(path, "resume_unmatched", {"run_id": "unmatched-test", "rollback_id": 1,
        "from_attempt_id": 0, "to_attempt_id": 1, "from_decision_sequence": 2,
        "rollback_target_sequence": None, "discarded_decision_range": None,
        "match_quality": "unmatched", "match_confidence": 0.0,
        "canonical_boundary": "quarantine", "room_key": "1:2:Map::"}, 4, previous)
    previous = _append(path, "decision", {"run_id": "unmatched-test", "attempt_id": 1,
        "phase": "map_select", "observation": obs,
        "action": {"action_id": "select_map_node", "args": {}}, "capture_quality": "complete"}, 5, previous)
    _append(path, "run_end", {"run_id": "unmatched-test", "reason": "abandoned", "won": False,
        "observation": {"phase": "game_over"}}, 6, previous)
    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    human.import_human_recordings(source)
    rows = human.pq.read_table(human.HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()
    assert [row["is_canonical"] for row in rows] == [False, False]
    assert [row["boundary_status"] for row in rows] == ["quarantine", "quarantine"]
    assert rows[0]["termination"] == "resume_unmatched"
    assert json.loads(rows[0]["next_observation_json"]) is None


def test_v021_mod_content_isolated_and_victory_inferred(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()
    path = source / "modded-win.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.2.1", "recorder_version": "0.2.1", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
        "environment": {
            "has_content_mods": True,
            "loaded_mods": [{"id": "Watcher", "assembly": "Watcher", "defines_game_entities": True}],
        },
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "modded-win", "attempt_id": 0}, 1, previous)
    legal = [{"action_id": "end_turn", "args": {}}]
    for sequence, scope, quality in [
        (2, "base_game", "complete"),
        (3, "modded", "complete"),
        (4, "unknown", "partial"),
    ]:
        observation = {
            "phase": "combat_play", "capture_quality": quality,
            "run": {"act": 3, "total_floor": 48, "room_type": "Monster"},
            "legal_actions": legal,
        }
        previous = _append(path, "decision", {
            "run_id": "modded-win", "attempt_id": 0, "phase": "combat_play",
            "observation": observation, "action": {"action_id": "end_turn", "args": {}},
            "capture_quality": quality, "content_scope": scope, "policy_id": "human_v2",
        }, sequence, previous)
    _append(path, "run_end", {
        "run_id": "modded-win", "reason": "game_ended", "won": None,
        "observation": {"phase": "game_over", "player": {"hp": 88},
                        "run": {"act": 3, "total_floor": 48}},
    }, 5, previous)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    manifest = human.import_human_recordings(source)
    assert manifest["schema_version"] == "human-dataset-0.3.0"
    episodes = human.pq.read_table(human.HUMAN_DATASET_ROOT / "episodes.parquet").to_pylist()
    rows = human.pq.read_table(human.HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()
    assert episodes[0]["victory"] is True
    assert episodes[0]["victory_source"] == "inferred_game_ended_positive_hp"
    assert episodes[0]["environment_scope"] == "modded"
    assert [row["is_training_eligible"] for row in rows] == [True, False, False]
    assert [row["strict_vanilla_eligible"] for row in rows] == [False, False, False]
    assert [row["exclusion_reason"] for row in rows] == [None, "mod_content", "capture_partial"]
    assert human.validate_human_dataset()["bad_training_eligibility"] == 0


def test_v022_combat_fields_survive_raw_import(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()
    path = source / "combat-v022.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.2.2", "recorder_version": "0.2.2", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "combat-v022", "attempt_id": 0}, 1, previous)
    observation = {
        "phase": "combat_play", "capture_quality": "complete",
        "run": {"act": 1, "total_floor": 4, "room_type": "Monster"},
        "player": {"hp": 61, "block": 3, "gold": 99},
        "combat": {
            "energy": 3,
            "player_powers": [{"id": "POWER.STRENGTH", "amount": 2}],
            "draw_pile": [{"id": "CARD.DEFEND_IRONCLAD", "count": 3}],
            "discard_pile": [], "exhaust_pile": [],
            "hand": [{"id": "CARD.STRIKE_IRONCLAD", "instance_id": "hand:0:abc", "cost": 1,
                      "stats": {"damage": 8},
                      "damage_by_target": [{"target_index": 0, "damage": 8, "hits": 1,
                                             "total_damage": 8}]}],
            "enemies": [{"id": "MONSTER.CULTIST", "hp": 42, "intends_attack": True,
                         "intent": [{"type": "Attack", "damage": 6, "hits": 1,
                                     "total_damage": 6}]}],
            "intent_capture_complete": True,
        },
        "legal_actions": [{"action_id": "end_turn", "args": {}}],
    }
    previous = _append(path, "decision", {
        "run_id": "combat-v022", "attempt_id": 0, "phase": "combat_play",
        "observation": observation, "action": {"action_id": "end_turn", "args": {}},
        "capture_quality": "complete", "content_scope": "base_game", "policy_id": "human_v2",
    }, 2, previous)
    _append(path, "run_end", {"run_id": "combat-v022", "reason": "abandoned", "won": False,
        "observation": {"phase": "game_over"}}, 3, previous)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    human.import_human_recordings(source)
    row = human.pq.read_table(human.HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()[0]
    restored = json.loads(row["observation_json"])
    assert restored["combat"]["enemies"][0]["intent"][0]["total_damage"] == 6
    assert restored["combat"]["hand"][0]["damage_by_target"][0]["damage"] == 8
    assert restored["combat"]["draw_pile"][0]["count"] == 3
    assert human.validate_human_dataset()["status"] == "PASS"


def test_v030_extended_semantic_actions_survive_import(tmp_path, monkeypatch):
    source = tmp_path / "inbox"
    source.mkdir()
    path = source / "actions-v030.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.3.0", "recorder_version": "0.3.0", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "actions-v030", "attempt_id": 0}, 1, previous)
    decisions = [
        ("bundle_select", "select_bundle", {"bundle_index": 1, "card_ids": ["CARD.BASH"]}),
        ("relic_select", "choose_relic", {"relic_index": 1, "relic_id": "RELIC.ANCHOR",
                                            "relic_instance_id": "relic_choice:1:abc"}),
        ("relic_select", "skip", {}),
        ("combat_play", "discard_potion", {"potion_instance_id": "potion:0:abc", "potion_id": "POTION.FIRE"}),
        ("reward_select", "select_reward", {"reward_index": 0, "reward_type": "Gold", "reward_id": "Gold"}),
        ("reward_select", "proceed", {}),
        ("treasure", "open_treasure", {}),
        ("treasure", "select_treasure_relic", {"relic_index": 0, "relic_id": "RELIC.ANCHOR"}),
        ("treasure", "skip_treasure_relic", {}),
    ]
    for offset, (phase, action_id, args) in enumerate(decisions, 2):
        observation = {
            "phase": phase, "capture_quality": "complete",
            "run": {"act": 1, "total_floor": offset, "room_type": phase},
            "legal_actions": [{"action_id": action_id, "args": args}],
        }
        previous = _append(path, "decision", {
            "run_id": "actions-v030", "attempt_id": 0, "phase": phase,
            "observation": observation, "action": {"action_id": action_id, "args": args},
            "capture_quality": "complete", "content_scope": "base_game", "policy_id": "human_v2",
        }, offset, previous)
    _append(path, "run_end", {"run_id": "actions-v030", "reason": "abandoned", "won": False,
        "observation": {"phase": "game_over"}}, len(decisions) + 2, previous)

    audit = audit_human_recording(path)
    assert audit["status"] == "PASS"
    assert audit["decision_count"] == len(decisions)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    manifest = human.import_human_recordings(source)
    assert manifest["transition_count"] == len(decisions)
    rows = human.pq.read_table(human.HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()
    restored = [json.loads(row["action_json"]) for row in rows]
    assert [(row["action_id"], row["args"]) for row in restored] == [
        (action_id, args) for _, action_id, args in decisions
    ]
    assert all(row["capture_quality"] == "complete" for row in rows)
    coverage = json.loads((human.HUMAN_DATASET_ROOT / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["action_counts"]["select_reward"] == 1
    assert coverage["action_arg_error_counts"] == {}
    assert human.validate_human_dataset()["status"] == "PASS"


def test_v030_missing_stable_action_args_are_quarantined():
    decision = {
        "capture_quality": "complete",
        "action": {"action_id": "use_potion", "args": {"potion_id": "POTION.FIRE"}},
        "observation": {
            "legal_actions": [{"action_id": "use_potion", "args": {"potion_instance_id": "potion:0:abc"}}]
        },
    }
    assert human._derived_capture_quality(decision) == "complete"
    assert human._derived_capture_quality(decision, strict_action_args=True) == "partial_action_args"


def test_v031_live_schema_is_accepted(tmp_path):
    path = tmp_path / "v031.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.3.1", "recorder_version": "0.3.1", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "v031", "attempt_id": 0}, 1, previous)
    previous = _append(path, "decision", {
        "run_id": "v031", "attempt_id": 0, "phase": "combat_play",
        "observation": {"phase": "combat_play", "capture_quality": "complete",
                        "legal_actions": [{"action_id": "end_turn", "args": {}}]},
        "action": {"action_id": "end_turn", "args": {}}, "capture_quality": "complete",
    }, 2, previous)
    previous = _append(path, "engine_event", {"run_id": "v031", "attempt_id": 0,
        "after_decision_sequence": 2, "event_type": "heal_requested",
        "details": {"target_id": "MONSTER.WATERFALL_GIANT", "requested_amount": 10}}, 3, previous)
    _append(path, "run_end", {"run_id": "v031", "reason": "abandoned", "won": False,
        "observation": {"phase": "game_over"}}, 4, previous)
    records = human.read_and_verify_recording(path)
    assert records[0]["payload"]["schema_version"] == "human-live-0.3.1"
    assert records[3]["record_type"] == "engine_event"


def test_v032_run_context_is_preserved_in_episode(tmp_path, monkeypatch):
    path = tmp_path / "v032.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.3.2", "recorder_version": "0.3.2", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {
        "run_id": "v032", "attempt_id": 0,
        "run_context": {"capture_quality": "complete", "seed": "TEST-SEED", "seed_numeric": 123,
                        "character_ids": ["CHARACTER.IRONCLAD"], "ascension": 7, "game_mode": "Standard",
                        "act_ids": ["ACT.OVERGROWTH"], "modifier_ids": [], "badge_ids": [],
                        "should_save": True, "daily_time": None},
    }, 1, previous)
    previous = _append(path, "decision", {
        "run_id": "v032", "attempt_id": 0, "phase": "combat_play",
        "observation": {"phase": "combat_play", "capture_quality": "complete",
                        "legal_actions": [{"action_id": "end_turn", "args": {}}]},
        "action": {"action_id": "end_turn", "args": {}}, "capture_quality": "complete",
        "content_scope": "base_game",
    }, 2, previous)
    _append(path, "run_end", {"run_id": "v032", "reason": "abandoned", "won": False,
        "observation": {"phase": "game_over"}}, 3, previous)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    manifest = human.import_human_recordings(path)
    episode = human.pq.read_table(human.HUMAN_DATASET_ROOT / "episodes.parquet").to_pylist()[0]
    assert manifest["schema_version"] == "human-dataset-0.3.0"
    assert episode["seed"] == "TEST-SEED"
    assert episode["seed_numeric"] == "123"
    assert episode["character_ids_json"] == '["CHARACTER.IRONCLAD"]'
    assert episode["ascension"] == 7
    assert episode["game_mode"] == "Standard"
    assert episode["run_context_quality"] == "complete"


def test_v040_native_audit_state_stays_out_of_training_observation(tmp_path, monkeypatch):
    path = tmp_path / "v040.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.4.0", "recorder_version": "0.4.0", "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "v040", "attempt_id": 0}, 1, previous)
    observation = {
        "phase": "combat_play", "capture_quality": "complete",
        "player": {"deck": [{"id": "CARD.STRIKE", "enchantment": {
            "id": "ENCHANTMENT.GLAM", "amount": 1, "status": "Normal"}}]},
        "legal_actions": [{"action_id": "end_turn", "args": {}}],
    }
    audit_state = {
        "schema_version": "native-model-state-0.1.0", "capture_quality": "complete",
        "combat": {"draw_pile_ordered": [{"id": "CARD.BASH"}]},
    }
    previous = _append(path, "decision", {
        "run_id": "v040", "attempt_id": 0, "phase": "combat_play",
        "observation": observation, "audit_state": audit_state,
        "action": {"action_id": "end_turn", "args": {}}, "capture_quality": "complete",
        "content_scope": "base_game", "policy_id": "human_v3",
    }, 2, previous)
    _append(path, "run_end", {"run_id": "v040", "reason": "abandoned", "won": False,
        "observation": {"phase": "game_over"}, "audit_state": audit_state}, 3, previous)

    monkeypatch.setattr(human, "HUMAN_RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(human, "HUMAN_DATASET_ROOT", tmp_path / "data" / "dataset")
    human.import_human_recordings(path)
    episode = human.pq.read_table(human.HUMAN_DATASET_ROOT / "episodes.parquet").to_pylist()[0]
    transition = human.pq.read_table(human.HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()[0]
    exported_observation = json.loads(transition["observation_json"])
    assert episode["policy_id"] == "human_v3"
    assert exported_observation["player"]["deck"][0]["enchantment"]["id"] == "ENCHANTMENT.GLAM"
    assert "audit_state" not in exported_observation


def test_v040_missing_or_leaked_native_state_is_quarantined():
    decision = {
        "capture_quality": "complete",
        "observation": {"capture_quality": "complete", "legal_actions": [
            {"action_id": "end_turn", "args": {}}
        ]},
        "action": {"action_id": "end_turn", "args": {}},
    }
    assert human._derived_capture_quality(
        decision, strict_action_args=True, require_native_state=True
    ) == "partial_native_state"
    decision["audit_state"] = {
        "schema_version": "native-model-state-0.1.0", "capture_quality": "complete"
    }
    decision["observation"]["draw_pile_ordered"] = []
    assert human._derived_capture_quality(
        decision, strict_action_args=True, require_native_state=True
    ) == "partial_native_state"


def test_v041_schema_telemetry_and_lineage_contract(tmp_path):
    path = tmp_path / "v041.jsonl"
    previous = _append(path, "recorder_start", {
        "schema_version": "human-live-0.4.1", "recorder_version": "0.4.1-internal",
        "actor_id": "anon-test",
        "game": {"expected_game_version": "0.107.1", "expected_build": "23811903",
                 "assembly_sha256": human.EXPECTED_ASSEMBLY_SHA256},
        "environment": {"loaded_mods": [], "has_content_mods": False},
        "hook_manifest": [{"type": "T", "method": "M", "phase": "combat_play",
                           "action_id": "play_card", "required": True, "status": "installed"}],
        "privacy": {"contains_account_id": False, "contains_input_events": False},
    }, 0, "0" * 64)
    previous = _append(path, "run_start", {"run_id": "v041", "attempt_id": 0}, 1, previous)
    card = {
        "id": "CARD.STRIKE", "instance_id": "hand:0:abc", "lineage_id": "card-lineage",
        "lineage_quality": "stable_object", "engine_object_ref": "abc",
        "energy_cost": {"current": 1}, "state_schema": "native-model-state-0.1.0",
    }
    observation = {
        "phase": "combat_play", "capture_quality": "complete", "player": {"deck": []},
        "combat": {"hand": [card]},
        "legal_actions": [{"action_id": "play_card", "args": {
            "card_instance_id": "hand:0:abc", "card_lineage_id": "card-lineage"}}],
    }
    audit_state = {"schema_version": "native-model-state-0.1.0", "capture_quality": "complete"}
    previous = _append(path, "decision", {
        "run_id": "v041", "step_id": 2, "attempt_id": 0, "phase": "combat_play",
        "observation": observation, "audit_state": audit_state,
        "action": {"action_id": "play_card", "args": {
            "card_instance_id": "hand:0:abc", "card_lineage_id": "card-lineage"}},
        "capture_quality": "complete", "content_scope": "base_game", "policy_id": "human_v4",
        "commit_status": "method_returned", "telemetry": {
            "capture_ms": 1.25, "observation_ms": 0.5, "audit_ms": 0.4,
            "encode_and_classify_ms": 0.2, "fingerprint_ms": 0.15, "writer_queue_depth": 0},
    }, 2, previous)
    _append(path, "run_end", {"run_id": "v041", "reason": "abandoned", "won": False}, 3, previous)

    records = read_and_verify_recording(path)
    assert len(records) == 4
    assert human._derived_capture_quality(records[2]["payload"], strict_action_args=True,
                                          require_native_state=True, require_v041=True) == "complete"

    broken = json.loads(json.dumps(records[2]["payload"]))
    del broken["observation"]["combat"]["hand"][0]["lineage_id"]
    assert human._derived_capture_quality(broken, strict_action_args=True,
                                          require_native_state=True, require_v041=True) == "partial_v041_contract"


def test_required_card_selection_cancel_is_repaired_without_hiding_real_partial_data():
    card = {
        "id": "CARD.STRIKE", "instance_id": "card_select:0:abc",
        "lineage_id": "card-lineage", "engine_object_ref": "abc",
        "source_kind": "base_game", "source_mod_id": "sts2",
    }
    decision = {
        "phase": "card_select", "capture_quality": "partial", "content_scope": "unknown",
        "commit_status": "method_returned",
        "telemetry": {"capture_ms": 1.0},
        "observation": {
            "phase": "card_select", "capture_quality": "partial",
            "player": {"character_source": {"source_kind": "base_game", "source_mod_id": "sts2"}},
            "card_select": {
                "cards": [card], "selected_cards": [], "can_skip": False,
                "min_select": 1, "max_select": 1,
            },
            "legal_actions": [{"action_id": "skip_card_selection", "args": {}}],
        },
        "audit_state": {
            "schema_version": "native-model-state-0.1.0", "capture_quality": "complete"
        },
        "action": {"action_id": "skip_card_selection", "args": {}},
    }
    quality = human._derived_capture_quality(
        decision, strict_action_args=True, require_native_state=True, require_v041=True
    )
    assert quality == "complete"
    assert human._derived_content_scope(decision, quality) == "base_game"

    broken = json.loads(json.dumps(decision))
    broken["observation"]["card_select"]["cards"] = []
    assert human._derived_capture_quality(broken) == "partial"
