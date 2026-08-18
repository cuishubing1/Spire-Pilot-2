from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator

from .combat_dataset import COMBAT_CONFIG_PATH, COMBAT_DATASET_ROOT, validate_combat_dataset
from .combat_difficulty import combat_difficulty_tier
from .combat_encounter import encounter_signature_from_observation
from .human import HumanRecordingError
from .util import canonical_json, load_json, sha256_file, utc_now, write_json_atomic

OBSERVATION_VERSION = "combat-observation-0.1.0"
ACTION_VERSION = "combat-action-0.1.0"
SAMPLE_VERSION = "combat-model-sample-0.2.0"
SUPPORTED_ACTIONS = {"play_card", "use_potion", "discard_potion", "end_turn"}
MODEL_METADATA_KEYS = {
    "display_name", "display_text", "engine_object_ref", "source_assembly", "source_kind", "source_mod_id",
    "state_schema", "projection_version", "persistent_state_capture_error", "persistent_state_capture_quality",
}

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "schemas" / "combat_model_v0.schema.json"
COMBAT_MODEL_ROOT = COMBAT_DATASET_ROOT / "model_v0"
_OBSERVATION_VALIDATOR = Draft202012Validator(json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))

COMBAT_MODEL_SAMPLE_SCHEMA = pa.schema(
    [
        ("transition_id", pa.string()),
        ("combat_id", pa.string()),
        ("split", pa.string()),
        ("ascension", pa.int32()),
        ("combat_difficulty_tier", pa.string()),
        ("encounter_signature", pa.string()),
        ("act", pa.int32()),
        ("floor", pa.int32()),
        ("observation_version", pa.string()),
        ("action_version", pa.string()),
        ("observation_v0_json", pa.large_string()),
        ("candidates_json", pa.large_string()),
        ("candidate_count", pa.int32()),
        ("label_index", pa.int32()),
        ("label_action_type", pa.string()),
        ("source_transition_sha256", pa.string()),
    ]
)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(child) for key, child in value.items()
            if key not in MODEL_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_value(child) for child in value]
    return copy.deepcopy(value)


def _copy_fields(source: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _sanitize_value(source[name]) for name in names if name in source}


def _public_state_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, dict) or item.get("visibility", "public") != "public":
            continue
        rows.append(_copy_fields(item, (
            "normalized_key", "value_type", "int_value", "float_value", "bool_value", "string_value",
            "lifecycle", "visibility",
        )))
    return rows


def _project_card(card: dict[str, Any], *, bind_instance: bool) -> dict[str, Any]:
    projected = _copy_fields(card, (
        "id", "index", "count", "upgrade_level", "max_upgrade_level", "floor_added", "type", "rarity",
        "target_type", "tags", "keywords", "cost", "energy_cost", "star_cost", "star_cost_state",
        "can_play", "playability", "stats", "dynamic_vars", "damage_by_target", "runtime_flags",
        "enchantment", "affliction",
    ))
    projected["persistent_state"] = _public_state_rows(card.get("persistent_state"))
    projected["runtime_state"] = _public_state_rows(card.get("runtime_state"))
    if bind_instance:
        projected["entity_ref"] = card.get("instance_id")
        projected["lineage_ref"] = card.get("lineage_id")
    return projected


def _project_power(power: dict[str, Any]) -> dict[str, Any]:
    return _copy_fields(power, ("id", "index", "amount", "target_type", "vars"))


def _project_relic(relic: dict[str, Any]) -> dict[str, Any]:
    projected = _copy_fields(relic, (
        "id", "index", "stack_count", "status", "floor_added", "visible_state", "dynamic_vars",
    ))
    projected["persistent_state"] = _public_state_rows(relic.get("persistent_state"))
    projected["runtime_state"] = _public_state_rows(relic.get("runtime_state"))
    return projected


def _project_potion(potion: dict[str, Any]) -> dict[str, Any]:
    projected = _copy_fields(potion, ("id", "index", "amount", "target_type", "vars"))
    projected["entity_ref"] = potion.get("instance_id")
    return projected


def _project_enemy(enemy: dict[str, Any]) -> dict[str, Any]:
    intents = []
    for intent in enemy.get("intent") or []:
        if isinstance(intent, dict):
            intents.append(_copy_fields(intent, (
                "type", "damage", "hits", "repeats", "total_damage", "is_attack",
            )))
    return {
        **_copy_fields(enemy, ("id", "index", "hp", "max_hp", "block", "intends_attack")),
        "entity_ref": enemy.get("combat_id"),
        "intent": intents,
        "powers": [_project_power(value) for value in enemy.get("powers") or [] if isinstance(value, dict)],
    }


def project_observation_v0(observation: dict[str, Any]) -> dict[str, Any]:
    if observation.get("phase") != "combat_play":
        raise HumanRecordingError("Combat Observation V0 requires phase=combat_play")
    run = observation.get("run") or {}
    player = observation.get("player") or {}
    combat = observation.get("combat") or {}
    if not isinstance(run, dict) or not isinstance(player, dict) or not isinstance(combat, dict):
        raise HumanRecordingError("combat observation is missing run/player/combat objects")
    piles = {}
    for output_name, source_name in (
        ("draw", "draw_pile"), ("discard", "discard_pile"), ("exhaust", "exhaust_pile")
    ):
        cards = [_project_card(value, bind_instance=False) for value in combat.get(source_name) or []
                 if isinstance(value, dict)]
        piles[output_name] = sorted(cards, key=canonical_json)
    return {
        "schema_version": OBSERVATION_VERSION,
        "global": {
            "character_id": player.get("character_id"),
            "act": int(run.get("act") or 0),
            "floor": int(run.get("total_floor") or 0),
            "ascension": int(run.get("ascension") or 0),
            "room_type": run.get("room_type"),
            "turn": int(combat.get("turn") or 0),
            "round": int(combat.get("round") or 0),
            "turn_phase": combat.get("turn_phase"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "gold": player.get("gold"),
            "energy": combat.get("energy"),
            "max_energy": combat.get("max_energy"),
            "stars": combat.get("stars"),
            "orb_slots": combat.get("orb_slots"),
            "draw_pile_count": combat.get("draw_pile_count"),
            "discard_pile_count": combat.get("discard_pile_count"),
            "exhaust_pile_count": combat.get("exhaust_pile_count"),
        },
        "hand": [_project_card(value, bind_instance=True) for value in combat.get("hand") or []
                 if isinstance(value, dict)],
        "piles": piles,
        "enemies": [_project_enemy(value) for value in combat.get("enemies") or [] if isinstance(value, dict)],
        "relics": [_project_relic(value) for value in player.get("relics") or [] if isinstance(value, dict)],
        "potions": [_project_potion(value) for value in player.get("potions") or [] if isinstance(value, dict)],
        "player_powers": [_project_power(value) for value in combat.get("player_powers") or []
                          if isinstance(value, dict)],
        "orbs": [_copy_fields(value, ("id", "index", "passive", "evoke")) for value in combat.get("orbs") or []
                 if isinstance(value, dict)],
    }


def _target_kind(source_target_type: Any, *, action_type: str, has_enemy: bool) -> str:
    if has_enemy:
        return "enemy"
    if action_type == "discard_potion":
        return "none"
    normalized = str(source_target_type or "").lower()
    if normalized == "self":
        return "self"
    if normalized in {"allenemies", "all_enemies"}:
        return "all_enemies"
    return "none"


def _normalize_action(action: dict[str, Any], observation: dict[str, Any], *, candidate_index: int) -> dict[str, Any]:
    action_type = str(action.get("action_id") or "")
    if action_type not in SUPPORTED_ACTIONS:
        raise HumanRecordingError(f"unsupported Combat Action V0 action: {action_type!r}")
    args = action.get("args") or {}
    if not isinstance(args, dict):
        raise HumanRecordingError(f"combat action {action_type!r} has non-object args")
    combat = observation.get("combat") or {}
    player = observation.get("player") or {}
    enemies = [value for value in combat.get("enemies") or [] if isinstance(value, dict)]
    hand = [value for value in combat.get("hand") or [] if isinstance(value, dict)]
    potions = [value for value in player.get("potions") or [] if isinstance(value, dict)]

    source_type = None
    source_ref = None
    source_id = None
    source_index = None
    source_target_type = None
    if action_type == "play_card":
        source_type = "card"
        source_ref = args.get("card_instance_id")
        matches = [(index, value) for index, value in enumerate(hand) if value.get("instance_id") == source_ref]
    elif action_type in {"use_potion", "discard_potion"}:
        source_type = "potion"
        source_ref = args.get("potion_instance_id")
        matches = [(index, value) for index, value in enumerate(potions) if value.get("instance_id") == source_ref]
    else:
        matches = []
    if source_type is not None:
        if len(matches) != 1:
            raise HumanRecordingError(f"{action_type} source {source_ref!r} does not uniquely bind to an entity")
        source_index, source = matches[0]
        source_id = source.get("id")
        source_target_type = source.get("target_type")

    target_ref = args.get("target_combat_id")
    target_index = args.get("target_index")
    target_id = args.get("target_id")
    enemy = None
    if target_ref not in {None, "", "0", 0}:
        matches = [(index, value) for index, value in enumerate(enemies)
                   if str(value.get("combat_id")) == str(target_ref)]
        if len(matches) == 1:
            target_index, enemy = matches[0]
    elif isinstance(target_index, int) and 0 <= target_index < len(enemies):
        enemy = enemies[target_index]
        target_ref = enemy.get("combat_id")
    if enemy is None and isinstance(target_id, str) and target_id.startswith("MONSTER."):
        matches = [(index, value) for index, value in enumerate(enemies) if value.get("id") == target_id]
        if len(matches) == 1:
            target_index, enemy = matches[0]
            target_ref = enemy.get("combat_id")
    if enemy is not None:
        target_id = enemy.get("id")
    else:
        target_ref = None
        target_index = None
        target_id = None

    target_kind = _target_kind(source_target_type, action_type=action_type, has_enemy=enemy is not None)
    semantic = {
        "action_type": action_type,
        "source_ref": source_ref,
        "target_ref": str(target_ref) if target_ref is not None else None,
    }
    candidate_id = hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()[:20]
    return {
        "candidate_index": candidate_index,
        "candidate_id": f"action-{candidate_id}",
        "action_type": action_type,
        "source_type": source_type,
        "source_ref": source_ref,
        "source_id": source_id,
        "source_index": source_index,
        "target_kind": target_kind,
        "target_ref": semantic["target_ref"],
        "target_id": target_id,
        "target_index": target_index,
        "engine_action": copy.deepcopy(action),
    }


def build_action_candidates_v0(
    observation: dict[str, Any], actual_action: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    legal_actions = observation.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise HumanRecordingError("combat observation has no legal actions")
    candidates = [
        _normalize_action(value, observation, candidate_index=index)
        for index, value in enumerate(legal_actions) if isinstance(value, dict)
    ]
    if len(candidates) != len(legal_actions):
        raise HumanRecordingError("combat legal action list contains non-object entries")
    ids = [value["candidate_id"] for value in candidates]
    if len(ids) != len(set(ids)):
        raise HumanRecordingError("Combat Action V0 produced duplicate semantic candidates")
    actual = _normalize_action(actual_action, observation, candidate_index=-1)
    matches = [
        value["candidate_index"] for value in candidates
        if value["candidate_id"] == actual["candidate_id"]
    ]
    if len(matches) != 1:
        raise HumanRecordingError(
            f"human action maps to {len(matches)} legal Combat Action V0 candidates: {actual_action}"
        )
    return candidates, matches[0]


def build_model_sample_v0(
    row: dict[str, Any], *, encounter_signature: str | None = None
) -> dict[str, Any]:
    observation = json.loads(row["observation_json"])
    actual_action = json.loads(row["action_json"])
    projected = project_observation_v0(observation)
    ascension = int(projected["global"]["ascension"])
    candidates, label_index = build_action_candidates_v0(observation, actual_action)
    return {
        "transition_id": row["transition_id"],
        "combat_id": row["combat_id"],
        "split": row["split"],
        "ascension": ascension,
        "combat_difficulty_tier": combat_difficulty_tier(ascension),
        "encounter_signature": (
            encounter_signature or encounter_signature_from_observation(observation)
        ),
        "act": int(row["act"]),
        "floor": int(row["floor"]),
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "observation_v0_json": canonical_json(projected),
        "candidates_json": canonical_json(candidates),
        "candidate_count": len(candidates),
        "label_index": label_index,
        "label_action_type": candidates[label_index]["action_type"],
        "source_transition_sha256": row["source_transition_sha256"],
    }


def _write_parquet_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=COMBAT_MODEL_SAMPLE_SCHEMA), temporary, compression="zstd")
    os.replace(temporary, path)


def build_combat_model_examples(*, rebuild: bool = False) -> dict[str, Any]:
    validate_combat_dataset(COMBAT_CONFIG_PATH)
    source_path = COMBAT_DATASET_ROOT / "transitions.parquet"
    source_rows = pq.read_table(source_path).to_pylist()
    combat_rows = pq.read_table(COMBAT_DATASET_ROOT / "combats.parquet").to_pylist()
    encounter_by_combat = {
        str(row["combat_id"]): str(row["encounter_signature"])
        for row in combat_rows
    }
    source_by_id = {row["transition_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise HumanRecordingError("combat transition source contains duplicate transition ids")
    output_path = COMBAT_MODEL_ROOT / "samples.parquet"
    manifest_path = COMBAT_MODEL_ROOT / "manifest.json"
    existing: list[dict[str, Any]] = []
    if not rebuild and manifest_path.exists():
        validate_combat_model_examples()
        existing = pq.read_table(output_path).to_pylist()
        for sample in existing:
            source = source_by_id.get(sample["transition_id"])
            if source is None:
                raise HumanRecordingError(
                    f"combat model samples are not append-only; missing {sample['transition_id']}; use --rebuild"
                )
            if (
                sample["source_transition_sha256"] != source["source_transition_sha256"]
                or sample["combat_id"] != source["combat_id"]
                or sample["split"] != source["split"]
            ):
                raise HumanRecordingError(
                    f"combat model source changed for {sample['transition_id']}; use --rebuild"
                )
    existing_ids = {row["transition_id"] for row in existing}
    new_samples = [
        build_model_sample_v0(
            row,
            encounter_signature=encounter_by_combat[str(row["combat_id"])],
        )
        for row in source_rows
        if row["transition_id"] not in existing_ids
    ]
    samples = [*existing, *new_samples]
    samples.sort(key=lambda row: (row["combat_id"], row["transition_id"]))
    _write_parquet_atomic(samples, output_path)
    split_counts = Counter(row["split"] for row in samples)
    action_counts = Counter(row["label_action_type"] for row in samples)
    act_counts = Counter(str(row["act"]) for row in samples)
    difficulty_counts = Counter(row["combat_difficulty_tier"] for row in samples)
    difficulty_split_counts = Counter(
        f"{row['combat_difficulty_tier']}:{row['split']}" for row in samples
    )
    encounter_counts = Counter(
        row["encounter_signature"]
        for row in {row["combat_id"]: row for row in samples}.values()
    )
    manifest = {
        "schema_version": SAMPLE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "generated_at": utc_now(),
        "update_mode": "full_rebuild" if rebuild or not existing else "incremental_append",
        "contract_path": str(CONTRACT_PATH),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_manifest_sha256": sha256_file(COMBAT_DATASET_ROOT / "manifest.json"),
        "sample_count": len(samples),
        "new_sample_count": len(new_samples),
        "split_counts": dict(sorted(split_counts.items())),
        "act_counts": dict(sorted(act_counts.items())),
        "combat_difficulty_counts": dict(sorted(difficulty_counts.items())),
        "combat_difficulty_split_counts": dict(sorted(difficulty_split_counts.items())),
        "encounter_signature_counts": dict(sorted(encounter_counts.items())),
        "label_action_counts": dict(sorted(action_counts.items())),
        "max_candidate_count": max(row["candidate_count"] for row in samples),
        "files": [{
            "path": str(output_path), "sha256": sha256_file(output_path), "size": output_path.stat().st_size,
        }],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def iter_combat_model_samples(split: str | None = None):
    """Yield decoded model-facing samples without introducing a PyTorch dependency."""
    path = COMBAT_MODEL_ROOT / "samples.parquet"
    if not path.exists():
        raise HumanRecordingError("combat model samples do not exist; run build-combat-examples")
    filters = [("split", "=", split)] if split is not None else None
    for row in pq.read_table(path, filters=filters).to_pylist():
        observation = json.loads(row.pop("observation_v0_json"))
        candidates = json.loads(row.pop("candidates_json"))
        yield {
            **row,
            "observation": observation,
            "candidates": candidates,
        }


def validate_combat_model_examples() -> dict[str, Any]:
    manifest_path = COMBAT_MODEL_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise HumanRecordingError("combat model sample manifest does not exist")
    manifest = load_json(manifest_path)
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise HumanRecordingError("Combat V0 contract changed; rebuild model samples")
    for entry in manifest.get("files", []):
        path = Path(entry["path"])
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            raise HumanRecordingError(f"missing or modified combat model sample file: {path}")
    rows = pq.read_table(COMBAT_MODEL_ROOT / "samples.parquet").to_pylist()
    if len(rows) != manifest.get("sample_count"):
        raise HumanRecordingError("combat model sample count does not match manifest")
    forbidden = {"audit_state", "draw_pile_ordered", *MODEL_METADATA_KEYS}
    for row in rows:
        observation = json.loads(row["observation_v0_json"])
        candidates = json.loads(row["candidates_json"])
        errors = sorted(_OBSERVATION_VALIDATOR.iter_errors(observation), key=lambda error: list(error.path))
        if errors:
            location = "/".join(str(part) for part in errors[0].absolute_path)
            raise HumanRecordingError(
                f"Combat Observation V0 schema error at {location}: {errors[0].message}"
            )
        if observation.get("schema_version") != OBSERVATION_VERSION:
            raise HumanRecordingError(f"wrong observation version in {row['transition_id']}")
        expected_tier = combat_difficulty_tier(
            int(observation["global"].get("ascension") or 0)
        )
        if row["combat_difficulty_tier"] != expected_tier:
            raise HumanRecordingError(
                f"combat difficulty tier mismatch in {row['transition_id']}"
            )
        if not str(row.get("encounter_signature") or "").startswith("encounter:"):
            raise HumanRecordingError(
                f"invalid encounter signature in {row['transition_id']}"
            )
        if forbidden.intersection(_nested_keys(observation)):
            raise HumanRecordingError(f"forbidden field leaked into Combat Observation V0: {row['transition_id']}")
        if row["candidate_count"] != len(candidates) or not 0 <= row["label_index"] < len(candidates):
            raise HumanRecordingError(f"invalid candidate label in {row['transition_id']}")
        if candidates[row["label_index"]]["action_type"] != row["label_action_type"]:
            raise HumanRecordingError(f"label action mismatch in {row['transition_id']}")
        if len({value["candidate_id"] for value in candidates}) != len(candidates):
            raise HumanRecordingError(f"duplicate candidates in {row['transition_id']}")
    return {
        "status": "PASS",
        "samples": len(rows),
        "unmatched_labels": 0,
        "duplicate_candidates": 0,
        "forbidden_observation_fields": 0,
        "split_counts": manifest["split_counts"],
        "combat_difficulty_counts": manifest["combat_difficulty_counts"],
        "combat_difficulty_split_counts": manifest["combat_difficulty_split_counts"],
        "label_action_counts": manifest["label_action_counts"],
        "max_candidate_count": manifest["max_candidate_count"],
    }


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*( _nested_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(child) for child in value), set())
    return set()
