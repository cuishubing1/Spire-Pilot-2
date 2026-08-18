from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .combat_contract import (
    ACTION_VERSION,
    COMBAT_MODEL_ROOT,
    OBSERVATION_VERSION,
    iter_combat_model_samples,
    validate_combat_model_examples,
)
from .combat_difficulty import combat_difficulty_tier
from .combat_encounter import encounter_signature_from_observation
from .combat_engine_features import (
    CANDIDATE_ENGINE_FEATURE_DIM,
    CANDIDATE_ENGINE_FEATURE_VERSION,
    candidate_engine_feature_vector,
)
from .human import HumanRecordingError
from .util import load_json, sha256_file, utc_now, write_json_atomic

TENSORIZER_VERSION = "combat-tensorizer-0.3.0"
SUPPORTED_TENSORIZER_VERSIONS = {
    "combat-tensorizer-0.2.0",
    TENSORIZER_VERSION,
}
VOCAB_PATH = COMBAT_MODEL_ROOT / "vocab.json"
NUMERIC_FEATURE_DIM = 64
CATEGORICAL_FEATURE_DIM = 64
ENCOUNTER_MIN_TRAIN_COMBATS = 3

ENTITY_TYPES = ["<PAD>", "global", "hand", "draw", "discard", "exhaust", "enemy", "relic", "potion", "power", "orb"]
ACTION_TYPES = ["<PAD>", "play_card", "use_potion", "discard_potion", "end_turn"]
TARGET_KINDS = ["<PAD>", "none", "self", "enemy", "all_enemies"]


def _identity(entity_type: str, entity: dict[str, Any]) -> str:
    if entity_type == "global":
        return "global"
    identity_type = "card" if entity_type in {"hand", "draw", "discard", "exhaust"} else entity_type
    return f"{identity_type}:{entity.get('id') or '<UNKNOWN>'}"


def _iter_entities(observation: dict[str, Any]):
    yield "global", observation["global"]
    for value in observation.get("hand", []):
        yield "hand", value
    for zone in ("draw", "discard", "exhaust"):
        for value in observation.get("piles", {}).get(zone, []):
            yield zone, value
    for entity_type, field in (
        ("enemy", "enemies"), ("relic", "relics"), ("potion", "potions"),
        ("power", "player_powers"), ("orb", "orbs"),
    ):
        for value in observation.get(field, []):
            yield entity_type, value


def build_combat_vocabulary(*, rebuild: bool = False) -> dict[str, Any]:
    validate_combat_model_examples()
    train_samples = list(iter_combat_model_samples("train"))
    identities = sorted({
        _identity(entity_type, entity)
        for sample in train_samples
        for entity_type, entity in _iter_entities(sample["observation"])
    })
    encounter_by_combat = {
        str(sample["combat_id"]): str(sample["encounter_signature"])
        for sample in train_samples
    }
    encounter_counts = Counter(encounter_by_combat.values())
    eligible_encounters = sorted(
        signature
        for signature, count in encounter_counts.items()
        if count >= ENCOUNTER_MIN_TRAIN_COMBATS
    )
    existing: dict[str, Any] | None = None
    if VOCAB_PATH.exists() and not rebuild:
        existing = load_json(VOCAB_PATH)
        if existing.get("tensorizer_version") not in SUPPORTED_TENSORIZER_VERSIONS:
            raise HumanRecordingError("combat vocabulary tensorizer version changed; rebuild vocabulary")
        values = list(existing["entity_identity"])
        encounter_values = list(existing.get("encounter_identity", ["<PAD>", "<UNK>"]))
    else:
        values = ["<PAD>", "<UNK>"]
        encounter_values = ["<PAD>", "<UNK>"]
    seen = set(values)
    additions = [value for value in identities if value not in seen]
    values.extend(additions)
    seen_encounters = set(encounter_values)
    encounter_additions = [
        value for value in eligible_encounters if value not in seen_encounters
    ]
    encounter_values.extend(encounter_additions)
    vocabulary = {
        "tensorizer_version": TENSORIZER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "generated_at": utc_now(),
        "update_mode": "full_rebuild" if rebuild or existing is None else "incremental_append",
        "source_manifest_sha256": sha256_file(COMBAT_MODEL_ROOT / "manifest.json"),
        "numeric_feature_dim": NUMERIC_FEATURE_DIM,
        "categorical_feature_dim": CATEGORICAL_FEATURE_DIM,
        "candidate_engine_feature_version": CANDIDATE_ENGINE_FEATURE_VERSION,
        "candidate_engine_feature_dim": CANDIDATE_ENGINE_FEATURE_DIM,
        "entity_types": ENTITY_TYPES,
        "action_types": ACTION_TYPES,
        "target_kinds": TARGET_KINDS,
        "entity_identity": values,
        "encounter_identity": encounter_values,
        "encounter_identity_min_train_combats": ENCOUNTER_MIN_TRAIN_COMBATS,
        "new_encounter_identity_count": len(encounter_additions),
        "train_encounter_identity_count": len(eligible_encounters),
        "new_entity_identity_count": len(additions),
        "train_entity_identity_count": len(identities),
    }
    write_json_atomic(VOCAB_PATH, vocabulary)
    return vocabulary


def _numeric_hash_features(value: Any, *, prefix: str = "", dim: int = NUMERIC_FEATURE_DIM) -> np.ndarray:
    result = np.zeros(dim, dtype=np.float32)

    def visit(child: Any, path: str) -> None:
        if isinstance(child, dict):
            for key in sorted(child):
                if key in {"entity_ref", "lineage_ref", "id", "schema_version"}:
                    continue
                visit(child[key], f"{path}.{key}" if path else key)
        elif isinstance(child, list):
            for item in child:
                visit(item, f"{path}[]")
        elif isinstance(child, bool):
            add(path, float(child))
        elif isinstance(child, (int, float)) and math.isfinite(float(child)):
            raw = float(child)
            add(path, math.copysign(math.log1p(abs(raw)), raw))

    def add(path: str, numeric: float) -> None:
        digest = hashlib.sha256(f"{prefix}:{path}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        result[bucket] += sign * numeric

    visit(value, "")
    return result


def _categorical_hash_features(
    value: Any,
    *,
    prefix: str = "",
    dim: int = CATEGORICAL_FEATURE_DIM,
) -> np.ndarray:
    """Encode only categorical fields that are present in Observation V0.

    Exact card/enemy/relic/potion identities use the append-stable identity
    vocabulary. This second channel preserves lower-cardinality strings such as
    room type, card type/rarity/target, intent type, keywords and status. Runtime
    references are deliberately excluded because they are bindings, not
    learnable categories.
    """
    result = np.zeros(dim, dtype=np.float32)
    excluded = {"entity_ref", "lineage_ref", "id", "schema_version", "target_combat_id", "target_id"}

    def visit(child: Any, path: str) -> None:
        if isinstance(child, dict):
            for key in sorted(child):
                if key in excluded:
                    continue
                visit(child[key], f"{path}.{key}" if path else key)
        elif isinstance(child, list):
            for item in child:
                visit(item, f"{path}[]")
        elif isinstance(child, str):
            digest = hashlib.sha256(f"{prefix}:{path}={child}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            result[bucket] += sign

    visit(value, "")
    return result


class CombatTensorizerV0:
    def __init__(self, vocabulary: dict[str, Any]):
        if vocabulary.get("tensorizer_version") not in SUPPORTED_TENSORIZER_VERSIONS:
            raise HumanRecordingError("unsupported combat tensorizer vocabulary")
        if int(vocabulary.get("numeric_feature_dim", 0)) != NUMERIC_FEATURE_DIM:
            raise HumanRecordingError("combat numeric feature dimension mismatch")
        if int(vocabulary.get("categorical_feature_dim", 0)) != CATEGORICAL_FEATURE_DIM:
            raise HumanRecordingError("combat categorical feature dimension mismatch")
        self.vocabulary = vocabulary
        self.identity_to_index = {value: index for index, value in enumerate(vocabulary["entity_identity"])}
        self.entity_type_to_index = {value: index for index, value in enumerate(vocabulary["entity_types"])}
        self.action_type_to_index = {value: index for index, value in enumerate(vocabulary["action_types"])}
        self.target_kind_to_index = {value: index for index, value in enumerate(vocabulary["target_kinds"])}
        self.encounter_to_index = {
            value: index
            for index, value in enumerate(vocabulary.get("encounter_identity", []))
        }

    @classmethod
    def from_default_vocabulary(cls) -> "CombatTensorizerV0":
        if not VOCAB_PATH.exists():
            raise HumanRecordingError("combat vocabulary does not exist; run build-combat-vocab")
        return cls(load_json(VOCAB_PATH))

    def tensorize(self, sample: dict[str, Any]) -> dict[str, Any]:
        ascension = int(
            sample.get(
                "ascension",
                sample["observation"].get("global", {}).get("ascension", 0),
            )
        )
        encounter_signature = str(
            sample.get("encounter_signature")
            or encounter_signature_from_observation(sample["observation"])
        )
        entities = list(_iter_entities(sample["observation"]))
        entity_type = np.asarray([self.entity_type_to_index[kind] for kind, _ in entities], dtype=np.int64)
        entity_identity = np.asarray([
            self.identity_to_index.get(_identity(kind, value), 1) for kind, value in entities
        ], dtype=np.int64)
        entity_numeric = np.stack([
            _numeric_hash_features(value, prefix=kind) for kind, value in entities
        ]).astype(np.float32, copy=False)
        entity_categorical = np.stack([
            _categorical_hash_features(value, prefix=kind) for kind, value in entities
        ]).astype(np.float32, copy=False)

        reference_to_index: dict[tuple[str, str], int] = {}
        for index, (kind, value) in enumerate(entities):
            reference = value.get("entity_ref") if isinstance(value, dict) else None
            if reference is not None:
                reference_to_index[(kind, str(reference))] = index
        action_type = []
        action_source = []
        action_target = []
        action_target_kind = []
        candidate_engine_numeric = []
        for candidate in sample["candidates"]:
            action_type.append(self.action_type_to_index[candidate["action_type"]])
            source_type = {"card": "hand", "potion": "potion"}.get(candidate.get("source_type"))
            source_ref = candidate.get("source_ref")
            if source_type is None:
                source_index = 0
            else:
                source_index = reference_to_index.get((source_type, str(source_ref)))
                if source_index is None:
                    raise HumanRecordingError(
                        f"candidate source does not bind to tensor entity: {sample['transition_id']}"
                    )
            target_kind = candidate["target_kind"]
            if target_kind == "enemy":
                target_index = reference_to_index.get(("enemy", str(candidate.get("target_ref"))))
                if target_index is None:
                    raise HumanRecordingError(
                        f"candidate target does not bind to tensor entity: {sample['transition_id']}"
                    )
            else:
                target_index = 0
            action_source.append(source_index)
            action_target.append(target_index)
            action_target_kind.append(self.target_kind_to_index[target_kind])
            candidate_engine_numeric.append(
                candidate_engine_feature_vector(sample["observation"], candidate)
            )
        return {
            "transition_id": sample["transition_id"],
            "source_transition_sha256": sample.get("source_transition_sha256"),
            "combat_id": sample["combat_id"],
            "split": sample["split"],
            "entity_type": entity_type,
            "entity_identity": entity_identity,
            "entity_numeric": entity_numeric,
            "entity_categorical": entity_categorical,
            "action_type": np.asarray(action_type, dtype=np.int64),
            "action_source": np.asarray(action_source, dtype=np.int64),
            "action_target": np.asarray(action_target, dtype=np.int64),
            "action_target_kind": np.asarray(action_target_kind, dtype=np.int64),
            "candidate_engine_numeric": np.asarray(
                candidate_engine_numeric, dtype=np.float32
            ).reshape(-1, CANDIDATE_ENGINE_FEATURE_DIM),
            "label": int(sample["label_index"]),
            "ascension": ascension,
            "combat_difficulty_tier": str(
                sample.get("combat_difficulty_tier")
                or combat_difficulty_tier(ascension)
            ),
            "encounter_signature": encounter_signature,
            "encounter_identity": self.encounter_to_index.get(
                encounter_signature, 1 if self.encounter_to_index else 0
            ),
            "act": int(sample["act"]),
            "floor": int(sample["floor"]),
            "room_type": str(sample["observation"]["global"].get("room_type") or "<UNKNOWN>"),
            "label_action_type": str(sample["label_action_type"]),
            "resource_target_mask": bool(sample.get("value_target")),
            "target_hp_loss_fraction": float((sample.get("value_target") or {}).get("hp_loss_fraction", 0.0)),
            "target_immediate_hp_loss_fraction": float(
                (sample.get("value_target") or {}).get("immediate_hp_loss_fraction", 0.0)
            ),
            "target_death": float(bool((sample.get("value_target") or {}).get("death", False))),
            "target_terminal_hp_fraction": (
                float((sample.get("value_target") or {}).get("terminal_hp", 0.0))
                / max(float((sample.get("value_target") or {}).get("terminal_max_hp", 1.0)), 1.0)
            ),
            "target_potion_spent": float((sample.get("value_target") or {}).get("potion_spent_to_end", 0.0)),
            "target_max_hp_delta": float((sample.get("value_target") or {}).get("max_hp_delta_to_end", 0.0)),
        }


def collate_combat_numpy(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    if not rows:
        raise HumanRecordingError("cannot collate an empty combat batch")
    batch = len(rows)
    max_entities = max(len(row["entity_type"]) for row in rows)
    max_actions = max(len(row["action_type"]) for row in rows)
    entity_type = np.zeros((batch, max_entities), dtype=np.int64)
    entity_identity = np.zeros((batch, max_entities), dtype=np.int64)
    entity_numeric = np.zeros((batch, max_entities, NUMERIC_FEATURE_DIM), dtype=np.float32)
    entity_categorical = np.zeros((batch, max_entities, CATEGORICAL_FEATURE_DIM), dtype=np.float32)
    entity_mask = np.zeros((batch, max_entities), dtype=np.bool_)
    action_type = np.zeros((batch, max_actions), dtype=np.int64)
    action_source = np.zeros((batch, max_actions), dtype=np.int64)
    action_target = np.zeros((batch, max_actions), dtype=np.int64)
    action_target_kind = np.zeros((batch, max_actions), dtype=np.int64)
    candidate_engine_numeric = np.zeros(
        (batch, max_actions, CANDIDATE_ENGINE_FEATURE_DIM), dtype=np.float32
    )
    action_mask = np.zeros((batch, max_actions), dtype=np.bool_)
    labels = np.zeros(batch, dtype=np.int64)
    acts = np.zeros(batch, dtype=np.int64)
    floors = np.zeros(batch, dtype=np.int64)
    ascensions = np.zeros(batch, dtype=np.int64)
    encounter_identities = np.zeros(batch, dtype=np.int64)
    resource_target_mask = np.zeros(batch, dtype=np.bool_)
    target_hp_loss_fraction = np.zeros(batch, dtype=np.float32)
    target_immediate_hp_loss_fraction = np.zeros(batch, dtype=np.float32)
    target_death = np.zeros(batch, dtype=np.float32)
    target_terminal_hp_fraction = np.zeros(batch, dtype=np.float32)
    target_potion_spent = np.zeros(batch, dtype=np.float32)
    target_max_hp_delta = np.zeros(batch, dtype=np.float32)
    for batch_index, row in enumerate(rows):
        entity_count = len(row["entity_type"])
        action_count = len(row["action_type"])
        entity_type[batch_index, :entity_count] = row["entity_type"]
        entity_identity[batch_index, :entity_count] = row["entity_identity"]
        entity_numeric[batch_index, :entity_count] = row["entity_numeric"]
        entity_categorical[batch_index, :entity_count] = row["entity_categorical"]
        entity_mask[batch_index, :entity_count] = True
        action_type[batch_index, :action_count] = row["action_type"]
        action_source[batch_index, :action_count] = row["action_source"]
        action_target[batch_index, :action_count] = row["action_target"]
        action_target_kind[batch_index, :action_count] = row["action_target_kind"]
        candidate_engine_numeric[batch_index, :action_count] = row[
            "candidate_engine_numeric"
        ]
        action_mask[batch_index, :action_count] = True
        labels[batch_index] = row["label"]
        acts[batch_index] = row["act"]
        floors[batch_index] = row["floor"]
        ascensions[batch_index] = row["ascension"]
        encounter_identities[batch_index] = row["encounter_identity"]
        resource_target_mask[batch_index] = row.get("resource_target_mask", False)
        target_hp_loss_fraction[batch_index] = row.get("target_hp_loss_fraction", 0.0)
        target_immediate_hp_loss_fraction[batch_index] = row.get("target_immediate_hp_loss_fraction", 0.0)
        target_death[batch_index] = row.get("target_death", 0.0)
        target_terminal_hp_fraction[batch_index] = row.get("target_terminal_hp_fraction", 0.0)
        target_potion_spent[batch_index] = row.get("target_potion_spent", 0.0)
        target_max_hp_delta[batch_index] = row.get("target_max_hp_delta", 0.0)
    return {
        "transition_id": [row["transition_id"] for row in rows],
        "combat_id": [row["combat_id"] for row in rows],
        "entity_type": entity_type,
        "entity_identity": entity_identity,
        "entity_numeric": entity_numeric,
        "entity_categorical": entity_categorical,
        "entity_mask": entity_mask,
        "action_type": action_type,
        "action_source": action_source,
        "action_target": action_target,
        "action_target_kind": action_target_kind,
        "candidate_engine_numeric": candidate_engine_numeric,
        "action_mask": action_mask,
        "label": labels,
        "act": acts,
        "floor": floors,
        "ascension": ascensions,
        "combat_difficulty_tier": [row["combat_difficulty_tier"] for row in rows],
        "encounter_signature": [row["encounter_signature"] for row in rows],
        "encounter_identity": encounter_identities,
        "room_type": [row["room_type"] for row in rows],
        "label_action_type": [row["label_action_type"] for row in rows],
        "resource_target_mask": resource_target_mask,
        "target_hp_loss_fraction": target_hp_loss_fraction,
        "target_immediate_hp_loss_fraction": target_immediate_hp_loss_fraction,
        "target_death": target_death,
        "target_terminal_hp_fraction": target_terminal_hp_fraction,
        "target_potion_spent": target_potion_spent,
        "target_max_hp_delta": target_max_hp_delta,
    }
