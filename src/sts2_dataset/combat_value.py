from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .combat_dataset import COMBAT_DATASET_ROOT, validate_combat_dataset
from .human import HumanRecordingError
from .util import load_json, sha256_file, utc_now, write_json_atomic


VALUE_TARGET_VERSION = "combat-resource-targets-0.2.0"
COMBAT_VALUE_ROOT = COMBAT_DATASET_ROOT.parent / "combat_value_v1"
VALUE_TARGET_PATH = COMBAT_VALUE_ROOT / "targets.parquet"
VALUE_MANIFEST_PATH = COMBAT_VALUE_ROOT / "manifest.json"

VALUE_TARGET_SCHEMA = pa.schema([
    ("transition_id", pa.string()),
    ("combat_id", pa.string()),
    ("split", pa.string()),
    ("current_hp", pa.int32()),
    ("current_max_hp", pa.int32()),
    ("terminal_hp", pa.int32()),
    ("terminal_max_hp", pa.int32()),
    ("hp_loss_to_end", pa.int32()),
    ("hp_loss_fraction", pa.float32()),
    ("death", pa.bool_()),
    ("potion_spent_to_end", pa.int32()),
    ("max_hp_delta_to_end", pa.int32()),
    ("immediate_hp_loss", pa.int32()),
    ("immediate_hp_loss_fraction", pa.float32()),
    ("immediate_max_hp_delta", pa.int32()),
    ("source_transition_sha256", pa.string()),
])


def _player(observation: dict[str, Any], transition_id: str) -> dict[str, Any]:
    player = observation.get("player")
    if not isinstance(player, dict):
        raise HumanRecordingError(f"combat value target lacks player state: {transition_id}")
    for field in ("hp", "max_hp"):
        if not isinstance(player.get(field), (int, float)):
            raise HumanRecordingError(f"combat value target lacks player {field}: {transition_id}")
    return player


def _potion_action(action_json: str) -> bool:
    action = json.loads(action_json)
    return action.get("action_id") in {"use_potion", "discard_potion"}


def _is_death(terminal_observation: dict[str, Any], terminal_player: dict[str, Any]) -> bool:
    if int(terminal_player["hp"]) <= 0:
        return True
    screen = terminal_observation.get("screen")
    return bool(isinstance(screen, dict) and screen.get("victory") is False)


def derive_combat_value_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["combat_id"])].append(row)
    targets: list[dict[str, Any]] = []
    for combat_id, combat_rows in grouped.items():
        ordered = sorted(combat_rows, key=lambda row: (int(row["record_sequence"]), int(row["step_id"])))
        terminal_observation = json.loads(ordered[-1]["next_observation_json"])
        terminal_player = _player(terminal_observation, str(ordered[-1]["transition_id"]))
        terminal_hp = int(terminal_player["hp"])
        terminal_max_hp = int(terminal_player["max_hp"])
        death = _is_death(terminal_observation, terminal_player)
        potion_spent_by_transition: dict[str, int] = {}
        future_potion_spent = 0
        for row in reversed(ordered):
            if _potion_action(str(row["action_json"])):
                future_potion_spent += 1
            potion_spent_by_transition[str(row["transition_id"])] = future_potion_spent
        for row in ordered:
            transition_id = str(row["transition_id"])
            observation = json.loads(row["observation_json"])
            following = json.loads(row["next_observation_json"])
            current_player = _player(observation, transition_id)
            following_player = _player(following, transition_id)
            current_hp = int(current_player["hp"])
            current_max_hp = int(current_player["max_hp"])
            immediate_hp_loss = max(0, current_hp - int(following_player["hp"]))
            targets.append({
                "transition_id": transition_id,
                "combat_id": combat_id,
                "split": str(row["split"]),
                "current_hp": current_hp,
                "current_max_hp": current_max_hp,
                "terminal_hp": terminal_hp,
                "terminal_max_hp": terminal_max_hp,
                "hp_loss_to_end": max(0, current_hp - terminal_hp),
                "hp_loss_fraction": max(0.0, (current_hp - terminal_hp) / max(current_max_hp, 1)),
                "death": death,
                "potion_spent_to_end": potion_spent_by_transition[transition_id],
                "max_hp_delta_to_end": terminal_max_hp - current_max_hp,
                "immediate_hp_loss": immediate_hp_loss,
                "immediate_hp_loss_fraction": immediate_hp_loss / max(current_max_hp, 1),
                "immediate_max_hp_delta": int(following_player["max_hp"]) - current_max_hp,
                "source_transition_sha256": str(row["source_transition_sha256"]),
            })
    targets.sort(key=lambda row: (row["combat_id"], row["transition_id"]))
    return targets


def _write_parquet_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=VALUE_TARGET_SCHEMA), temporary, compression="zstd")
    os.replace(temporary, path)


def build_combat_value_targets() -> dict[str, Any]:
    validate_combat_dataset()
    source_path = COMBAT_DATASET_ROOT / "transitions.parquet"
    rows = pq.read_table(source_path).to_pylist()
    targets = derive_combat_value_targets(rows)
    _write_parquet_atomic(targets, VALUE_TARGET_PATH)
    manifest = {
        "schema_version": VALUE_TARGET_VERSION,
        "generated_at": utc_now(),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "target_count": len(targets),
        "combat_count": len({row["combat_id"] for row in targets}),
        "death_target_count": sum(bool(row["death"]) for row in targets),
        "positive_hp_loss_count": sum(int(row["hp_loss_to_end"]) > 0 for row in targets),
        "positive_potion_cost_count": sum(int(row["potion_spent_to_end"]) > 0 for row in targets),
        "positive_growth_count": sum(int(row["max_hp_delta_to_end"]) > 0 for row in targets),
        "files": [{
            "path": str(VALUE_TARGET_PATH),
            "sha256": sha256_file(VALUE_TARGET_PATH),
            "size": VALUE_TARGET_PATH.stat().st_size,
        }],
    }
    write_json_atomic(VALUE_MANIFEST_PATH, manifest)
    return manifest


def validate_combat_value_targets() -> dict[str, Any]:
    if not VALUE_MANIFEST_PATH.exists() or not VALUE_TARGET_PATH.exists():
        raise HumanRecordingError("combat value targets do not exist")
    manifest = load_json(VALUE_MANIFEST_PATH)
    if manifest.get("schema_version") != VALUE_TARGET_VERSION:
        raise HumanRecordingError("unsupported combat value target version")
    source_path = COMBAT_DATASET_ROOT / "transitions.parquet"
    if manifest.get("source_sha256") != sha256_file(source_path):
        raise HumanRecordingError("combat value target source changed; rebuild targets")
    if manifest["files"][0].get("sha256") != sha256_file(VALUE_TARGET_PATH):
        raise HumanRecordingError("combat value target file fingerprint mismatch")
    rows = pq.read_table(VALUE_TARGET_PATH).to_pylist()
    if len(rows) != int(manifest["target_count"]):
        raise HumanRecordingError("combat value target row count mismatch")
    if len({row["transition_id"] for row in rows}) != len(rows):
        raise HumanRecordingError("combat value targets contain duplicate transition ids")
    if any(row["hp_loss_to_end"] < 0 or row["potion_spent_to_end"] < 0 for row in rows):
        raise HumanRecordingError("combat value targets contain invalid negative costs")
    return {
        "status": "PASS",
        "targets": len(rows),
        "combats": len({row["combat_id"] for row in rows}),
        "death_targets": sum(bool(row["death"]) for row in rows),
        "positive_growth_targets": sum(int(row["max_hp_delta_to_end"]) > 0 for row in rows),
    }


def load_combat_value_targets() -> dict[str, dict[str, Any]]:
    validate_combat_value_targets()
    return {
        str(row["transition_id"]): row
        for row in pq.read_table(VALUE_TARGET_PATH).to_pylist()
    }
