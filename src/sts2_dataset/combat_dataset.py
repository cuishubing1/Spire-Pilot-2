from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .constants import ROOT
from .combat_difficulty import (
    COMBAT_DIFFICULTY_POLICY,
    combat_difficulty_definitions,
    combat_difficulty_tier,
)
from .combat_encounter import (
    ENCOUNTER_SIGNATURE_VERSION,
    encounter_signature_from_observation,
)
from .human import (
    HUMAN_DATASET_ROOT,
    HUMAN_TRANSITION_SCHEMA,
    HumanRecordingError,
    validate_human_dataset,
)
from .util import canonical_json, load_json, sha256_bytes, sha256_file, utc_now, write_json_atomic

COMBAT_CONFIG_PATH = ROOT / "config" / "combat_dataset_v1.json"
COMBAT_DATASET_ROOT = HUMAN_DATASET_ROOT.parent / "combat_v1"

COMBAT_TRANSITION_SCHEMA = pa.schema(
    [
        ("combat_id", pa.string()),
        ("split", pa.string()),
        ("source_transition_sha256", pa.string()),
        *[(field.name, field.type) for field in HUMAN_TRANSITION_SCHEMA],
    ]
)

COMBAT_SCHEMA = pa.schema(
    [
        ("combat_id", pa.string()),
        ("run_id", pa.string()),
        ("ascension", pa.int32()),
        ("combat_difficulty_tier", pa.string()),
        ("encounter_signature", pa.string()),
        ("act", pa.int32()),
        ("floor", pa.int32()),
        ("room_type", pa.string()),
        ("act_id", pa.string()),
        ("map_col", pa.int32()),
        ("map_row", pa.int32()),
        ("room_model_id", pa.string()),
        ("split", pa.string()),
        ("transition_count", pa.int32()),
        ("first_record_sequence", pa.int64()),
        ("last_record_sequence", pa.int64()),
        ("strict_vanilla", pa.bool_()),
    ]
)


def _write_parquet_atomic(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    os.replace(temporary, path)


def _transition_fingerprint(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode("utf-8"))


def _combat_locator(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    observation = json.loads(row["observation_json"])
    run = observation.get("run") if isinstance(observation, dict) else {}
    run = run if isinstance(run, dict) else {}
    coord = run.get("map_coord")
    coord = coord if isinstance(coord, dict) else {}
    locator = {
        "run_id": row["run_id"],
        "act": int(row["act"]),
        "floor": int(row["floor"]),
        "room_type": row["room_type"],
        "act_id": run.get("act_id"),
        "map_col": coord.get("col"),
        "map_row": coord.get("row"),
        "room_model_id": run.get("room_model_id"),
    }
    digest = hashlib.sha256(canonical_json(locator).encode("utf-8")).hexdigest()[:24]
    return f"combat-{digest}", locator


def _split_ratios(config: dict[str, Any]) -> list[tuple[str, float]]:
    raw = config.get("splits")
    if not isinstance(raw, dict) or not raw:
        raise HumanRecordingError("combat dataset config must define non-empty splits")
    ratios = [(str(name), float(value)) for name, value in raw.items()]
    if "test" in {name for name, _ in ratios}:
        raise HumanRecordingError(
            "combat test split is run-held-out; configure test_run_ids instead of a test ratio"
        )
    if any(value <= 0 for _, value in ratios) or abs(sum(value for _, value in ratios) - 1.0) > 1e-9:
        raise HumanRecordingError("combat split ratios must be positive and sum to 1")
    return ratios


def _target_counts(total: int, ratios: list[tuple[str, float]]) -> dict[str, int]:
    exact = [(name, total * ratio) for name, ratio in ratios]
    counts = {name: int(value) for name, value in exact}
    remaining = total - sum(counts.values())
    order = sorted(
        range(len(exact)),
        key=lambda index: (-(exact[index][1] - int(exact[index][1])), index),
    )
    for index in order[:remaining]:
        counts[exact[index][0]] += 1
    return counts


def _assign_new_combats(
    combat_strata: dict[str, tuple[int, str]],
    existing: dict[str, str],
    ratios: list[tuple[str, float]],
    seed: str,
) -> dict[str, str]:
    assignments = dict(existing)
    split_order = [name for name, _ in ratios]
    by_stratum: dict[tuple[int, str], list[str]] = defaultdict(list)
    for combat_id, stratum in combat_strata.items():
        by_stratum[stratum].append(combat_id)
    for (act, difficulty_tier), combat_ids in sorted(by_stratum.items()):
        targets = _target_counts(len(combat_ids), ratios)
        counts = Counter(assignments[combat_id] for combat_id in combat_ids if combat_id in assignments)
        pending = sorted(
            (combat_id for combat_id in combat_ids if combat_id not in assignments),
            key=lambda combat_id: hashlib.sha256(
                f"{seed}:{act}:{difficulty_tier}:{combat_id}".encode("utf-8")
            ).hexdigest(),
        )
        for combat_id in pending:
            deficits = {name: targets[name] - counts[name] for name in split_order}
            split = max(split_order, key=lambda name: (deficits[name], -split_order.index(name)))
            if deficits[split] <= 0:
                split = min(
                    split_order,
                    key=lambda name: (counts[name] / max(targets[name], 1), split_order.index(name)),
                )
            assignments[combat_id] = split
            counts[split] += 1
    return assignments


def _eligible_run_ids(episodes: list[dict[str, Any]], config: dict[str, Any]) -> set[str]:
    allowed = set(config.get("character_ids") or [])
    if not allowed:
        return {str(row["run_id"]) for row in episodes}
    result: set[str] = set()
    for row in episodes:
        character_ids = set(json.loads(row["character_ids_json"]))
        if character_ids and character_ids.issubset(allowed):
            result.add(str(row["run_id"]))
    return result


def _configured_test_run_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("test_run_ids")
    if not isinstance(raw, list) or not raw:
        raise HumanRecordingError("combat dataset config must define at least one test_run_id")
    run_ids = [str(value) for value in raw]
    if len(run_ids) != len(set(run_ids)):
        raise HumanRecordingError("combat dataset config contains duplicate test_run_ids")
    return run_ids


def _test_run_metadata(
    test_run_ids: list[str],
    episodes_by_run: dict[str, dict[str, Any]],
    combat_acts_by_run: dict[str, set[int]],
    allowed_acts: set[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    constraints = config.get("test_run_constraints") or {}
    if not isinstance(constraints, dict):
        raise HumanRecordingError("test_run_constraints must be an object")
    missing = [run_id for run_id in test_run_ids if run_id not in episodes_by_run]
    if missing:
        raise HumanRecordingError(f"configured test runs are absent from human episodes: {missing}")
    required_ascensions = {int(value) for value in constraints.get("required_ascensions", [])}
    minimum_max_act = int(constraints.get("minimum_max_act", 0))
    minimum_max_floor = int(constraints.get("minimum_max_floor", 0))
    require_complete_context = bool(constraints.get("require_complete_context", False))
    require_victory = bool(constraints.get("require_victory", False))
    metadata: list[dict[str, Any]] = []
    for run_id in test_run_ids:
        episode = episodes_by_run[run_id]
        acts = combat_acts_by_run.get(run_id, set())
        if not acts:
            raise HumanRecordingError(f"configured test run has no eligible combat: {run_id}")
        if not allowed_acts.issubset(acts):
            raise HumanRecordingError(
                f"configured test run does not cover every requested act: {run_id}; acts={sorted(acts)}"
            )
        if require_complete_context and episode.get("run_context_quality") != "complete":
            raise HumanRecordingError(f"configured test run has incomplete run context: {run_id}")
        if require_victory and episode.get("victory") is not True:
            raise HumanRecordingError(f"configured test run is not a recorded victory: {run_id}")
        if int(episode.get("max_act") or 0) < minimum_max_act:
            raise HumanRecordingError(f"configured test run ended before required act: {run_id}")
        if int(episode.get("max_floor") or 0) < minimum_max_floor:
            raise HumanRecordingError(f"configured test run ended before required floor: {run_id}")
        metadata.append({
            "run_id": run_id,
            "seed": episode.get("seed"),
            "ascension": episode.get("ascension"),
            "combat_difficulty_tier": combat_difficulty_tier(
                int(episode.get("ascension") or 0)
            ),
            "victory": episode.get("victory"),
            "terminal_reason": episode.get("terminal_reason"),
            "max_act": episode.get("max_act"),
            "max_floor": episode.get("max_floor"),
            "rollback_count": episode.get("rollback_count"),
            "unmatched_resume_count": episode.get("unmatched_resume_count"),
            "combat_acts": sorted(acts),
        })
    observed_ascensions = {int(row["ascension"]) for row in metadata if row["ascension"] is not None}
    if not required_ascensions.issubset(observed_ascensions):
        raise HumanRecordingError(
            "configured test runs do not satisfy required ascensions: "
            f"missing={sorted(required_ascensions - observed_ascensions)}"
        )
    return metadata


def build_combat_dataset(config_path: Path = COMBAT_CONFIG_PATH, *, rebuild: bool = False) -> dict[str, Any]:
    validate_human_dataset()
    config = load_json(config_path)
    ratios = _split_ratios(config)
    allowed_acts = {int(value) for value in config.get("acts", [1, 2, 3])}
    eligibility_column = str(config.get("eligibility_column", "is_training_eligible"))
    if eligibility_column not in {"is_training_eligible", "strict_vanilla_eligible"}:
        raise HumanRecordingError(f"unsupported combat eligibility column: {eligibility_column}")

    episodes = pq.read_table(HUMAN_DATASET_ROOT / "episodes.parquet").to_pylist()
    episodes_by_run = {str(row["run_id"]): row for row in episodes}
    eligible_runs = _eligible_run_ids(episodes, config)
    test_run_ids = _configured_test_run_ids(config)
    ineligible_test_runs = sorted(set(test_run_ids) - eligible_runs)
    if ineligible_test_runs:
        raise HumanRecordingError(
            f"configured test runs do not match character eligibility: {ineligible_test_runs}"
        )
    human_rows = pq.read_table(HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()
    selected = [
        row for row in human_rows
        if row["phase"] == "combat_play"
        and bool(row[eligibility_column])
        and int(row["act"]) in allowed_acts
        and row["run_id"] in eligible_runs
    ]
    if not selected:
        raise HumanRecordingError("no eligible combat transitions matched the combat dataset config")

    current_by_transition: dict[str, dict[str, Any]] = {}
    locator_by_combat: dict[str, dict[str, Any]] = {}
    combat_acts: dict[str, int] = {}
    combat_strata: dict[str, tuple[int, str]] = {}
    combat_acts_by_run: dict[str, set[int]] = defaultdict(set)
    for row in selected:
        combat_id, locator = _combat_locator(row)
        transition_id = str(row["transition_id"])
        if transition_id in current_by_transition:
            raise HumanRecordingError(f"duplicate human transition_id: {transition_id}")
        current_by_transition[transition_id] = {
            "combat_id": combat_id,
            "source_transition_sha256": _transition_fingerprint(row),
            **row,
        }
        previous_locator = locator_by_combat.setdefault(combat_id, locator)
        if previous_locator != locator:
            raise HumanRecordingError(f"combat id collision: {combat_id}")
        combat_acts[combat_id] = int(row["act"])
        ascension = int(episodes_by_run[str(row["run_id"])].get("ascension") or 0)
        combat_strata[combat_id] = (
            int(row["act"]),
            combat_difficulty_tier(ascension),
        )
        combat_acts_by_run[str(row["run_id"])].add(int(row["act"]))

    test_metadata = _test_run_metadata(
        test_run_ids, episodes_by_run, combat_acts_by_run, allowed_acts, config
    )
    test_run_id_set = set(test_run_ids)
    test_combat_ids = {
        combat_id for combat_id, locator in locator_by_combat.items()
        if locator["run_id"] in test_run_id_set
    }

    transition_path = COMBAT_DATASET_ROOT / "transitions.parquet"
    combat_path = COMBAT_DATASET_ROOT / "combats.parquet"
    manifest_path = COMBAT_DATASET_ROOT / "manifest.json"
    existing_rows: list[dict[str, Any]] = []
    existing_assignments: dict[str, str] = {}
    if not rebuild and manifest_path.exists():
        validate_combat_dataset(config_path)
        existing_rows = pq.read_table(transition_path).to_pylist()
        for row in existing_rows:
            transition_id = row["transition_id"]
            current = current_by_transition.get(transition_id)
            if current is None:
                raise HumanRecordingError(
                    f"combat dataset is not append-only; missing source transition {transition_id}; rerun with --rebuild"
                )
            if row["source_transition_sha256"] != current["source_transition_sha256"]:
                raise HumanRecordingError(
                    f"source transition changed for {transition_id}; rerun with --rebuild"
                )
            existing_assignments[row["combat_id"]] = row["split"]

    existing_ids = {row["transition_id"] for row in existing_rows}
    new_rows = [row for transition_id, row in current_by_transition.items() if transition_id not in existing_ids]
    non_test_existing = {
        combat_id: split for combat_id, split in existing_assignments.items()
        if combat_id not in test_combat_ids
    }
    assignments = _assign_new_combats(
        {
            combat_id: stratum
            for combat_id, stratum in combat_strata.items()
            if combat_id not in test_combat_ids
        },
        non_test_existing,
        ratios,
        str(config.get("split_seed", "sts2-combat-v1")),
    )
    assignments.update({combat_id: "test" for combat_id in test_combat_ids})
    combined: list[dict[str, Any]] = []
    for row in [*existing_rows, *new_rows]:
        combined.append({**row, "split": assignments[row["combat_id"]]})
    combined.sort(key=lambda row: (row["run_id"], row["record_sequence"], row["transition_id"]))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        grouped[row["combat_id"]].append(row)
    combat_rows: list[dict[str, Any]] = []
    for combat_id, rows in grouped.items():
        locator = locator_by_combat[combat_id]
        ascension = int(episodes_by_run[str(locator["run_id"])].get("ascension") or 0)
        first_row = min(
            rows,
            key=lambda row: (int(row["record_sequence"]), int(row["step_id"])),
        )
        encounter_signature = encounter_signature_from_observation(
            json.loads(first_row["observation_json"])
        )
        combat_rows.append(
            {
                **locator,
                "combat_id": combat_id,
                "ascension": ascension,
                "combat_difficulty_tier": combat_difficulty_tier(ascension),
                "encounter_signature": encounter_signature,
                "split": assignments[combat_id],
                "transition_count": len(rows),
                "first_record_sequence": min(int(row["record_sequence"]) for row in rows),
                "last_record_sequence": max(int(row["record_sequence"]) for row in rows),
                "strict_vanilla": all(bool(row["strict_vanilla_eligible"]) for row in rows),
            }
        )
    combat_rows.sort(key=lambda row: (row["run_id"], row["act"], row["floor"], row["combat_id"]))

    _write_parquet_atomic(combined, COMBAT_TRANSITION_SCHEMA, transition_path)
    _write_parquet_atomic(combat_rows, COMBAT_SCHEMA, combat_path)
    split_names = [name for name, _ in ratios] + ["test"]
    split_counts = {
        name: {
            "combats": sum(row["split"] == name for row in combat_rows),
            "transitions": sum(row["split"] == name for row in combined),
        }
        for name in split_names
    }
    act_split_counts = {
        str(act): {
            name: {
                "combats": sum(row["act"] == act and row["split"] == name for row in combat_rows),
                "transitions": sum(row["act"] == act and row["split"] == name for row in combined),
            }
            for name in split_names
        }
        for act in sorted({row["act"] for row in combat_rows})
    }
    difficulty_split_counts = {
        tier: {
            name: {
                "combats": sum(
                    row["combat_difficulty_tier"] == tier and row["split"] == name
                    for row in combat_rows
                ),
                "transitions": sum(
                    combat_strata[row["combat_id"]][1] == tier and row["split"] == name
                    for row in combined
                ),
            }
            for name in split_names
        }
        for tier in sorted({row["combat_difficulty_tier"] for row in combat_rows})
    }
    act_difficulty_split_counts = {
        str(act): {
            tier: {
                name: {
                    "combats": sum(
                        row["act"] == act
                        and row["combat_difficulty_tier"] == tier
                        and row["split"] == name
                        for row in combat_rows
                    ),
                    "transitions": sum(
                        int(row["act"]) == act
                        and combat_strata[row["combat_id"]][1] == tier
                        and row["split"] == name
                        for row in combined
                    ),
                }
                for name in split_names
            }
            for tier in sorted({row["combat_difficulty_tier"] for row in combat_rows})
            if any(
                row["act"] == act and row["combat_difficulty_tier"] == tier
                for row in combat_rows
            )
        }
        for act in sorted({row["act"] for row in combat_rows})
    }
    splits_by_run: dict[str, set[str]] = defaultdict(set)
    for row in combat_rows:
        splits_by_run[row["run_id"]].add(row["split"])
    test_run_leak_count = sum(
        splits_by_run.get(run_id, set()) != {"test"} for run_id in test_run_ids
    ) + sum(
        "test" in splits and run_id not in test_run_id_set
        for run_id, splits in splits_by_run.items()
    )
    manifest = {
        "schema_version": str(config.get("schema_version", "combat-dataset-1.2.0")),
        "generated_at": utc_now(),
        "update_mode": "full_rebuild" if rebuild or not existing_rows else "incremental_append",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(HUMAN_DATASET_ROOT / "manifest.json"),
        "eligibility_column": eligibility_column,
        "character_ids": config.get("character_ids", []),
        "acts": sorted(allowed_acts),
        "combat_difficulty_policy": COMBAT_DIFFICULTY_POLICY,
        "combat_difficulty_tiers": combat_difficulty_definitions(),
        "encounter_signature_version": ENCOUNTER_SIGNATURE_VERSION,
        "split_seed": str(config.get("split_seed", "sts2-combat-v1")),
        "splits": dict(ratios),
        "test_split_policy": "explicit_run_holdout",
        "test_run_ids": test_run_ids,
        "test_runs": test_metadata,
        "test_run_count": len(test_run_ids),
        "combat_count": len(combat_rows),
        "transition_count": len(combined),
        "new_combat_count": len({row["combat_id"] for row in new_rows} - set(existing_assignments)),
        "new_transition_count": len(new_rows),
        "split_counts": split_counts,
        "act_split_counts": act_split_counts,
        "difficulty_split_counts": difficulty_split_counts,
        "act_difficulty_split_counts": act_difficulty_split_counts,
        "run_count": len(splits_by_run),
        "cross_split_run_count": sum(len(splits) > 1 for splits in splits_by_run.values()),
        "test_run_leak_count": test_run_leak_count,
        "files": [
            {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in (combat_path, transition_path)
        ],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def validate_combat_dataset(config_path: Path = COMBAT_CONFIG_PATH) -> dict[str, Any]:
    manifest_path = COMBAT_DATASET_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise HumanRecordingError("combat dataset manifest does not exist")
    manifest = load_json(manifest_path)
    if manifest.get("combat_difficulty_policy") != COMBAT_DIFFICULTY_POLICY:
        raise HumanRecordingError("combat difficulty policy changed; rebuild the combat dataset")
    if manifest.get("encounter_signature_version") != ENCOUNTER_SIGNATURE_VERSION:
        raise HumanRecordingError("encounter signature version changed; rebuild the combat dataset")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise HumanRecordingError("combat dataset config changed; rebuild the combat dataset")
    for entry in manifest.get("files", []):
        path = Path(entry["path"])
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            raise HumanRecordingError(f"missing or modified combat dataset file: {path}")
    transitions = pq.read_table(COMBAT_DATASET_ROOT / "transitions.parquet").to_pylist()
    combats = pq.read_table(COMBAT_DATASET_ROOT / "combats.parquet").to_pylist()
    if len(transitions) != manifest.get("transition_count") or len(combats) != manifest.get("combat_count"):
        raise HumanRecordingError("combat dataset row count does not match manifest")
    if len({row["transition_id"] for row in transitions}) != len(transitions):
        raise HumanRecordingError("combat dataset contains duplicate transitions")
    split_by_combat: dict[str, str] = {}
    for row in transitions:
        previous = split_by_combat.setdefault(row["combat_id"], row["split"])
        if previous != row["split"]:
            raise HumanRecordingError(f"combat crosses dataset splits: {row['combat_id']}")
    combat_ids = {row["combat_id"] for row in combats}
    if combat_ids != set(split_by_combat):
        raise HumanRecordingError("combat index and transition combat ids differ")
    config = load_json(config_path)
    combat_by_id = {str(row["combat_id"]): row for row in combats}
    for combat_id, row in combat_by_id.items():
        if row["split"] != split_by_combat[combat_id]:
            raise HumanRecordingError(
                f"combat index split differs from transitions: {combat_id}"
            )
    transitions_by_combat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transitions:
        transitions_by_combat[str(row["combat_id"])].append(row)
    for combat_id, rows in transitions_by_combat.items():
        first_row = min(
            rows,
            key=lambda row: (int(row["record_sequence"]), int(row["step_id"])),
        )
        expected_signature = encounter_signature_from_observation(
            json.loads(first_row["observation_json"])
        )
        if combat_by_id[combat_id]["encounter_signature"] != expected_signature:
            raise HumanRecordingError(
                f"combat encounter signature differs from first decision: {combat_id}"
            )
    split_names = [name for name, _ in _split_ratios(config)] + ["test"]
    computed_split_counts = {
        name: {
            "combats": sum(row["split"] == name for row in combats),
            "transitions": sum(row["split"] == name for row in transitions),
        }
        for name in split_names
    }
    computed_act_split_counts = {
        str(act): {
            name: {
                "combats": sum(
                    int(row["act"]) == act and row["split"] == name
                    for row in combats
                ),
                "transitions": sum(
                    int(row["act"]) == act and row["split"] == name
                    for row in transitions
                ),
            }
            for name in split_names
        }
        for act in sorted({int(row["act"]) for row in combats})
    }
    tiers = sorted({str(row["combat_difficulty_tier"]) for row in combats})
    computed_difficulty_split_counts = {
        tier: {
            name: {
                "combats": sum(
                    row["combat_difficulty_tier"] == tier and row["split"] == name
                    for row in combats
                ),
                "transitions": sum(
                    combat_by_id[str(row["combat_id"])]["combat_difficulty_tier"] == tier
                    and row["split"] == name
                    for row in transitions
                ),
            }
            for name in split_names
        }
        for tier in tiers
    }
    computed_act_difficulty_split_counts = {
        str(act): {
            tier: {
                name: {
                    "combats": sum(
                        int(row["act"]) == act
                        and row["combat_difficulty_tier"] == tier
                        and row["split"] == name
                        for row in combats
                    ),
                    "transitions": sum(
                        int(row["act"]) == act
                        and combat_by_id[str(row["combat_id"])]["combat_difficulty_tier"] == tier
                        and row["split"] == name
                        for row in transitions
                    ),
                }
                for name in split_names
            }
            for tier in tiers
            if any(
                int(row["act"]) == act and row["combat_difficulty_tier"] == tier
                for row in combats
            )
        }
        for act in sorted({int(row["act"]) for row in combats})
    }
    for key, computed in (
        ("split_counts", computed_split_counts),
        ("act_split_counts", computed_act_split_counts),
        ("difficulty_split_counts", computed_difficulty_split_counts),
        ("act_difficulty_split_counts", computed_act_difficulty_split_counts),
    ):
        if manifest.get(key) != computed:
            raise HumanRecordingError(f"combat dataset {key} does not match Parquet rows")
    test_run_ids = set(_configured_test_run_ids(config))
    splits_by_run: dict[str, set[str]] = defaultdict(set)
    combat_acts_by_run: dict[str, set[int]] = defaultdict(set)
    for row in combats:
        expected_tier = combat_difficulty_tier(int(row["ascension"]))
        if row["combat_difficulty_tier"] != expected_tier:
            raise HumanRecordingError(
                f"combat difficulty tier mismatch: {row['combat_id']}"
            )
        splits_by_run[row["run_id"]].add(row["split"])
        combat_acts_by_run[row["run_id"]].add(int(row["act"]))
    leaked_test_runs = sorted(
        run_id for run_id in test_run_ids if splits_by_run.get(run_id) != {"test"}
    )
    unexpected_test_runs = sorted(
        run_id for run_id, splits in splits_by_run.items()
        if "test" in splits and run_id not in test_run_ids
    )
    if leaked_test_runs or unexpected_test_runs:
        raise HumanRecordingError(
            "test run isolation failed: "
            f"leaked={leaked_test_runs}, unexpected={unexpected_test_runs}"
        )
    episodes = pq.read_table(HUMAN_DATASET_ROOT / "episodes.parquet").to_pylist()
    episodes_by_run = {str(row["run_id"]): row for row in episodes}
    for row in combats:
        episode = episodes_by_run.get(str(row["run_id"]))
        if episode is None or int(episode.get("ascension") or 0) != int(row["ascension"]):
            raise HumanRecordingError(
                f"combat ascension differs from source episode: {row['combat_id']}"
            )
    _test_run_metadata(
        list(manifest["test_run_ids"]),
        episodes_by_run,
        combat_acts_by_run,
        {int(value) for value in config.get("acts", [1, 2, 3])},
        config,
    )
    required_tiers = {
        str(value) for value in config.get("required_combat_difficulty_tiers", [])
    }
    observed_tiers = {str(row["combat_difficulty_tier"]) for row in combats}
    if not required_tiers.issubset(observed_tiers):
        raise HumanRecordingError(
            "combat dataset is missing required difficulty tiers: "
            f"{sorted(required_tiers - observed_tiers)}"
        )
    minimum_stratum = int(
        config.get("minimum_stratum_combats_for_split_coverage", 0)
    )
    non_test_splits = [name for name, _ in _split_ratios(config)]
    stratum_counts = Counter(
        (int(row["act"]), str(row["combat_difficulty_tier"]))
        for row in combats if row["split"] != "test"
    )
    missing_stratum_splits: list[dict[str, Any]] = []
    for (act, tier), count in sorted(stratum_counts.items()):
        if count < minimum_stratum:
            continue
        present = {
            str(row["split"])
            for row in combats
            if row["split"] != "test"
            and int(row["act"]) == act
            and row["combat_difficulty_tier"] == tier
        }
        missing = sorted(set(non_test_splits) - present)
        if missing:
            missing_stratum_splits.append({
                "act": act,
                "combat_difficulty_tier": tier,
                "combat_count": count,
                "missing_splits": missing,
            })
    if missing_stratum_splits:
        raise HumanRecordingError(
            "act/difficulty validation coverage failed: "
            f"{missing_stratum_splits}"
        )
    return {
        "status": "PASS",
        "combats": len(combats),
        "transitions": len(transitions),
        "split_counts": manifest["split_counts"],
        "act_split_counts": manifest["act_split_counts"],
        "difficulty_split_counts": manifest["difficulty_split_counts"],
        "act_difficulty_split_counts": manifest["act_difficulty_split_counts"],
        "combat_difficulty_policy": manifest["combat_difficulty_policy"],
        "missing_stratum_splits": 0,
        "cross_split_combats": 0,
        "cross_split_runs": manifest.get("cross_split_run_count", 0),
        "test_runs": len(test_run_ids),
        "test_run_leaks": 0,
    }
