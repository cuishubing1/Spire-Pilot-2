from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .constants import DATASET_ROOT, FIXTURE_ROOT, LOCK_PATH, RAW_ROOT
from .util import canonical_json, iter_jsonl_zst, sha256_file, utc_now, write_json_atomic


def export_dataset(config: dict[str, Any]) -> dict[str, Any]:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    room_counts: Counter[str] = Counter()
    raw_files = sorted(RAW_ROOT.glob("*.jsonl.zst"))

    for path in raw_files:
        start = end = None
        for record in iter_jsonl_zst(path):
            kind = record["record_type"]
            if kind == "run_start":
                start = record
            elif kind == "auto_transition":
                phase_counts[record["phase"]] += 1
            elif kind == "decision":
                obs = record["observation"]
                phase_counts[obs["phase"]] += 1
                room = obs.get("context", {}).get("room_type")
                if room:
                    room_counts[str(room)] += 1
                transition = record.get("transition")
                if transition:
                    action = transition["action_t"]
                    action_counts[action["action"]] += 1
                    transitions.append(_transition_row(record, transition))
            elif kind == "run_end":
                end = record
        if start is None or end is None:
            raise ValueError(f"Unsealed/incomplete logical run: {path}")
        episodes.append(_episode_row(path, start, end))

    for path in sorted(FIXTURE_ROOT.glob("*.jsonl.zst")):
        for record in iter_jsonl_zst(path):
            if record["record_type"] in {"fixture", "auto_transition"}:
                phase = record.get("phase")
                if phase:
                    phase_counts[phase] += 1
                fixtures.append(
                    {
                        "fixture_id": record.get("fixture_id", record["run_id"]),
                        "phase": phase,
                        "source_path": str(path),
                        "payload_json": canonical_json(record.get("payload")),
                    }
                )

    episode_path = DATASET_ROOT / "episodes.parquet"
    transition_path = DATASET_ROOT / "transitions.parquet"
    fixture_path = DATASET_ROOT / "fixtures.parquet"
    _write_table(episode_path, episodes, EPISODE_SCHEMA)
    _write_table(transition_path, transitions, TRANSITION_SCHEMA)
    _write_table(fixture_path, fixtures, FIXTURE_SCHEMA)

    coverage = {
        "generated_at": utc_now(),
        "phase_counts": dict(sorted(phase_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "room_counts": dict(sorted(room_counts.items())),
        "missing_required_phases": sorted(set(config["required_phases"]) - set(phase_counts)),
    }
    write_json_atomic(DATASET_ROOT / "coverage.json", coverage)
    files = [episode_path, transition_path, fixture_path, DATASET_ROOT / "coverage.json"]
    manifest = {
        "schema_version": config["schema_version"],
        "dataset_version": config["dataset_version"],
        "generated_at": utc_now(),
        "environment_lock_sha256": sha256_file(LOCK_PATH),
        "natural_run_count": len(episodes),
        "transition_count": len(transitions),
        "fixture_count": len(fixtures),
        "raw_files": [{"path": str(p), "sha256": sha256_file(p)} for p in raw_files],
        "files": [{"path": str(p), "sha256": sha256_file(p), "size": p.stat().st_size} for p in files],
    }
    write_json_atomic(DATASET_ROOT / "manifest.json", manifest)
    return manifest


def _episode_row(path: Path, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": start["run_id"],
        "seed": start["seed"],
        "character": start["character"],
        "ascension": int(start["ascension"]),
        "policy_id": start["policy_id"],
        "started_at": end["started_at"],
        "ended_at": end["ended_at"],
        "terminal": bool(end["terminal"]),
        "victory": bool(end["victory"]),
        "max_act": int(end["max_act"]),
        "max_floor": int(end["max_floor"]),
        "transitions": int(end["transitions"]),
        "raw_path": str(path),
        "raw_sha256": sha256_file(path),
    }


def _transition_row(record: dict[str, Any], transition: dict[str, Any]) -> dict[str, Any]:
    obs = transition["obs_t"]
    nxt = transition["obs_t1"]
    return {
        "transition_id": transition["transition_id"],
        "run_id": record["run_id"],
        "step_id": int(record["step_id"]),
        "phase": obs["phase"],
        "act": int(obs.get("context", {}).get("act") or 0),
        "floor": int(obs.get("context", {}).get("floor") or 0),
        "room_type": obs.get("context", {}).get("room_type"),
        "state_hash_t": obs["state_hash"],
        "state_hash_t1": nxt["state_hash"],
        "observation_json": canonical_json(obs["agent_observation"]),
        "legal_actions_json": canonical_json(transition["legal_actions_t"]),
        "action_json": canonical_json(transition["action_t"]),
        "outcome_json": canonical_json(transition["outcome"]),
        "next_observation_json": canonical_json(nxt["agent_observation"]),
        "audit_before_json": canonical_json(obs["audit_ref"]),
        "audit_after_json": canonical_json(nxt["audit_ref"]),
        "done": bool(transition["done"]),
    }


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", version="2.6")


EPISODE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()), ("seed", pa.string()), ("character", pa.string()),
        ("ascension", pa.int32()), ("policy_id", pa.string()), ("started_at", pa.string()),
        ("ended_at", pa.string()), ("terminal", pa.bool_()), ("victory", pa.bool_()),
        ("max_act", pa.int32()), ("max_floor", pa.int32()), ("transitions", pa.int32()),
        ("raw_path", pa.string()), ("raw_sha256", pa.string()),
    ]
)

TRANSITION_SCHEMA = pa.schema(
    [
        ("transition_id", pa.string()), ("run_id", pa.string()), ("step_id", pa.int32()),
        ("phase", pa.string()), ("act", pa.int32()), ("floor", pa.int32()),
        ("room_type", pa.string()), ("state_hash_t", pa.string()), ("state_hash_t1", pa.string()),
        ("observation_json", pa.large_string()), ("legal_actions_json", pa.large_string()),
        ("action_json", pa.large_string()), ("outcome_json", pa.large_string()),
        ("next_observation_json", pa.large_string()), ("audit_before_json", pa.large_string()),
        ("audit_after_json", pa.large_string()), ("done", pa.bool_()),
    ]
)

FIXTURE_SCHEMA = pa.schema(
    [("fixture_id", pa.string()), ("phase", pa.string()), ("source_path", pa.string()), ("payload_json", pa.large_string())]
)

