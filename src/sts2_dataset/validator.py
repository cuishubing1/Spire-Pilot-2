from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from jsonschema import Draft202012Validator

from .constants import AUDIT_ROOT, DATASET_ROOT, FIXTURE_ROOT, FORBIDDEN_AGENT_KEYS, KNOWN_PHASES, LOCK_PATH, RAW_ROOT, ROOT
from .engine import Sts2Engine
from .exporter import EPISODE_SCHEMA, FIXTURE_SCHEMA, TRANSITION_SCHEMA
from .normalize import normalize_observation
from .types import AuditRef
from .util import iter_jsonl_zst, load_json, sha256_file, write_json_atomic, utc_now


class ValidationFailure(RuntimeError):
    pass


def validate_dataset(config: dict[str, Any], *, acceptance: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    transition_count = 0
    runs = 0
    phases: Counter[str] = Counter()
    transitions_for_replay: list[dict[str, Any]] = []
    schema_record_count = 0
    seen_seeds: set[str] = set()
    raw_paths = sorted(RAW_ROOT.glob("*.jsonl.zst"))
    raw_schema = Draft202012Validator(load_json(ROOT / "schemas" / "raw_record.schema.json"))
    expected_fingerprint = {
        "game_version": config["game_version"],
        "steam_build_id": config["steam_build_id"],
        "sts2_cli_commit": config["sts2_cli_commit"],
        "sts2_cli_protocol": config["sts2_cli_protocol"],
        "sts2_dll_sha256": config["sts2_dll_sha256"],
    }

    for path in raw_paths:
        records = list(iter_jsonl_zst(path))
        _validate_json_records(path, records, raw_schema, errors)
        schema_record_count += len(records)
        if not records or records[0].get("record_type") != "run_start":
            errors.append(f"{path.name}: missing run_start")
            continue
        if records[-1].get("record_type") != "run_end":
            errors.append(f"{path.name}: missing run_end")
            continue
        runs += 1
        start = records[0]
        for key, expected in (
            ("dataset_version", config["dataset_version"]),
            ("character", config["character"]),
            ("ascension", config["ascension"]),
            ("policy_id", config["policy_id"]),
        ):
            if start.get(key) != expected:
                errors.append(f"{path.name}: run_start {key} mismatch")
        seed = str(start.get("seed"))
        if seed in seen_seeds:
            errors.append(f"{path.name}: duplicate natural-run seed {seed}")
        seen_seeds.add(seed)
        expected_sequence = list(range(len(records)))
        actual_sequence = [record.get("sequence_no") for record in records]
        if actual_sequence != expected_sequence:
            errors.append(f"{path.name}: non-contiguous sequence numbers")
        for record in records:
            if record.get("schema_version") != config["schema_version"]:
                errors.append(f"{path.name}: schema version mismatch at {record.get('sequence_no')}")
            if record.get("record_type") == "auto_transition":
                phases[record["phase"]] += 1
            if record.get("record_type") != "decision":
                continue
            obs = record.get("observation") or {}
            if obs.get("game_fingerprint") != expected_fingerprint:
                errors.append(f"{path.name}:{record.get('step_id')}: game fingerprint mismatch")
            phase = obs.get("phase")
            phases[str(phase)] += 1
            if phase not in KNOWN_PHASES:
                errors.append(f"{path.name}: unknown phase {phase!r}")
            leaks = sorted(_find_forbidden(obs.get("agent_observation"), FORBIDDEN_AGENT_KEYS))
            if leaks:
                errors.append(f"{path.name}:{record.get('step_id')}: hidden keys {leaks}")
            action = record.get("action")
            if action is not None:
                legal = {a["action_id"] for a in obs.get("legal_actions", [])}
                if action.get("action_id") not in legal:
                    errors.append(f"{path.name}:{record.get('step_id')}: illegal action")
            transition = record.get("transition")
            if transition:
                transition_count += 1
                transitions_for_replay.append(transition)
                before_ref = transition["obs_t"]["audit_ref"]
                after_ref = transition["obs_t1"]["audit_ref"]
                for ref in (before_ref, after_ref):
                    audit_path = AUDIT_ROOT.parent / ref["path"]
                    if not audit_path.exists():
                        errors.append(f"Missing audit checkpoint {audit_path}")
                    elif sha256_file(audit_path) != ref["sha256"]:
                        errors.append(f"Audit hash mismatch {audit_path}")

    manifest_path = DATASET_ROOT / "manifest.json"
    coverage_path = DATASET_ROOT / "coverage.json"
    if not manifest_path.exists():
        errors.append("manifest.json is missing; run export")
    else:
        manifest = load_json(manifest_path)
        if manifest.get("natural_run_count") != runs:
            errors.append("Manifest natural_run_count differs from raw runs")
        if manifest.get("transition_count") != transition_count:
            errors.append("Manifest transition_count differs from raw transitions")
        if manifest.get("environment_lock_sha256") != sha256_file(LOCK_PATH):
            errors.append("Manifest environment lock hash mismatch")
        for entry in [*manifest.get("raw_files", []), *manifest.get("files", [])]:
            listed_path = Path(entry["path"])
            if not listed_path.exists():
                errors.append(f"Manifest-listed file missing: {listed_path}")
            elif sha256_file(listed_path) != entry.get("sha256"):
                errors.append(f"Manifest-listed file hash mismatch: {listed_path}")
    fixture_records = []
    for fixture_path in sorted(FIXTURE_ROOT.glob("*.jsonl.zst")):
        records = list(iter_jsonl_zst(fixture_path))
        _validate_json_records(fixture_path, records, raw_schema, errors)
        schema_record_count += len(records)
        fixture_records.extend(r for r in records if r.get("record_type") in {"fixture", "auto_transition"})

    parquet_expectations = (
        ("episodes.parquet", runs, EPISODE_SCHEMA),
        ("transitions.parquet", transition_count, TRANSITION_SCHEMA),
        ("fixtures.parquet", len(fixture_records), FIXTURE_SCHEMA),
    )
    for parquet_name, expected_rows, expected_schema in parquet_expectations:
        parquet_path = DATASET_ROOT / parquet_name
        if not parquet_path.exists():
            errors.append(f"{parquet_name} is missing")
        else:
            if pq.read_metadata(parquet_path).num_rows != expected_rows:
                errors.append(f"{parquet_name} row count mismatch")
            if not pq.read_schema(parquet_path).equals(expected_schema, check_metadata=True):
                errors.append(f"{parquet_name} Arrow schema mismatch")

    if acceptance:
        required_runs = int(config["required_natural_runs"])
        if runs != required_runs:
            errors.append(f"Acceptance requires exactly {required_runs} natural runs, found {runs}")
        if coverage_path.exists():
            coverage = load_json(coverage_path)
            missing = coverage.get("missing_required_phases", [])
            if missing:
                errors.append(f"Missing required phases: {missing}")
        else:
            errors.append("coverage.json is missing")
        replay = _validate_replays(config, transitions_for_replay, errors)
    else:
        replay = {"restore_checks": 0, "action_replay_checks": 0, "boundary_pairs": 0}

    report = {
        "status": "PASS" if not errors else "FAIL",
        "validated_at": utc_now(),
        "natural_runs": runs,
        "transitions": transition_count,
        "phase_counts_from_natural": dict(sorted(phases.items())),
        "errors": errors,
        "warnings": warnings,
        "schema_validation": {
            "json_schema_draft": "2020-12",
            "json_records": schema_record_count,
            "arrow_schemas": 3,
        },
        "replay": replay,
    }
    write_json_atomic(DATASET_ROOT / "validation_report.json", report)
    if errors:
        raise ValidationFailure("\n".join(errors))
    return report


def _validate_replays(
    config: dict[str, Any], transitions: list[dict[str, Any]], errors: list[str]
) -> dict[str, int]:
    if not transitions:
        errors.append("No transitions available for replay validation")
        return {"restore_checks": 0, "action_replay_checks": 0, "boundary_pairs": 0}
    by_boundary: dict[tuple[str, str], dict[str, Any]] = {}
    for transition in transitions:
        key = (transition["obs_t"]["phase"], transition["obs_t1"]["phase"])
        by_boundary.setdefault(key, transition)
    selected = list(by_boundary.values())
    remaining = [t for t in transitions if t not in selected]
    rng = random.Random(config["dataset_version"])
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, 100 - len(selected))])
    selected = selected[: max(100, len(by_boundary))]

    engine = Sts2Engine(config, "acceptance-replay")
    restore_checks = action_checks = 0
    try:
        for index, transition in enumerate(selected):
            obs = transition["obs_t"]
            ref = AuditRef(**obs["audit_ref"])
            try:
                raw = engine.restore(ref)
                raw = engine.get_state()
                visible_map = engine.get_map() if raw.get("decision") == "map_select" else None
                restored = normalize_observation(
                    raw,
                    config=config,
                    run_id=obs["run_id"],
                    step_id=obs["step_id"],
                    audit_ref=None,
                    visible_map=visible_map,
                )
                if restored.state_hash != obs["state_hash"]:
                    errors.append(
                        f"Restore hash mismatch {transition['transition_id']}: "
                        f"{restored.state_hash} != {obs['state_hash']}"
                    )
                    continue
                restore_checks += 1
                if index < 20:
                    raw_next = engine.step(transition["action_t"])
                    next_map = engine.get_map() if raw_next.get("decision") == "map_select" else None
                    expected_next = transition["obs_t1"]
                    replayed_next = normalize_observation(
                        raw_next,
                        config=config,
                        run_id=expected_next["run_id"],
                        step_id=expected_next["step_id"],
                        audit_ref=None,
                        visible_map=next_map,
                    )
                    if replayed_next.state_hash != expected_next["state_hash"]:
                        errors.append(f"Action replay hash mismatch {transition['transition_id']}")
                    else:
                        action_checks += 1
            except Exception as exc:
                errors.append(f"Replay failed {transition['transition_id']}: {exc}")
    finally:
        engine.close()
    return {
        "restore_checks": restore_checks,
        "action_replay_checks": action_checks,
        "boundary_pairs": len(by_boundary),
    }


def _find_forbidden(value: Any, forbidden: set[str], prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in forbidden:
                yield path
            yield from _find_forbidden(item, forbidden, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _find_forbidden(item, forbidden, f"{prefix}[{index}]")


def _validate_json_records(
    path: Path,
    records: list[dict[str, Any]],
    validator: Draft202012Validator,
    errors: list[str],
) -> None:
    for record in records:
        for error in validator.iter_errors(record):
            location = ".".join(str(part) for part in error.absolute_path) or "root"
            errors.append(
                f"{path.name}:{record.get('sequence_no')} JSON Schema {location}: {error.message}"
            )
