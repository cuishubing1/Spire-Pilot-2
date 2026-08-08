from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from jsonschema import Draft202012Validator

from .constants import DATA_ROOT, ROOT
from .util import canonical_json, sha256_file, utc_now, write_json_atomic

HUMAN_ROOT = DATA_ROOT / "human"
HUMAN_RAW_ROOT = HUMAN_ROOT / "raw"
HUMAN_DATASET_ROOT = HUMAN_ROOT / "dataset"
EXPECTED_ASSEMBLY_SHA256 = "a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52"
_HUMAN_V041_VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "schemas" / "human_live_record.schema.json").read_text(encoding="utf-8"))
)


class HumanRecordingError(ValueError):
    pass


def _hash_record(record: dict[str, Any]) -> str:
    unhashed = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(unhashed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_and_verify_recording(path: Path, *, require_complete: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected_sequence = 0
    previous = "0" * 64
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise HumanRecordingError(f"{path}:{line_no}: truncated final line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HumanRecordingError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if record.get("sequence") != expected_sequence:
                raise HumanRecordingError(f"{path}:{line_no}: sequence is not continuous")
            if record.get("prev_record_sha256") != previous:
                raise HumanRecordingError(f"{path}:{line_no}: broken previous-record hash")
            actual = _hash_record(record)
            if record.get("record_sha256") != actual:
                raise HumanRecordingError(f"{path}:{line_no}: record hash mismatch")
            records.append(record)
            expected_sequence += 1
            previous = actual
    kinds = [row.get("record_type") for row in records]
    if not records or kinds[:2] != ["recorder_start", "run_start"]:
        raise HumanRecordingError(f"{path}: missing recorder_start/run_start")
    if require_complete and kinds[-1] != "run_end":
        raise HumanRecordingError(f"{path}: recording is not sealed with run_end")
    start = records[0]["payload"]
    if start.get("schema_version") not in {
        "human-live-0.1.0", "human-live-0.2.0", "human-live-0.2.1", "human-live-0.2.2", "human-live-0.3.0",
        "human-live-0.3.1", "human-live-0.3.2", "human-live-0.4.0", "human-live-0.4.1"
    }:
        raise HumanRecordingError(f"{path}: unsupported live schema {start.get('schema_version')!r}")
    if start.get("schema_version") == "human-live-0.4.1":
        for index, record in enumerate(records, 1):
            errors = sorted(_HUMAN_V041_VALIDATOR.iter_errors(record), key=lambda error: list(error.path))
            if errors:
                location = "/".join(str(part) for part in errors[0].absolute_path)
                raise HumanRecordingError(f"{path}:{index}: HumanRecorder schema error at {location}: {errors[0].message}")
    game = start.get("game", {})
    if str(game.get("expected_game_version")) != "0.107.1" or str(game.get("expected_build")) != "23811903":
        raise HumanRecordingError(f"{path}: wrong game version/build")
    if str(game.get("assembly_sha256", "")).lower() != EXPECTED_ASSEMBLY_SHA256:
        raise HumanRecordingError(f"{path}: sts2.dll fingerprint mismatch")
    return records


def recover_recording(path: Path, destination: Path | None = None) -> dict[str, Any]:
    """Create a sealed copy from a crash-left .partial file without mutating the source."""
    records = read_and_verify_recording(path, require_complete=False)
    if records[-1]["record_type"] == "run_end":
        raise HumanRecordingError(f"{path}: recording is already sealed")
    destination = destination or path.with_name(path.name.removesuffix(".partial") + ".recovered")
    if destination.exists():
        raise HumanRecordingError(f"destination already exists: {destination}")
    shutil.copyfile(path, destination)
    run_id = records[1]["payload"].get("run_id")
    record: dict[str, Any] = {
        "payload": {"run_id": run_id, "reason": "recovered_after_crash", "won": None, "observation": None},
        "prev_record_sha256": records[-1]["record_sha256"],
        "record_type": "run_end",
        "sequence": int(records[-1]["sequence"]) + 1,
        "timestamp_utc": utc_now(),
    }
    record["record_sha256"] = _hash_record(record)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    read_and_verify_recording(destination)
    return {"status": "PASS", "source": str(path), "sealed_copy": str(destination), "records": len(records) + 1}


def audit_human_recording(source: Path) -> dict[str, Any]:
    """Audit raw recorder output without importing or mutating it."""
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(
            path for pattern in ("*.jsonl", "*.jsonl.partial", "*.jsonl.recovered")
            for path in source.glob(pattern) if path.is_file()
        )
    if not paths:
        raise HumanRecordingError(f"no HumanRecorder JSONL files found under {source}")
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    commit_status_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    total_decisions = 0
    for path in paths:
        records = read_and_verify_recording(path, require_complete=not path.name.endswith(".partial"))
        live_schema = str(records[0]["payload"].get("schema_version") or "")
        strict_action_args = live_schema.startswith(("human-live-0.3.", "human-live-0.4."))
        require_native_state = live_schema.startswith("human-live-0.4.")
        require_v041 = live_schema.startswith("human-live-0.4.1")
        decisions = [row for row in records if row["record_type"] == "decision"]
        total_decisions += len(decisions)
        for record in decisions:
            decision = record["payload"]
            phase = str(decision.get("phase") or "<missing>")
            action_id = str(decision.get("action", {}).get("action_id") or "<missing>")
            quality = _derived_capture_quality(decision, strict_action_args=strict_action_args,
                                               require_native_state=require_native_state,
                                               require_v041=require_v041)
            phase_counts[phase] += 1
            action_counts[action_id] += 1
            quality_counts[quality] += 1
            commit_status_counts[str(decision.get("commit_status") or "legacy")] += 1
            if quality != "complete":
                issues.append({
                    "path": str(path), "sequence": int(record["sequence"]),
                    "phase": phase, "action_id": action_id, "quality": quality,
                })
        file_rows.append({
            "path": str(path), "schema_version": live_schema, "decisions": len(decisions),
            "sealed": records[-1]["record_type"] == "run_end",
            "sha256": sha256_file(path),
        })
    return {
        "status": "PASS" if total_decisions > 0 and not issues else "FAIL",
        "files": file_rows, "decision_count": total_decisions,
        "phase_counts": dict(sorted(phase_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "commit_status_counts": dict(sorted(commit_status_counts.items())),
        "issue_count": len(issues), "issues": issues[:100],
        "issues_truncated": len(issues) > 100,
    }


def _recording_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.glob("*.jsonl") if path.is_file()) + sorted(
        path for path in source.glob("*.jsonl.recovered") if path.is_file()
    )


def import_human_recordings(source: Path, *, include_partial: bool = True) -> dict[str, Any]:
    paths = _recording_paths(source)
    if not paths:
        raise HumanRecordingError(f"no sealed JSONL recordings found under {source}")
    HUMAN_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    HUMAN_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    rollbacks: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    action_arg_error_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    commit_status_counts: Counter[str] = Counter()
    imported_raw: list[Path] = []

    for path in paths:
        records = read_and_verify_recording(path)
        meta = records[0]["payload"]
        live_schema = str(meta.get("schema_version", ""))
        strict_action_args = live_schema.startswith(("human-live-0.3.", "human-live-0.4."))
        require_native_state = live_schema.startswith("human-live-0.4.")
        require_v041 = live_schema.startswith("human-live-0.4.1")
        start = records[1]["payload"]
        run_context = start.get("run_context", {})
        end = records[-1]["payload"]
        environment_scope, loaded_mods = _environment_scope(meta)
        victory, victory_source = _derived_victory(end)
        decision_entries = [(int(row["sequence"]), row["payload"]) for row in records if row["record_type"] == "decision"]
        decisions = [row for _, row in decision_entries]
        rollback_entries = [row for row in records if row["record_type"] in {"rollback", "resume_unmatched"}]
        partial = [row for row in decisions if _derived_capture_quality(
            row, strict_action_args=strict_action_args, require_native_state=require_native_state,
            require_v041=require_v041
        ) != "complete"]
        if partial and not include_partial:
            raise HumanRecordingError(
                f"{path}: {len(partial)} partial decisions; repair hooks or pass --include-partial to quarantine-tag them"
            )
        run_id = str(start["run_id"])
        run_transition_start = len(transitions)
        raw_path = HUMAN_RAW_ROOT / f"{run_id}.jsonl.zst"
        if raw_path.exists():
            raise HumanRecordingError(f"immutable raw destination already exists: {raw_path}")
        compressor = zstd.ZstdCompressor(level=9)
        with path.open("rb") as src, raw_path.open("xb") as dst:
            compressor.copy_stream(src, dst)
        imported_raw.append(raw_path)

        invalidated_ranges: list[tuple[int, int, int]] = []
        quarantine_attempts: set[int] = set()
        rollback_by_from_sequence: dict[int, dict[str, Any]] = {}
        for event in rollback_entries:
            payload = event["payload"]
            rollback_id = int(payload.get("rollback_id") or 0)
            discarded = payload.get("discarded_decision_range")
            if event["record_type"] == "rollback" and isinstance(discarded, list) and len(discarded) == 2:
                invalidated_ranges.append((int(discarded[0]), int(discarded[1]), rollback_id))
            else:
                quarantine_attempts.add(int(payload.get("from_attempt_id") or 0))
                quarantine_attempts.add(int(payload.get("to_attempt_id") or 0))
            from_sequence = payload.get("from_decision_sequence")
            if from_sequence is not None:
                rollback_by_from_sequence[int(from_sequence)] = payload
            rollbacks.append(_rollback_row(run_id, event))

        def invalidating_rollback(sequence: int) -> int | None:
            matches = [rollback_id for start_seq, end_seq, rollback_id in invalidated_ranges if start_seq <= sequence <= end_seq]
            return matches[-1] if matches else None

        for index, (record_sequence, decision) in enumerate(decision_entries):
            obs = decision["observation"]
            attempt_id = int(decision.get("attempt_id") or 0)
            next_entry = decision_entries[index + 1] if index + 1 < len(decision_entries) else None
            rollback_event = rollback_by_from_sequence.get(record_sequence)
            if rollback_event is not None:
                next_obs = None
                termination = "rollback" if rollback_event.get("canonical_boundary") == "resolved" else "resume_unmatched"
            elif next_entry is not None and int(next_entry[1].get("attempt_id") or 0) == attempt_id:
                next_obs = next_entry[1]["observation"]
                termination = "continued"
            elif next_entry is not None:
                next_obs = None
                termination = "attempt_boundary"
            else:
                next_obs = end.get("observation")
                termination = "run_end"
            quality = _derived_capture_quality(decision, strict_action_args=strict_action_args,
                                               require_native_state=require_native_state,
                                               require_v041=require_v041)
            content_scope = _derived_content_scope(decision, quality)
            phase = str(decision["phase"])
            phase_counts[phase] += 1
            action_id = str(decision.get("action", {}).get("action_id") or "<missing>")
            action_counts[action_id] += 1
            if quality in {"partial_action_args", "partial_action_mismatch"}:
                action_arg_error_counts[action_id] += 1
            quality_counts[quality] += 1
            commit_status = str(decision.get("commit_status") or "legacy")
            commit_status_counts[commit_status] += 1
            run = obs.get("run", {})
            invalidated_by = invalidating_rollback(record_sequence)
            boundary_status = "quarantine" if attempt_id in quarantine_attempts else "resolved"
            canonical = invalidated_by is None and attempt_id not in quarantine_attempts
            training_eligible = canonical and quality == "complete" and content_scope in {"base_game", "legacy_unclassified"}
            strict_vanilla_eligible = training_eligible and environment_scope == "base_game"
            transitions.append(
                {
                    "transition_id": f"{run_id}:{index}", "run_id": run_id, "step_id": index,
                    "record_sequence": record_sequence, "attempt_id": attempt_id,
                    "phase": phase, "act": int(run.get("act") or 0), "floor": int(run.get("total_floor") or 0),
                    "room_type": run.get("room_type"), "capture_quality": quality,
                    "commit_status": commit_status,
                    "is_canonical": canonical,
                    "sl_contaminated": attempt_id > 0, "boundary_status": boundary_status,
                    "environment_scope": environment_scope, "content_scope": content_scope,
                    "is_training_eligible": training_eligible,
                    "strict_vanilla_eligible": strict_vanilla_eligible,
                    "exclusion_reason": _exclusion_reason(
                        canonical=canonical, quality=quality, content_scope=content_scope,
                        environment_scope=environment_scope, strict=False,
                    ),
                    "termination": termination, "invalidated_by_rollback_id": invalidated_by,
                    "observation_json": canonical_json(obs),
                    "legal_actions_json": canonical_json(obs.get("legal_actions", [])),
                    "action_json": canonical_json(decision["action"]),
                    "next_observation_json": canonical_json(next_obs),
                    "done": termination in {"rollback", "resume_unmatched", "attempt_boundary", "run_end"},
                    "policy_id": decision.get("policy_id", "human_v1"),
                }
            )
        max_act = max((int(row["observation"].get("run", {}).get("act") or 0) for row in decisions), default=0)
        max_floor = max((int(row["observation"].get("run", {}).get("total_floor") or 0) for row in decisions), default=0)
        episode_policy = "human_v4" if live_schema.startswith("human-live-0.4.1") else (
            "human_v3" if live_schema.startswith("human-live-0.4.0") else (
            "human_v2" if live_schema.startswith(("human-live-0.2.", "human-live-0.3.")) else "human_v1"
        ))
        run_transitions = transitions[run_transition_start:]
        episodes.append(
            {
                "run_id": run_id, "actor_id": meta["actor_id"], "policy_id": episode_policy,
                "seed": run_context.get("seed"), "seed_numeric": (
                    str(run_context["seed_numeric"]) if run_context.get("seed_numeric") is not None else None
                ),
                "character_ids_json": canonical_json(run_context.get("character_ids", [])),
                "ascension": int(run_context.get("ascension") or 0),
                "game_mode": run_context.get("game_mode"),
                "act_ids_json": canonical_json(run_context.get("act_ids", [])),
                "modifier_ids_json": canonical_json(run_context.get("modifier_ids", [])),
                "badge_ids_json": canonical_json(run_context.get("badge_ids", [])),
                "should_save": run_context.get("should_save"),
                "daily_time": run_context.get("daily_time"),
                "run_context_quality": run_context.get("capture_quality"),
                "started_at": records[0]["timestamp_utc"], "ended_at": records[-1]["timestamp_utc"],
                "terminal_reason": end.get("reason"), "victory": victory, "victory_source": victory_source,
                "max_act": max_act,
                "max_floor": max_floor, "transitions": len(decisions), "partial_transitions": len(partial),
                "environment_scope": environment_scope, "loaded_mods_json": canonical_json(loaded_mods),
                "training_eligible_transitions": sum(1 for row in run_transitions if row["is_training_eligible"]),
                "strict_vanilla_eligible_transitions": sum(1 for row in run_transitions if row["strict_vanilla_eligible"]),
                "rollback_count": len([x for x in rollback_entries if x["record_type"] == "rollback"]),
                "unmatched_resume_count": len([x for x in rollback_entries if x["record_type"] == "resume_unmatched"]),
                "raw_path": str(raw_path), "raw_sha256": sha256_file(raw_path),
            }
        )

    episode_path = HUMAN_DATASET_ROOT / "episodes.parquet"
    transition_path = HUMAN_DATASET_ROOT / "transitions.parquet"
    rollback_path = HUMAN_DATASET_ROOT / "rollbacks.parquet"
    pq.write_table(pa.Table.from_pylist(episodes, schema=HUMAN_EPISODE_SCHEMA), episode_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(transitions, schema=HUMAN_TRANSITION_SCHEMA), transition_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(rollbacks, schema=HUMAN_ROLLBACK_SCHEMA), rollback_path, compression="zstd")
    coverage = {
        "generated_at": utc_now(), "phase_counts": dict(sorted(phase_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "action_arg_error_counts": dict(sorted(action_arg_error_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())), "partial_included": include_partial,
        "commit_status_counts": dict(sorted(commit_status_counts.items())),
        "rollback_count": len([x for x in rollbacks if x["event_type"] == "rollback"]),
        "unmatched_resume_count": len([x for x in rollbacks if x["event_type"] == "resume_unmatched"]),
        "canonical_transition_count": sum(1 for x in transitions if x["is_canonical"]),
        "training_eligible_transition_count": sum(1 for x in transitions if x["is_training_eligible"]),
        "strict_vanilla_eligible_transition_count": sum(1 for x in transitions if x["strict_vanilla_eligible"]),
        "content_scope_counts": dict(sorted(Counter(x["content_scope"] for x in transitions).items())),
        "environment_scope_counts": dict(sorted(Counter(x["environment_scope"] for x in episodes).items())),
        "sl_contaminated_transition_count": sum(1 for x in transitions if x["sl_contaminated"]),
    }
    write_json_atomic(HUMAN_DATASET_ROOT / "coverage.json", coverage)
    files = [episode_path, transition_path, rollback_path, HUMAN_DATASET_ROOT / "coverage.json"]
    manifest = {
        "schema_version": "human-dataset-0.3.0", "generated_at": utc_now(),
        "episode_count": len(episodes), "transition_count": len(transitions), "rollback_count": len(rollbacks),
        "raw_files": [{"path": str(path), "sha256": sha256_file(path)} for path in imported_raw],
        "files": [{"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size} for path in files],
    }
    write_json_atomic(HUMAN_DATASET_ROOT / "manifest.json", manifest)
    return manifest


def _derived_capture_quality(decision: dict[str, Any], *, strict_action_args: bool = False,
                             require_native_state: bool = False, require_v041: bool = False) -> str:
    raw_quality = str(decision.get("capture_quality", "partial"))
    if raw_quality != "complete":
        if not _is_complete_observed_cancellation(decision):
            return raw_quality
    legal = decision.get("observation", {}).get("legal_actions", [])
    action = decision.get("action", {})
    action_id = action.get("action_id")
    if not legal or action_id not in {item.get("action_id") for item in legal}:
        return "partial_legal_mismatch"
    if strict_action_args and not _action_args_valid(action):
        return "partial_action_args"
    if strict_action_args and not _action_matches_legal(action, legal):
        return "partial_action_mismatch"
    if require_native_state and _native_state_validation_error(decision) is not None:
        return "partial_native_state"
    if require_v041 and _v041_semantic_error(decision) is not None:
        return "partial_v041_contract"
    return "complete"


def _is_complete_observed_cancellation(decision: dict[str, Any]) -> bool:
    """Recognize a fully captured return/cancel from a required card picker.

    The engine's MinSelect describes confirmation. It does not make the UI's
    separately observed CloseSelection action incomplete.
    """
    if decision.get("phase") != "card_select":
        return False
    action = decision.get("action")
    observation = decision.get("observation")
    if not isinstance(action, dict) or action.get("action_id") != "skip_card_selection":
        return False
    if not isinstance(observation, dict) or observation.get("capture_errors"):
        return False
    selection = observation.get("card_select")
    legal = observation.get("legal_actions")
    if not isinstance(selection, dict) or not isinstance(selection.get("cards"), list) or not selection["cards"]:
        return False
    if not isinstance(legal, list) or not any(
        isinstance(item, dict) and item.get("action_id") == "skip_card_selection" for item in legal
    ):
        return False
    audit = decision.get("audit_state")
    return (
        isinstance(audit, dict)
        and audit.get("capture_quality") == "complete"
        and decision.get("commit_status") in {"method_returned", "committed"}
    )


def _derived_content_scope(decision: dict[str, Any], quality: str) -> str:
    raw_scope = str(decision.get("content_scope") or "legacy_unclassified")
    if raw_scope != "unknown" or quality != "complete" or not _is_complete_observed_cancellation(decision):
        return raw_scope
    source_kinds: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            kind = value.get("source_kind")
            if isinstance(kind, str):
                source_kinds.append(kind)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(decision.get("observation"))
    collect(decision.get("action"))
    return "base_game" if source_kinds and all(kind == "base_game" for kind in source_kinds) else raw_scope


def _v041_semantic_error(decision: dict[str, Any]) -> str | None:
    if decision.get("commit_status") not in {"method_returned", "committed"}:
        return "commit_status"
    telemetry = decision.get("telemetry")
    if not isinstance(telemetry, dict) or not isinstance(telemetry.get("capture_ms"), (int, float)):
        return "telemetry"

    def card_error(value: Any) -> str | None:
        if isinstance(value, dict):
            if "energy_cost" in value and "state_schema" in value:
                if not value.get("lineage_id") or not value.get("engine_object_ref"):
                    return "card_lineage"
            for child in value.values():
                error = card_error(child)
                if error:
                    return error
        elif isinstance(value, list):
            for child in value:
                error = card_error(child)
                if error:
                    return error
        return None

    error = card_error(decision.get("observation"))
    if error:
        return error
    action = decision.get("action", {})
    args = action.get("args", {}) if isinstance(action, dict) else {}
    action_id = action.get("action_id") if isinstance(action, dict) else None
    if action_id in {"play_card", "choose_card", "choose_card_reward"} and not args.get("card_lineage_id"):
        return "action_card_lineage"
    if action_id == "confirm_card_selection":
        instance_ids = args.get("selected_card_instance_ids")
        lineage_ids = args.get("selected_card_lineage_ids")
        if not isinstance(instance_ids, list) or not isinstance(lineage_ids, list) or len(instance_ids) != len(lineage_ids):
            return "selected_card_lineage"
    return None


def _native_state_validation_error(decision: dict[str, Any]) -> str | None:
    observation = decision.get("observation")
    if not isinstance(observation, dict):
        return "observation_missing"
    if "audit_state" in observation or _contains_key(observation, "draw_pile_ordered"):
        return "audit_state_leaked_into_observation"
    if _contains_key(observation, "enchantment_id") or _contains_key(observation, "affliction_id"):
        return "legacy_flat_card_modifier_fields"
    audit_state = decision.get("audit_state")
    if not isinstance(audit_state, dict):
        return "audit_state_missing"
    if audit_state.get("schema_version") != "native-model-state-0.1.0":
        return "audit_state_schema"
    if audit_state.get("capture_quality") != "complete":
        return "audit_state_incomplete"
    return None


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


_REQUIRED_ACTION_ARGS: dict[str, tuple[str, ...]] = {
    "select_map_node": ("coord",),
    "play_card": ("card_instance_id",),
    "choose_event_option": ("index",),
    "choose_rest_option": ("index",),
    "choose_reward_alternative": ("index",),
    "choose_card_reward": ("card_id", "card_instance_id"),
    "choose_card": ("card_id", "card_instance_id"),
    "confirm_card_selection": ("selected_card_ids",),
    "select_bundle": ("bundle_index", "card_ids"),
    "choose_relic": ("relic_index", "relic_id", "relic_instance_id"),
    "buy_shop_item": ("index", "id", "cost"),
    "remove_card": ("cost",),
    "use_potion": ("potion_id", "potion_instance_id"),
    "discard_potion": ("potion_id", "potion_instance_id"),
    "select_reward": ("reward_index", "reward_type", "reward_id"),
    "select_treasure_relic": ("relic_index", "relic_id"),
}

_STABLE_ACTION_ARG_KEYS = {
    "coord", "card_instance_id", "target_index", "index", "selected_card_ids",
    "bundle_index", "card_ids", "id", "cost", "potion_instance_id",
    "reward_index", "reward_type", "reward_id", "relic_index", "relic_id", "relic_instance_id",
}


def _action_args_valid(action: dict[str, Any]) -> bool:
    action_id = str(action.get("action_id") or "")
    args = action.get("args")
    if not isinstance(args, dict):
        return False
    for key in _REQUIRED_ACTION_ARGS.get(action_id, ()):
        value = args.get(key)
        if value is None or value == "" or (key in {"card_ids"} and not value):
            return False
    coord = args.get("coord")
    if action_id == "select_map_node" and (
        not isinstance(coord, dict) or coord.get("col") is None or coord.get("row") is None
    ):
        return False
    for key in ("bundle_index", "reward_index", "relic_index"):
        if key in args and (not isinstance(args[key], int) or args[key] < 0):
            return False
    return True


def _action_matches_legal(action: dict[str, Any], legal: list[dict[str, Any]]) -> bool:
    action_id = action.get("action_id")
    action_args = action.get("args") if isinstance(action.get("args"), dict) else {}
    for candidate in (item for item in legal if item.get("action_id") == action_id):
        candidate_args = candidate.get("args") if isinstance(candidate.get("args"), dict) else {}
        common = _STABLE_ACTION_ARG_KEYS.intersection(action_args).intersection(candidate_args)
        if not common or all(action_args[key] == candidate_args[key] for key in common):
            return True
    return False


def _environment_scope(meta: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    environment = meta.get("environment")
    if not isinstance(environment, dict):
        return "unclassified", []
    loaded_mods = environment.get("loaded_mods")
    if not isinstance(loaded_mods, list):
        loaded_mods = []
    return ("modded" if bool(environment.get("has_content_mods")) else "base_game"), loaded_mods


def _derived_victory(end: dict[str, Any]) -> tuple[bool | None, str]:
    recorded = end.get("won")
    if isinstance(recorded, bool):
        return recorded, "recorded"
    reason = str(end.get("reason") or "")
    if reason == "abandoned":
        return False, "inferred_abandoned"
    observation = end.get("observation")
    player = observation.get("player", {}) if isinstance(observation, dict) else {}
    hp = player.get("hp") if isinstance(player, dict) else None
    if reason == "game_ended" and isinstance(hp, (int, float)):
        return (hp > 0), "inferred_game_ended_positive_hp" if hp > 0 else "inferred_game_ended_zero_hp"
    return None, "unknown"


def _exclusion_reason(*, canonical: bool, quality: str, content_scope: str,
                      environment_scope: str, strict: bool) -> str | None:
    if not canonical:
        return "rollback_or_resume_quarantine"
    if quality != "complete":
        return f"capture_{quality}"
    if content_scope == "modded":
        return "mod_content"
    if content_scope == "unknown":
        return "content_provenance_unknown"
    if strict and environment_scope != "base_game":
        return "environment_not_verified_vanilla"
    return None


def _rollback_row(run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    payload = event["payload"]
    discarded = payload.get("discarded_decision_range")
    return {
        "run_id": run_id, "rollback_id": int(payload.get("rollback_id") or 0),
        "record_sequence": int(event["sequence"]), "event_type": str(event["record_type"]),
        "from_attempt_id": int(payload.get("from_attempt_id") or 0),
        "to_attempt_id": int(payload.get("to_attempt_id") or 0),
        "from_decision_sequence": payload.get("from_decision_sequence"),
        "rollback_target_sequence": payload.get("rollback_target_sequence"),
        "rollback_target_attempt_id": payload.get("rollback_target_attempt_id"),
        "discarded_start_sequence": discarded[0] if isinstance(discarded, list) and len(discarded) == 2 else None,
        "discarded_end_sequence": discarded[1] if isinstance(discarded, list) and len(discarded) == 2 else None,
        "match_quality": payload.get("match_quality"), "match_confidence": payload.get("match_confidence"),
        "canonical_boundary": payload.get("canonical_boundary"), "room_key": payload.get("room_key"),
        "payload_json": canonical_json(payload),
    }


def validate_human_dataset() -> dict[str, Any]:
    manifest_path = HUMAN_DATASET_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise HumanRecordingError("human dataset manifest does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["raw_files"] + manifest["files"]:
        path = Path(row["path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            raise HumanRecordingError(f"missing or modified human dataset file: {path}")
    transitions = pq.read_table(HUMAN_DATASET_ROOT / "transitions.parquet").to_pylist()
    if len(transitions) != manifest["transition_count"]:
        raise HumanRecordingError("human transition count mismatch")
    bad_actions = 0
    bad_rollback_links = 0
    bad_training_eligibility = 0
    for row in transitions:
        legal = json.loads(row["legal_actions_json"])
        action = json.loads(row["action_json"])
        action_id = action.get("action_id")
        if row["capture_quality"] == "complete" and legal and action_id not in {item.get("action_id") for item in legal}:
            # Semantic sub-actions such as a specific chosen card may be encoded with richer args;
            # membership is checked by stable action id here and arguments remain auditable.
            bad_actions += 1
        if row["termination"] in {"rollback", "resume_unmatched", "attempt_boundary"} and json.loads(row["next_observation_json"]) is not None:
            bad_rollback_links += 1
        if row["is_training_eligible"] and (
            not row["is_canonical"] or row["capture_quality"] != "complete"
            or row["content_scope"] not in {"base_game", "legacy_unclassified"}
        ):
            bad_training_eligibility += 1
        if row["strict_vanilla_eligible"] and (
            not row["is_training_eligible"] or row["environment_scope"] != "base_game"
        ):
            bad_training_eligibility += 1
    if bad_actions:
        raise HumanRecordingError(f"{bad_actions} complete transitions have actions outside legal_actions")
    if bad_rollback_links:
        raise HumanRecordingError(f"{bad_rollback_links} rollback boundaries were linked as normal transitions")
    if bad_training_eligibility:
        raise HumanRecordingError(f"{bad_training_eligibility} transitions violate Mod/quality isolation rules")
    return {"status": "PASS", "episodes": manifest["episode_count"], "transitions": len(transitions),
            "rollbacks": manifest.get("rollback_count", 0), "bad_actions": 0, "bad_rollback_links": 0,
            "bad_training_eligibility": 0}


HUMAN_EPISODE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()), ("actor_id", pa.string()), ("policy_id", pa.string()),
        ("seed", pa.string()), ("seed_numeric", pa.string()),
        ("character_ids_json", pa.large_string()), ("ascension", pa.int32()), ("game_mode", pa.string()),
        ("act_ids_json", pa.large_string()), ("modifier_ids_json", pa.large_string()),
        ("badge_ids_json", pa.large_string()), ("should_save", pa.bool_()), ("daily_time", pa.string()),
        ("run_context_quality", pa.string()),
        ("started_at", pa.string()), ("ended_at", pa.string()), ("terminal_reason", pa.string()),
        ("victory", pa.bool_()), ("victory_source", pa.string()),
        ("max_act", pa.int32()), ("max_floor", pa.int32()),
        ("transitions", pa.int32()), ("partial_transitions", pa.int32()),
        ("environment_scope", pa.string()), ("loaded_mods_json", pa.large_string()),
        ("training_eligible_transitions", pa.int32()), ("strict_vanilla_eligible_transitions", pa.int32()),
        ("rollback_count", pa.int32()), ("unmatched_resume_count", pa.int32()),
        ("raw_path", pa.string()), ("raw_sha256", pa.string()),
    ]
)

HUMAN_TRANSITION_SCHEMA = pa.schema(
    [
        ("transition_id", pa.string()), ("run_id", pa.string()), ("step_id", pa.int32()),
        ("record_sequence", pa.int64()), ("attempt_id", pa.int32()),
        ("phase", pa.string()), ("act", pa.int32()), ("floor", pa.int32()), ("room_type", pa.string()),
        ("capture_quality", pa.string()), ("commit_status", pa.string()),
        ("is_canonical", pa.bool_()), ("sl_contaminated", pa.bool_()),
        ("boundary_status", pa.string()), ("termination", pa.string()), ("invalidated_by_rollback_id", pa.int32()),
        ("environment_scope", pa.string()), ("content_scope", pa.string()),
        ("is_training_eligible", pa.bool_()), ("strict_vanilla_eligible", pa.bool_()),
        ("exclusion_reason", pa.string()),
        ("observation_json", pa.large_string()),
        ("legal_actions_json", pa.large_string()), ("action_json", pa.large_string()),
        ("next_observation_json", pa.large_string()), ("done", pa.bool_()), ("policy_id", pa.string()),
    ]
)

HUMAN_ROLLBACK_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()), ("rollback_id", pa.int32()), ("record_sequence", pa.int64()),
        ("event_type", pa.string()), ("from_attempt_id", pa.int32()), ("to_attempt_id", pa.int32()),
        ("from_decision_sequence", pa.int64()), ("rollback_target_sequence", pa.int64()),
        ("rollback_target_attempt_id", pa.int32()), ("discarded_start_sequence", pa.int64()),
        ("discarded_end_sequence", pa.int64()), ("match_quality", pa.string()),
        ("match_confidence", pa.float64()), ("canonical_boundary", pa.string()),
        ("room_key", pa.string()), ("payload_json", pa.large_string()),
    ]
)
