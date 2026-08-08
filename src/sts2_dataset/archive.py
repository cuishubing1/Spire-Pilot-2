from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

from .constants import ARCHIVE_ROOT, DATASET_ROOT, LOCK_PATH, RAW_ROOT
from .engine import Sts2Engine
from .normalize import normalize_observation
from .smoke import run_smoke
from .types import AuditRef
from .util import (
    canonical_json,
    iter_jsonl_zst,
    load_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .versioning import verify_game


class ArchiveError(RuntimeError):
    pass


def default_archive_path(config: dict[str, Any], destination: Path = ARCHIVE_ROOT) -> Path:
    version = str(config["game_version"]).replace("/", "-")
    return destination / f"sts2-{version}-build-{config['steam_build_id']}"


def archive_game(config: dict[str, Any], destination: Path = ARCHIVE_ROOT) -> dict[str, Any]:
    source = Path(config["game_dir"]).resolve()
    verify_game(config)
    target = default_archive_path(config, destination.resolve())
    if target.exists():
        report = verify_archive(config, target)
        report["status"] = "already_archived"
        return report

    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in source_files)
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < total_bytes + 1024**3:
        raise ArchiveError(
            f"Insufficient free space: need at least {total_bytes + 1024**3}, have {free_bytes}"
        )

    partial = destination / f".{target.name}.partial-{int(time.time())}"
    if partial.exists():
        raise ArchiveError(f"Partial archive already exists: {partial}")
    game_target = partial / "game"
    metadata_target = partial / "metadata"
    game_target.mkdir(parents=True)
    metadata_target.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    try:
        for index, source_path in enumerate(source_files, 1):
            relative = source_path.relative_to(source)
            copied_path = game_target / relative
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(source_files)}] copying {relative}", flush=True)
            shutil.copy2(source_path, copied_path)
            source_hash = sha256_file(source_path)
            copied_hash = sha256_file(copied_path)
            if copied_hash != source_hash:
                raise ArchiveError(f"Copy hash mismatch: {relative}")
            entries.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "size": copied_path.stat().st_size,
                    "sha256": copied_hash,
                }
            )

        file_manifest = {
            "algorithm": "SHA-256",
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "files": entries,
        }
        write_json_atomic(metadata_target / "files.sha256.json", file_manifest)
        file_manifest_hash = sha256_file(metadata_target / "files.sha256.json")
        archive_digest = sha256_bytes(canonical_json(entries).encode("utf-8"))
        steam = _sanitized_steam_metadata(source, config)
        archive_manifest = {
            "format": "sts2-private-game-archive-v1",
            "created_at": utc_now(),
            "source_path": str(source),
            "game_version": config["game_version"],
            "game_commit": config["game_commit"],
            "steam_build_id": config["steam_build_id"],
            "sts2_dll_sha256": config["sts2_dll_sha256"],
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "files_manifest_sha256": file_manifest_hash,
            "archive_content_digest": archive_digest,
            "steam": steam,
            "distribution": "private-personal-debug-only",
        }
        write_json_atomic(metadata_target / "archive.json", archive_manifest)
        if LOCK_PATH.exists():
            shutil.copy2(LOCK_PATH, metadata_target / "environment.lock.json")
        shutil.copy2(source / "release_info.json", metadata_target / "release_info.json")
        partial.rename(target)
    except Exception:
        print(f"Partial archive retained for diagnosis: {partial}", flush=True)
        raise

    return verify_archive(config, target)


def verify_archive(config: dict[str, Any], archive_path: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    metadata = archive_path / "metadata"
    game_dir = archive_path / "game"
    archive_manifest_path = metadata / "archive.json"
    files_manifest_path = metadata / "files.sha256.json"
    if not archive_manifest_path.exists() or not files_manifest_path.exists():
        raise ArchiveError(f"Incomplete archive: {archive_path}")
    archive_manifest = load_json(archive_manifest_path)
    files_manifest = load_json(files_manifest_path)
    errors: list[str] = []
    checked_bytes = 0
    for entry in files_manifest["files"]:
        path = game_dir / entry["path"]
        if not path.exists():
            errors.append(f"missing:{entry['path']}")
            continue
        checked_bytes += path.stat().st_size
        if path.stat().st_size != entry["size"]:
            errors.append(f"size:{entry['path']}")
        elif sha256_file(path) != entry["sha256"]:
            errors.append(f"sha256:{entry['path']}")
    if sha256_file(files_manifest_path) != archive_manifest["files_manifest_sha256"]:
        errors.append("files_manifest_sha256")
    digest = sha256_bytes(canonical_json(files_manifest["files"]).encode("utf-8"))
    if digest != archive_manifest["archive_content_digest"]:
        errors.append("archive_content_digest")

    archive_config = dict(config)
    archive_config["game_dir"] = str(game_dir)
    try:
        verify_game(archive_config)
    except Exception as exc:
        errors.append(f"version_gate:{exc}")
    if errors:
        raise ArchiveError("Archive verification failed: " + ", ".join(errors[:20]))
    return {
        "status": "PASS",
        "archive_path": str(archive_path),
        "game_dir": str(game_dir),
        "file_count": files_manifest["file_count"],
        "total_bytes": files_manifest["total_bytes"],
        "checked_bytes": checked_bytes,
        "archive_content_digest": archive_manifest["archive_content_digest"],
        "files_manifest_sha256": archive_manifest["files_manifest_sha256"],
        "verified_at": utc_now(),
    }


def test_archive_replay(
    config: dict[str, Any], archive_path: Path, *, phases: tuple[str, ...] = ("map_select", "combat_play", "card_reward")
) -> dict[str, Any]:
    archive_report = verify_archive(config, archive_path)
    archive_config = dict(config)
    archive_config["game_dir"] = str((archive_path.resolve() / "game"))
    smoke = run_smoke(archive_config)
    transitions = _load_transitions()
    selected = []
    for phase in phases:
        candidates = [item for item in transitions if item["obs_t"]["phase"] == phase]
        if not candidates:
            raise ArchiveError(f"Dataset has no transition for phase {phase}")
        selected.append(candidates[len(candidates) // 2])

    engine = Sts2Engine(archive_config, "archive-dataset-replay")
    samples: list[dict[str, Any]] = []
    try:
        for transition in selected:
            before = transition["obs_t"]
            expected_next = transition["obs_t1"]
            raw = engine.restore(AuditRef(**before["audit_ref"]))
            raw = engine.get_state()
            visible_map = engine.get_map() if raw.get("decision") == "map_select" else None
            restored = normalize_observation(
                raw,
                config=archive_config,
                run_id=before["run_id"],
                step_id=before["step_id"],
                audit_ref=None,
                visible_map=visible_map,
            )
            restore_match = restored.state_hash == before["state_hash"]
            raw_next = engine.step(transition["action_t"])
            next_map = engine.get_map() if raw_next.get("decision") == "map_select" else None
            replayed_next = normalize_observation(
                raw_next,
                config=archive_config,
                run_id=expected_next["run_id"],
                step_id=expected_next["step_id"],
                audit_ref=None,
                visible_map=next_map,
            )
            action_match = replayed_next.state_hash == expected_next["state_hash"]
            samples.append(
                {
                    "transition_id": transition["transition_id"],
                    "phase": before["phase"],
                    "action": transition["action_t"]["action"],
                    "restore_hash_match": restore_match,
                    "action_replay_hash_match": action_match,
                    "expected_state_hash": before["state_hash"],
                    "restored_state_hash": restored.state_hash,
                    "expected_next_state_hash": expected_next["state_hash"],
                    "replayed_next_state_hash": replayed_next.state_hash,
                }
            )
    finally:
        engine.close()

    failures = [sample for sample in samples if not sample["restore_hash_match"] or not sample["action_replay_hash_match"]]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "tested_at": utc_now(),
        "archive": archive_report,
        "smoke": smoke,
        "dataset_manifest_sha256": sha256_file(DATASET_ROOT / "manifest.json"),
        "samples": samples,
        "failures": failures,
    }
    write_json_atomic(DATASET_ROOT / "archive_restore_report.json", report)
    if failures:
        raise ArchiveError(f"Archive dataset replay failed for {len(failures)} samples")
    return report


def _load_transitions() -> list[dict[str, Any]]:
    transitions = []
    for path in sorted(RAW_ROOT.glob("*.jsonl.zst")):
        for record in iter_jsonl_zst(path):
            if record.get("record_type") == "decision" and record.get("transition"):
                transitions.append(record["transition"])
    if not transitions:
        raise ArchiveError("No Dataset V1 transitions found")
    return transitions


def _sanitized_steam_metadata(source: Path, config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "appid": "2868840",
        "buildid": str(config["steam_build_id"]),
    }
    steamapps = source.parent.parent
    manifests = sorted(steamapps.glob("appmanifest_*.acf"))
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8", errors="replace")
        install_dir = _vdf_value(text, "installdir")
        if install_dir != source.name:
            continue
        result["appid"] = _vdf_value(text, "appid") or result["appid"]
        result["buildid"] = _vdf_value(text, "buildid") or result["buildid"]
        depot_match = re.search(
            r'"InstalledDepots"\s*\{\s*"(?P<depot>\d+)"\s*\{(?P<body>.*?)\}\s*\}',
            text,
            flags=re.DOTALL,
        )
        if depot_match:
            result["depot_id"] = depot_match.group("depot")
            result["depot_manifest"] = _vdf_value(depot_match.group("body"), "manifest")
        break
    return result


def _vdf_value(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s+"([^"]*)"', text)
    return match.group(1) if match else None
