from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import CONFIG_PATH, DOTNET, ENGINE_PROJECT, LOCK_PATH, ROOT, THIRD_PARTY
from .util import command_version, load_json, platform_info, sha256_file, utc_now, write_json_atomic


class VersionGateError(RuntimeError):
    pass


def verify_game(config: dict[str, Any]) -> dict[str, Any]:
    game_dir = Path(config["game_dir"])
    release_path = game_dir / "release_info.json"
    dll_path = game_dir / "data_sts2_windows_x86_64" / "sts2.dll"
    if not release_path.exists() or not dll_path.exists():
        raise VersionGateError(f"Game files missing below {game_dir}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(dll_path)
    expected = {
        "version": config["game_version"],
        "commit": config["game_commit"],
        "main_assembly_hash": config["main_assembly_hash"],
    }
    for key, value in expected.items():
        if release.get(key) != value:
            raise VersionGateError(f"release_info {key}: expected {value!r}, got {release.get(key)!r}")
    if actual_hash != config["sts2_dll_sha256"]:
        raise VersionGateError(f"sts2.dll SHA-256 mismatch: {actual_hash}")
    return {"release_info": release, "sts2_dll_path": str(dll_path), "sts2_dll_sha256": actual_hash}


def verify_toolchain() -> dict[str, Any]:
    if not DOTNET.exists():
        raise VersionGateError(f"Project .NET SDK is missing: {DOTNET}")
    version = command_version([str(DOTNET), "--version"])
    if not version.startswith("9."):
        raise VersionGateError(f"Expected project-local .NET 9 SDK, got {version}")
    return {"dotnet": version, **platform_info()}


def setup_engine(config: dict[str, Any]) -> None:
    verify_game(config)
    verify_toolchain()
    bash = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"
    if not Path(bash).exists():
        raise VersionGateError("Git Bash is required to run the pinned sts2-cli setup")
    data_dir = Path(config["game_dir"]) / "data_sts2_windows_x86_64"
    env = dict(__import__("os").environ)
    env["PATH"] = str(DOTNET.parent) + __import__("os").pathsep + env.get("PATH", "")
    result = subprocess.run(
        [bash, "setup.sh", str(data_dir).replace("\\", "/")],
        cwd=str(THIRD_PARTY),
        env=env,
        text=True,
    )
    if result.returncode:
        raise VersionGateError(f"sts2-cli setup failed with exit code {result.returncode}")
    # setup.sh pipes build output through tail without pipefail, so its exit code cannot
    # be trusted on Windows. Re-run the build directly and require the executable.
    build = subprocess.run(
        [str(DOTNET), "build", str(ENGINE_PROJECT), "--no-restore"],
        cwd=str(THIRD_PARTY),
        env=env,
        text=True,
    )
    engine_dll = THIRD_PARTY / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
    if build.returncode or not engine_dll.exists():
        raise VersionGateError(f"sts2-cli build gate failed with exit code {build.returncode}")


def create_environment_lock(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_json(config_path)
    collector_files = [
        ROOT / "config" / "dataset_v1.json",
        ROOT / "config" / "seeds_v1.txt",
        ROOT / "schemas" / "raw_record.schema.json",
        *sorted((ROOT / "src" / "sts2_dataset").glob("*.py")),
    ]
    lock = {
        "created_at": utc_now(),
        "schema_version": config["schema_version"],
        "dataset_version": config["dataset_version"],
        "game": verify_game(config),
        "sts2_cli": {
            "url": config["sts2_cli_url"],
            "commit": config["sts2_cli_commit"],
            "protocol": config["sts2_cli_protocol"],
            "source_archive_sha256": sha256_file(THIRD_PARTY.parent.parent / "sts2-cli.zip")
            if (THIRD_PARTY.parent.parent / "sts2-cli.zip").exists()
            else None,
            "local_patches": [
                "stable visible entity IDs",
                "v0.107.1 async save-load compatibility",
                "headless ModManager initialization",
                "stable get_state protocol command",
                "relic purchase synchronization context",
                "asynchronous event reward settling",
                "event-task decision-boundary synchronization",
                "treasure voting-session cleanup at map boundaries",
            ],
            "patched_source_sha256": sha256_file(THIRD_PARTY / "src" / "Sts2Headless" / "RunSimulator.cs"),
            "protocol_source_sha256": sha256_file(THIRD_PARTY / "src" / "Sts2Headless" / "Program.cs"),
            "patched_game_dll_sha256": sha256_file(THIRD_PARTY / "lib" / "sts2.dll"),
            "original_copied_game_dll_sha256": sha256_file(THIRD_PARTY / "lib" / "sts2.dll.original"),
            "engine_binary_sha256": sha256_file(
                THIRD_PARTY / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
            ),
        },
        "toolchain": verify_toolchain(),
        "python_requirements_lock_sha256": sha256_file(config_path.parent.parent / "requirements.lock"),
        "collector_source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in collector_files
        },
        "engine_project": str(ENGINE_PROJECT),
    }
    write_json_atomic(LOCK_PATH, lock)
    return lock


def require_lock(config: dict[str, Any]) -> dict[str, Any]:
    if not LOCK_PATH.exists():
        raise VersionGateError("environment.lock.json is missing; run lock-environment")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("dataset_version") != config["dataset_version"]:
        raise VersionGateError("Environment lock belongs to another dataset version")
    verify_game(config)
    verify_toolchain()
    expected_source = lock.get("sts2_cli", {}).get("patched_source_sha256")
    actual_source = sha256_file(THIRD_PARTY / "src" / "Sts2Headless" / "RunSimulator.cs")
    if expected_source != actual_source:
        raise VersionGateError("Environment lock is stale: patched engine source changed")
    expected_binary = lock.get("sts2_cli", {}).get("engine_binary_sha256")
    actual_binary = sha256_file(
        THIRD_PARTY / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
    )
    if expected_binary != actual_binary:
        raise VersionGateError("Environment lock is stale: engine binary changed")
    for relative, expected_hash in lock.get("collector_source_sha256", {}).items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected_hash:
            raise VersionGateError(f"Environment lock is stale: collector source changed ({relative})")
    return lock
