from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .archive import archive_game, test_archive_replay, verify_archive
from .collector import Collector
from .combat_dataset import build_combat_dataset, validate_combat_dataset
from .combat_contract import build_combat_model_examples, validate_combat_model_examples
from .combat_tensorizer import build_combat_vocabulary
from .combat_value import build_combat_value_targets, validate_combat_value_targets
from .constants import ARCHIVE_ROOT, CONFIG_PATH
from .exporter import export_dataset
from .fixtures import collect_fixtures
from .human import audit_human_recording, import_human_recordings, recover_recording, validate_human_dataset
from .smoke import run_smoke
from .util import load_json, write_json_atomic
from .validator import ValidationFailure, validate_dataset
from .versioning import create_environment_lock, require_lock, setup_engine


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sts2-data", description="STS2 v0.107.1 Dataset V1 pipeline")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup-engine", help="Copy and patch game DLLs, then build the pinned headless engine")
    sub.add_parser("lock-environment", help="Verify and write environment.lock.json")
    sub.add_parser("smoke", help="Run the mandatory real-engine compatibility gate")
    collect = sub.add_parser("collect", help="Collect sealed natural runs")
    collect.add_argument("--runs", type=int, default=20)
    sub.add_parser("fixtures", help="Collect isolated screen coverage fixtures")
    sub.add_parser("export", help="Derive Parquet and manifests from sealed JSONL")
    validate = sub.add_parser("validate", help="Validate raw, audit and Parquet artifacts")
    validate.add_argument("--acceptance", action="store_true")
    pipeline = sub.add_parser("pipeline", help="Run smoke, collect, fixtures, export and acceptance validation")
    pipeline.add_argument("--runs", type=int, default=20)
    archive = sub.add_parser("archive-game", help="Create a private, versioned full-game archive")
    archive.add_argument("--destination", default=str(ARCHIVE_ROOT))
    verify = sub.add_parser("verify-archive", help="Verify every archived game file and version fingerprint")
    verify.add_argument("archive_path")
    replay = sub.add_parser("test-archive-replay", help="Run smoke and Dataset V1 replays against an archive")
    replay.add_argument("archive_path")
    human_import = sub.add_parser("import-human", help="Verify and import sealed HumanRecorder JSONL")
    human_import.add_argument("source")
    human_import.add_argument("--include-partial", action="store_true",
                              help="deprecated compatibility flag; partial decisions are isolated by default")
    human_import.add_argument("--reject-partial", action="store_true",
                              help="fail the whole import if any partial decision is present")
    human_recover = sub.add_parser("recover-human", help="Create a sealed copy of a crash-left partial recording")
    human_recover.add_argument("recording")
    human_recover.add_argument("--destination")
    human_audit = sub.add_parser("audit-human", help="Audit raw HumanRecorder JSONL without importing it")
    human_audit.add_argument("source")
    sub.add_parser("validate-human", help="Validate imported human raw data and Parquet")
    combat_build = sub.add_parser(
        "build-combat-dataset",
        help="Build or extend run-held-out test plus combat-level train/validation data",
    )
    combat_build.add_argument("--rebuild", action="store_true",
                              help="recompute the derived combat dataset and all split assignments")
    sub.add_parser("validate-combat-dataset", help="Validate combat grouping and split isolation")
    combat_examples = sub.add_parser(
        "build-combat-examples", help="Build or incrementally extend Combat Observation/Action V0 samples"
    )
    combat_examples.add_argument("--rebuild", action="store_true",
                                 help="recompute every model-facing Combat V0 sample")
    sub.add_parser("validate-combat-examples", help="Validate Combat V0 observations, candidates and labels")
    combat_vocab = sub.add_parser("build-combat-vocab", help="Build or extend the train-only Combat V0 vocabulary")
    combat_vocab.add_argument("--rebuild", action="store_true", help="reassign all vocabulary indices")
    sub.add_parser("build-combat-value-targets", help="Derive per-decision combat resource targets")
    sub.add_parser("validate-combat-value-targets", help="Validate derived combat resource targets")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_json(__import__("pathlib").Path(args.config))
    try:
        if args.command == "setup-engine":
            setup_engine(config)
            _print({"status": "PASS", "engine": "built"})
        elif args.command == "lock-environment":
            _print(create_environment_lock(__import__("pathlib").Path(args.config)))
        elif args.command == "smoke":
            require_lock(config)
            _print(run_smoke(config))
        elif args.command == "collect":
            require_lock(config)
            _print(Collector(config).collect_many(args.runs))
        elif args.command == "fixtures":
            require_lock(config)
            _print(collect_fixtures(config))
        elif args.command == "export":
            require_lock(config)
            _print(export_dataset(config))
        elif args.command == "validate":
            require_lock(config)
            _print(validate_dataset(config, acceptance=args.acceptance))
        elif args.command == "pipeline":
            require_lock(config)
            result = {
                "smoke": run_smoke(config),
                "runs": Collector(config).collect_many(args.runs),
                "fixtures": collect_fixtures(config),
                "manifest": export_dataset(config),
            }
            result["validation"] = validate_dataset(config, acceptance=True)
            _print(result)
        elif args.command == "archive-game":
            _print(archive_game(config, __import__("pathlib").Path(args.destination)))
        elif args.command == "verify-archive":
            _print(verify_archive(config, __import__("pathlib").Path(args.archive_path)))
        elif args.command == "test-archive-replay":
            _print(test_archive_replay(config, __import__("pathlib").Path(args.archive_path)))
        elif args.command == "import-human":
            _print(import_human_recordings(
                __import__("pathlib").Path(args.source),
                include_partial=args.include_partial or not args.reject_partial,
            ))
        elif args.command == "recover-human":
            destination = __import__("pathlib").Path(args.destination) if args.destination else None
            _print(recover_recording(__import__("pathlib").Path(args.recording), destination))
        elif args.command == "audit-human":
            result = audit_human_recording(__import__("pathlib").Path(args.source))
            _print(result)
            if result["status"] != "PASS":
                return 1
        elif args.command == "validate-human":
            _print(validate_human_dataset())
        elif args.command == "build-combat-dataset":
            _print(build_combat_dataset(rebuild=args.rebuild))
        elif args.command == "validate-combat-dataset":
            _print(validate_combat_dataset())
        elif args.command == "build-combat-examples":
            _print(build_combat_model_examples(rebuild=args.rebuild))
        elif args.command == "validate-combat-examples":
            _print(validate_combat_model_examples())
        elif args.command == "build-combat-vocab":
            _print(build_combat_vocabulary(rebuild=args.rebuild))
        elif args.command == "build-combat-value-targets":
            _print(build_combat_value_targets())
        elif args.command == "validate-combat-value-targets":
            _print(validate_combat_value_targets())
        return 0
    except (ValidationFailure, Exception) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
