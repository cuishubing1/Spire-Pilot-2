"""Build a compact, deduplicated train-only dataset from terminal rollout reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sts2_dataset.combat_counterfactual import (  # noqa: E402
    COUNTERFACTUAL_DATASET_VERSION,
    build_counterfactual_training_examples,
)
from sts2_dataset.util import (  # noqa: E402
    canonical_json,
    load_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


DEFAULT_REPORT = REPO_ROOT / "artifacts" / "combat_on_policy_counterfactual_v0.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "human"
    / "combat_v1"
    / "counterfactual_v0"
    / "train_roots.jsonl"
)


def _merge(reports: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_root: dict[str, dict[str, Any]] = {}
    sources = []
    for path in reports:
        resolved = path.resolve()
        report = load_json(resolved)
        examples = build_counterfactual_training_examples(report)
        sources.append({"path": str(resolved), "sha256": sha256_file(resolved)})
        for example in examples:
            key = str(example["root_fingerprint"])
            previous = by_root.get(key)
            if previous is None:
                by_root[key] = example
                continue
            if canonical_json(previous) != canonical_json(example):
                raise ValueError(f"conflicting counterfactual root: {key}")
    rows = sorted(
        by_root.values(),
        key=lambda row: (
            int(row["act"]),
            str(row["scenario_id"]),
            int(row["step"]),
            str(row["root_fingerprint"]),
        ),
    )
    return rows, sources


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    os.replace(str(temporary), str(path))


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [path.resolve() for path in args.reports]
    rows, sources = _merge(reports)
    output = args.output.resolve()
    _write_jsonl_atomic(output, rows)
    manifest = {
        "schema_version": COUNTERFACTUAL_DATASET_VERSION,
        "generated_at": utc_now(),
        "dataset_split": "train",
        "source_reports": sources,
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "examples": len(rows),
        "policy_suboptimal_examples": sum(
            bool(row["policy_suboptimal"]) for row in rows
        ),
        "pairwise_labels": sum(len(row["pairwise_labels"]) for row in rows),
        "acts": {
            str(act): sum(int(row["act"]) == act for row in rows)
            for act in (1, 2, 3)
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    write_json_atomic(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", default=[DEFAULT_REPORT])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
