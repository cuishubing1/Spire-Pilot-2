"""Build a reproducible failure-trajectory ratchet from the full validation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sts2_dataset.combat_failure import build_failure_ratchet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=REPO_ROOT / "artifacts" / "validation_combat_ablation_p2_full.json",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=REPO_ROOT / "data" / "human" / "combat_v1" / "model_v0" / "samples.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "combat_failure_ratchet_v0.json",
    )
    parser.add_argument("--high-regret-hp", type=float, default=20.0)
    parser.add_argument("--search-regression-hp", type=float, default=15.0)
    args = parser.parse_args()
    result = build_failure_ratchet(
        args.evaluation,
        args.samples,
        args.output,
        high_regret_threshold=args.high_regret_hp,
        search_regression_threshold=args.search_regression_hp,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "summary": result["summary"],
                "distribution_shift": result["distribution_shift"],
                "search_takeover": result["search_takeover"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
