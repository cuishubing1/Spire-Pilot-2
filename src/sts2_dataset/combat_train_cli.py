from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# On Windows, the locked PyArrow and PyTorch wheels both initialize native
# runtimes. Loading torch first avoids a DLL initialization conflict observed
# when pyarrow has already been imported by the dataset CLI.
try:
    import torch  # noqa: F401
except (ImportError, OSError) as exc:
    print(f"FAIL: PyTorch could not be loaded: {exc}", file=sys.stderr)
    raise SystemExit(1)

from .combat_training import evaluate_combat_checkpoint, train_combat_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sts2-train", description="Train Combat Policy Transformer V0")
    parser.add_argument("--training-config", default="config/combat_policy_v0.json")
    parser.add_argument("--artifact-root", default="artifacts/combat_policy_v0")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--evaluate-checkpoint")
    parser.add_argument("--initialize-checkpoint")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.evaluate_checkpoint:
            result = evaluate_combat_checkpoint(
                Path(args.evaluate_checkpoint),
                split=args.split,
                device=args.device,
                max_samples=args.max_eval_samples,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        result = train_combat_policy(
            config_path=Path(args.training_config),
            artifact_root=Path(args.artifact_root),
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            initialize_checkpoint=Path(args.initialize_checkpoint) if args.initialize_checkpoint else None,
            freeze_base=args.freeze_base,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
