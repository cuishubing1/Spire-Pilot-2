"""Evaluate a trigger-gated counterfactual ranker on untouched human validation states."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_combat_policy_online import _load_policy  # noqa: E402
from sts2_dataset.combat_contract import iter_combat_model_samples  # noqa: E402
from sts2_dataset.combat_counterfactual import (  # noqa: E402
    counterfactual_gate_variants,
    on_policy_trigger_reasons,
)
from sts2_dataset.combat_failure import _load_training_reference  # noqa: E402
from sts2_dataset.combat_model import numpy_batch_to_torch  # noqa: E402
from sts2_dataset.combat_online import visible_intent_end_turn_hp_loss  # noqa: E402
from sts2_dataset.combat_search import normalized_policy_entropy  # noqa: E402
from sts2_dataset.combat_training import _make_loader  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_BASE = REPO_ROOT / "artifacts" / "combat_policy_p2_cuda" / "20260818T125348Z" / "model.pt"
DEFAULT_LATEST = REPO_ROOT / "artifacts" / "combat_counterfactual_ranker_v0" / "latest_candidate.json"
DEFAULT_COLLECTION = REPO_ROOT / "artifacts" / "combat_on_policy_counterfactual_train30_v0.json"
DEFAULT_SAMPLES = REPO_ROOT / "data" / "human" / "combat_v1" / "model_v0" / "samples.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_counterfactual_gate_v0.json"


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual gate evaluation requires CUDA")
    latest = load_json(args.latest.resolve())
    tuned_path = Path(latest["checkpoint"]).resolve()
    base_path = args.base.resolve()
    base, tensorizer, device = _load_policy(base_path, "cuda")
    tuned, _, tuned_device = _load_policy(tuned_path, "cuda")
    if device != "cuda" or tuned_device != "cuda":
        raise RuntimeError("counterfactual gate did not resolve CUDA")
    base.eval()
    tuned.eval()

    collection = load_json(args.collection.resolve())
    trigger = collection["collection_config"]
    training = _load_training_reference(args.samples.resolve())
    decoded = list(iter_combat_model_samples("validation"))
    decoded_by_id = {str(row["transition_id"]): row for row in decoded}
    tensorized = [tensorizer.tensorize(row) for row in decoded]
    loader = _make_loader(
        tensorized,
        batch_size=args.batch_size,
        shuffle=False,
        seed=20260818,
    )

    total = 0
    base_correct = 0
    tuned_correct = 0
    variant_stats: dict[str, list[int]] = {
        name: [0, 0, 0]
        for name in ("any_trigger", "two_signals", "risk_and_uncertainty", "strict")
    }
    trigger_reasons: Counter[str] = Counter()
    grouped: dict[str, dict[str, list[int]]] = {
        "act": defaultdict(lambda: [0, 0, 0, 0]),
        "room_type": defaultdict(lambda: [0, 0, 0, 0]),
    }
    with torch.no_grad():
        for numpy_batch in loader:
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            base_logits = base(batch)
            tuned_logits = tuned(batch)
            base_predictions = base_logits.argmax(dim=1)
            tuned_predictions = tuned_logits.argmax(dim=1)
            labels = batch["label"].long()
            for index, transition_id in enumerate(numpy_batch["transition_id"]):
                sample = decoded_by_id[str(transition_id)]
                valid = batch["action_mask"][index].bool()
                probabilities = base_logits[index, valid].softmax(dim=0).tolist()
                entropy = normalized_policy_entropy(probabilities)
                ordered = sorted(probabilities, reverse=True)
                margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
                base_index = int(base_predictions[index].item())
                candidate = sample["candidates"][base_index]
                action_count = int(
                    training["action_label_count"][
                        (str(candidate.get("action_type") or ""), candidate.get("source_id"))
                    ]
                )
                observation = sample["observation"]
                global_state = observation.get("global") or {}
                hp_ratio = float(global_state.get("hp") or 0.0) / max(
                    float(global_state.get("max_hp") or 1.0), 1.0
                )
                exact_round = training["encounter_profile_quantiles"].get(
                    str(sample["encounter_signature"]), {}
                ).get("round")
                visible = visible_intent_end_turn_hp_loss(observation)
                incoming = float((visible or {}).get("hp_loss") or 0.0)
                reasons = on_policy_trigger_reasons(
                    hp_ratio=hp_ratio,
                    round_number=int(global_state.get("round") or 1),
                    exact_encounter_round_p95=exact_round[3] if exact_round else None,
                    incoming_hp_loss=incoming,
                    policy_entropy=entropy,
                    policy_margin=margin,
                    chosen_action_train_count=action_count,
                    low_hp_threshold=float(trigger["low_hp_threshold"]),
                    incoming_hp_loss_threshold=float(trigger["incoming_hp_loss_threshold"]),
                    high_entropy_threshold=float(trigger["high_entropy_threshold"]),
                    low_margin_threshold=float(trigger["low_margin_threshold"]),
                    rare_action_threshold=int(trigger["rare_action_threshold"]),
                )
                label = int(labels[index].item())
                base_match = base_index == label
                tuned_match = int(tuned_predictions[index].item()) == label
                total += 1
                base_correct += int(base_match)
                tuned_correct += int(tuned_match)
                variants = counterfactual_gate_variants(reasons)
                for name, use_tuned in variants.items():
                    prediction = (
                        tuned_predictions[index] if use_tuned else base_predictions[index]
                    )
                    values = variant_stats[name]
                    values[0] += int(use_tuned)
                    values[1] += int(prediction.item() == label)
                    values[2] += int(
                        use_tuned and prediction.item() != base_predictions[index].item()
                    )
                trigger_reasons.update(reasons)
                room_type = str(global_state.get("room_type") or "unknown")
                strict_tuned = variants["strict"]
                strict_prediction = (
                    tuned_predictions[index] if strict_tuned else base_predictions[index]
                )
                for group, key in (("act", str(sample["act"])), ("room_type", room_type)):
                    values = grouped[group][key]
                    values[0] += 1
                    values[1] += int(base_match)
                    values[2] += int(strict_prediction.item() == label)
                    values[3] += int(strict_tuned)

    group_report = {
        group: {
            key: {
                "samples": values[0],
                "base_accuracy": _accuracy(values[1], values[0]),
                "gated_accuracy": _accuracy(values[2], values[0]),
                "trigger_rate": _accuracy(values[3], values[0]),
            }
            for key, values in sorted(rows.items())
        }
        for group, rows in grouped.items()
    }
    base_accuracy = _accuracy(base_correct, total)
    variant_report = {
        name: {
            "triggered_samples": values[0],
            "trigger_rate": _accuracy(values[0], total),
            "accuracy": _accuracy(values[1], total),
            "accuracy_delta": _accuracy(values[1], total) - base_accuracy,
            "changed_actions": values[2],
            "changed_action_rate": _accuracy(values[2], total),
        }
        for name, values in variant_stats.items()
    }
    strict = variant_report["strict"]
    report = {
        "schema_version": "combat-counterfactual-gate-evaluation-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "device": device,
        "dataset_split": "validation",
        "base_checkpoint": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "tuned_checkpoint": {"path": str(tuned_path), "sha256": sha256_file(tuned_path)},
        "collection_report": {
            "path": str(args.collection.resolve()),
            "sha256": sha256_file(args.collection.resolve()),
        },
        "samples": total,
        "base_accuracy": base_accuracy,
        "global_tuned_accuracy": _accuracy(tuned_correct, total),
        "primary_gate": "strict",
        "gated_accuracy": strict["accuracy"],
        "gated_accuracy_delta": strict["accuracy_delta"],
        "triggered_samples": strict["triggered_samples"],
        "trigger_rate": strict["trigger_rate"],
        "changed_actions": strict["changed_actions"],
        "changed_action_rate": strict["changed_action_rate"],
        "max_accuracy_drop": args.max_accuracy_drop,
        "accepted": strict["accuracy_delta"] >= -args.max_accuracy_drop,
        "gate_variants": variant_report,
        "trigger_reasons": dict(sorted(trigger_reasons.items())),
        "grouped": group_report,
    }
    write_json_atomic(args.output.resolve(), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({
        key: report[key]
        for key in (
            "status",
            "samples",
            "base_accuracy",
            "global_tuned_accuracy",
            "gated_accuracy",
            "gated_accuracy_delta",
            "trigger_rate",
            "changed_action_rate",
            "accepted",
            "trigger_reasons",
            "gate_variants",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
