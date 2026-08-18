"""Fine-tune only P2's final candidate scorer with terminal pairwise labels."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_combat_policy_online import _load_policy  # noqa: E402
from sts2_dataset.combat_counterfactual_training import (  # noqa: E402
    pairwise_ranking_loss,
    pairwise_ranking_metrics,
    split_examples_by_scenario,
    teacher_best_action_loss,
)
from sts2_dataset.combat_model import numpy_batch_to_torch  # noqa: E402
from sts2_dataset.combat_tensorizer import collate_combat_numpy  # noqa: E402
from sts2_dataset.combat_training import (  # noqa: E402
    _load_split,
    _make_loader,
    evaluate_combat_policy,
)
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config" / "combat_counterfactual_ranker_v0.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_counterfactual_ranker_v0"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _counterfactual_batch(tensorizer: Any, rows: list[dict[str, Any]], device: str):
    numpy_batch = collate_combat_numpy(
        [tensorizer.tensorize(row["sample"]) for row in rows]
    )
    return numpy_batch_to_torch(numpy_batch, device=device)


def _pair_metrics(model: Any, tensorizer: Any, rows: list[dict[str, Any]], device: str):
    model.eval()
    with torch.no_grad():
        logits = model(_counterfactual_batch(tensorizer, rows, device))
    return pairwise_ranking_metrics(logits, rows)


def _distillation_loss(current: Any, reference: Any, mask: Any) -> Any:
    reference_probability = reference.softmax(dim=-1)
    current_log_probability = current.log_softmax(dim=-1).masked_fill(~mask, 0.0)
    return -(reference_probability * current_log_probability).sum(dim=-1).mean()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual ranker training requires CUDA")

    dataset_path = _resolve(config["counterfactual_dataset"])
    checkpoint_path = _resolve(config["initialize_checkpoint"])
    examples = _load_jsonl(dataset_path)
    train_examples, holdout_examples = split_examples_by_scenario(
        examples, validation_fraction=float(config["validation_fraction"])
    )
    model, tensorizer, device = _load_policy(checkpoint_path, "cuda")
    if device != "cuda":
        raise RuntimeError(f"counterfactual ranker resolved unexpected device: {device}")
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_mode = str(config.get("trainable_mode") or "final_scorer")
    if trainable_mode == "final_scorer":
        trainable_module = model.candidate_scorer[-1]
        if not isinstance(trainable_module, torch.nn.Linear):
            raise RuntimeError("P2 candidate scorer does not end in a Linear layer")
        trainable_parameters = list(trainable_module.parameters())
    elif trainable_mode == "encounter_embedding":
        if model.encounter_embedding is None or model.encounter_candidate_projection is None:
            raise RuntimeError("checkpoint does not expose an encounter residual adapter")
        trainable_module = model.encounter_embedding
        trainable_parameters = list(trainable_module.parameters())
    else:
        raise RuntimeError(f"unsupported counterfactual trainable mode: {trainable_mode}")
    for parameter in trainable_parameters:
        parameter.requires_grad = True
    initial_parameters = [parameter.detach().clone() for parameter in trainable_parameters]

    anchor_rows = _load_split(
        tensorizer,
        "train",
        limit=int(config["human_anchor_samples"]),
        seed=seed,
    )
    anchor_loader = _make_loader(
        anchor_rows,
        batch_size=int(config["human_anchor_batch_size"]),
        shuffle=True,
        seed=seed,
    )
    human_validation_rows = _load_split(
        tensorizer,
        "validation",
        limit=None,
        seed=seed + 1,
    )
    human_validation_loader = _make_loader(
        human_validation_rows,
        batch_size=int(config["human_anchor_batch_size"]),
        shuffle=False,
        seed=seed + 1,
    )
    baseline_pair_train = _pair_metrics(model, tensorizer, train_examples, device)
    baseline_pair_holdout = _pair_metrics(model, tensorizer, holdout_examples, device)
    baseline_human = evaluate_combat_policy(
        reference, human_validation_loader, device=device
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    counterfactual_batch = _counterfactual_batch(tensorizer, train_examples, device)
    best_holdout_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        counterfactual_logits = model(counterfactual_batch)
        pair_loss = pairwise_ranking_loss(counterfactual_logits, train_examples)
        best_action_loss = teacher_best_action_loss(
            counterfactual_logits,
            train_examples,
            policy_suboptimal_weight=float(config["policy_suboptimal_weight"]),
        )
        total_loss = (
            float(config["pairwise_coefficient"]) * pair_loss
            + float(config["teacher_best_action_coefficient"]) * best_action_loss
        )
        anchor_ce_sum = torch.zeros((), device=device)
        anchor_distill_sum = torch.zeros((), device=device)
        anchor_batches = 0
        for numpy_batch in anchor_loader:
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            current_logits = model(batch)
            with torch.no_grad():
                reference_logits = reference(batch)
            anchor_ce_sum = anchor_ce_sum + F.cross_entropy(
                current_logits, batch["label"].long()
            )
            anchor_distill_sum = anchor_distill_sum + _distillation_loss(
                current_logits, reference_logits, batch["action_mask"].bool()
            )
            anchor_batches += 1
        anchor_ce = anchor_ce_sum / anchor_batches
        anchor_distill = anchor_distill_sum / anchor_batches
        delta_loss = sum(
            (parameter - initial).square().mean()
            for parameter, initial in zip(trainable_parameters, initial_parameters)
        )
        total_loss = (
            total_loss
            + float(config["behavior_cloning_coefficient"]) * anchor_ce
            + float(config["distillation_coefficient"]) * anchor_distill
            + float(config["parameter_delta_coefficient"]) * delta_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()

        train_metrics = _pair_metrics(model, tensorizer, train_examples, device)
        holdout_metrics = _pair_metrics(model, tensorizer, holdout_examples, device)
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total_loss.item()),
                "pairwise_loss": float(pair_loss.item()),
                "teacher_best_action_loss": float(best_action_loss.item()),
                "anchor_ce": float(anchor_ce.item()),
                "anchor_distillation": float(anchor_distill.item()),
                "parameter_delta": float(delta_loss.item()),
                "train": train_metrics,
                "holdout": holdout_metrics,
            }
        )
        holdout_loss = float(holdout_metrics["pairwise_loss"])
        if holdout_loss < best_holdout_loss - 1e-6:
            best_holdout_loss = holdout_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("counterfactual ranker did not produce a candidate state")
    model.load_state_dict(best_state)
    tuned_pair_train = _pair_metrics(model, tensorizer, train_examples, device)
    tuned_pair_holdout = _pair_metrics(model, tensorizer, holdout_examples, device)
    tuned_human = evaluate_combat_policy(model, human_validation_loader, device=device)
    human_accuracy_drop = float(baseline_human["top1_accuracy"]) - float(
        tuned_human["top1_accuracy"]
    )
    gate = {
        "holdout_pairwise_loss_improved": (
            float(tuned_pair_holdout["pairwise_loss"])
            < float(baseline_pair_holdout["pairwise_loss"])
        ),
        "human_validation_accuracy_drop": human_accuracy_drop,
        "human_validation_accuracy_within_limit": human_accuracy_drop
        <= float(config["max_human_validation_accuracy_drop"]),
        "holdout_discrete_ranking_improved": bool(
            float(tuned_pair_holdout["pairwise_accuracy"])
            > float(baseline_pair_holdout["pairwise_accuracy"])
            or float(tuned_pair_holdout["policy_suboptimal_top1_accuracy"] or 0.0)
            > float(baseline_pair_holdout["policy_suboptimal_top1_accuracy"] or 0.0)
        ),
    }
    gate["accepted"] = bool(
        gate["holdout_pairwise_loss_improved"]
        and gate["human_validation_accuracy_within_limit"]
        and gate["holdout_discrete_ranking_improved"]
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = args.output.resolve() / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = {
        **original,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "counterfactual_tuning": {
            "dataset_sha256": sha256_file(dataset_path),
            "config_sha256": sha256_file(config_path),
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
            "trainable_mode": trainable_mode,
            "accepted": gate["accepted"],
        },
    }
    checkpoint_output = artifact_dir / "model.pt"
    torch.save(checkpoint, checkpoint_output)
    vocabulary_source = checkpoint_path.parent / "vocab.json"
    vocabulary_output = artifact_dir / "vocab.json"
    shutil.copy2(vocabulary_source, vocabulary_output)
    dataset_index_source = checkpoint_path.parent / "dataset_index.json"
    dataset_index_output = artifact_dir / "dataset_index.json"
    shutil.copy2(dataset_index_source, dataset_index_output)
    report = {
        "schema_version": "combat-counterfactual-ranker-training-0.1.0",
        "generated_at": utc_now(),
        "device": device,
        "status": "pass" if gate["accepted"] else "rejected",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "initial_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "candidate_checkpoint": {
            "path": str(checkpoint_output),
            "sha256": sha256_file(checkpoint_output),
            "vocabulary_path": str(vocabulary_output),
            "vocabulary_sha256": sha256_file(vocabulary_output),
            "dataset_index_path": str(dataset_index_output),
            "dataset_index_sha256": sha256_file(dataset_index_output),
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "trainable_mode": trainable_mode,
        "split": {
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
            "train_scenarios": len({row["scenario_id"] for row in train_examples}),
            "holdout_scenarios": len({row["scenario_id"] for row in holdout_examples}),
        },
        "baseline": {
            "counterfactual_train": baseline_pair_train,
            "counterfactual_holdout": baseline_pair_holdout,
            "human_validation": baseline_human,
        },
        "tuned": {
            "counterfactual_train": tuned_pair_train,
            "counterfactual_holdout": tuned_pair_holdout,
            "human_validation": tuned_human,
        },
        "gate": gate,
        "epochs_completed": len(history),
        "history": history,
    }
    write_json_atomic(artifact_dir / "report.json", report)
    write_json_atomic(args.output.resolve() / "latest_candidate.json", {
        "report": str(artifact_dir / "report.json"),
        "checkpoint": str(checkpoint_output),
        "accepted": gate["accepted"],
    })
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    summary = {
        "status": report["status"],
        "trainable_parameters": report["trainable_parameters"],
        "split": report["split"],
        "baseline_pair_holdout": report["baseline"]["counterfactual_holdout"],
        "tuned_pair_holdout": report["tuned"]["counterfactual_holdout"],
        "baseline_human_top1": report["baseline"]["human_validation"]["top1_accuracy"],
        "tuned_human_top1": report["tuned"]["human_validation"]["top1_accuracy"],
        "gate": report["gate"],
        "candidate_checkpoint": report["candidate_checkpoint"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
