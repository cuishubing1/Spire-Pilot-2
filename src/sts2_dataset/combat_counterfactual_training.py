from __future__ import annotations

import hashlib
import math
from typing import Any


def split_examples_by_scenario(
    examples: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    scenario_ids = sorted({str(row["scenario_id"]) for row in examples})
    if len(scenario_ids) < 2:
        raise ValueError("counterfactual split requires at least two scenarios")
    ranked = sorted(
        scenario_ids,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    validation_count = max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
    validation_ids = set(ranked[:validation_count])
    train = [row for row in examples if str(row["scenario_id"]) not in validation_ids]
    validation = [row for row in examples if str(row["scenario_id"]) in validation_ids]
    return train, validation


def pairwise_ranking_loss(
    logits: Any,
    examples: list[dict[str, Any]],
    *,
    utility_weight_cap: float = 2.0,
) -> Any:
    import torch
    from torch.nn import functional as F

    losses = []
    weights = []
    for batch_index, example in enumerate(examples):
        for pair in example["pairwise_labels"]:
            left = int(pair["left_candidate_index"])
            right = int(pair["right_candidate_index"])
            winner = int(pair["winner_candidate_index"])
            loser = right if winner == left else left
            losses.append(F.softplus(-(logits[batch_index, winner] - logits[batch_index, loser])))
            delta = abs(float(pair["mean_utility_delta_left_minus_right"]))
            weights.append(1.0 + min(delta, utility_weight_cap))
    if not losses:
        raise ValueError("counterfactual batch contains no non-tied pairwise labels")
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()


def pairwise_ranking_metrics(
    logits: Any,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_count = 0
    correct = 0
    losses = []
    root_correct = 0
    suboptimal_roots = 0
    suboptimal_correct = 0
    for batch_index, example in enumerate(examples):
        for pair in example["pairwise_labels"]:
            left = int(pair["left_candidate_index"])
            right = int(pair["right_candidate_index"])
            winner = int(pair["winner_candidate_index"])
            loser = right if winner == left else left
            margin = float((logits[batch_index, winner] - logits[batch_index, loser]).item())
            correct += int(margin > 0.0)
            pair_count += 1
            losses.append(math.log1p(math.exp(-max(-50.0, min(50.0, margin)))))
        evaluated = [int(row["candidate_index"]) for row in example["actions"]]
        predicted = max(evaluated, key=lambda index: float(logits[batch_index, index].item()))
        best_id = str(example["best_candidate_id"])
        best_index = next(
            int(row["candidate_index"])
            for row in example["actions"]
            if str(row["candidate_id"]) == best_id
        )
        matched = predicted == best_index
        root_correct += int(matched)
        if example["policy_suboptimal"]:
            suboptimal_roots += 1
            suboptimal_correct += int(matched)
    return {
        "pairs": pair_count,
        "pairwise_accuracy": correct / pair_count if pair_count else None,
        "pairwise_loss": sum(losses) / len(losses) if losses else None,
        "roots": len(examples),
        "teacher_top1_accuracy": root_correct / len(examples) if examples else None,
        "policy_suboptimal_roots": suboptimal_roots,
        "policy_suboptimal_top1_accuracy": (
            suboptimal_correct / suboptimal_roots if suboptimal_roots else None
        ),
    }


def teacher_best_action_loss(
    logits: Any,
    examples: list[dict[str, Any]],
    *,
    policy_suboptimal_weight: float = 2.0,
) -> Any:
    import torch
    from torch.nn import functional as F

    losses = []
    weights = []
    for batch_index, example in enumerate(examples):
        evaluated = [int(row["candidate_index"]) for row in example["actions"]]
        best_id = str(example["best_candidate_id"])
        best_index = next(
            int(row["candidate_index"])
            for row in example["actions"]
            if str(row["candidate_id"]) == best_id
        )
        local_target = evaluated.index(best_index)
        local_logits = logits[batch_index, evaluated].unsqueeze(0)
        losses.append(
            F.cross_entropy(
                local_logits,
                torch.as_tensor([local_target], device=logits.device),
            )
        )
        weights.append(policy_suboptimal_weight if example["policy_suboptimal"] else 1.0)
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()
