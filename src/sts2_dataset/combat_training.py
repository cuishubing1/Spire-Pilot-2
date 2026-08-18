from __future__ import annotations

import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .combat_contract import COMBAT_MODEL_ROOT, iter_combat_model_samples, validate_combat_model_examples
from .combat_engine_features import CANDIDATE_ENGINE_FEATURE_DIM
from .combat_tensorizer import VOCAB_PATH, CombatTensorizerV0, collate_combat_numpy
from .combat_value import VALUE_MANIFEST_PATH, load_combat_value_targets
from .constants import ROOT
from .human import HumanRecordingError
from .util import load_json, sha256_file, write_json_atomic

TRAINING_SCHEMA_VERSION = "combat-policy-training-0.1.0"
DEFAULT_TRAINING_CONFIG = ROOT / "config" / "combat_policy_v0.json"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "combat_policy_v0"


def _require_torch():
    try:
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise HumanRecordingError(
            'PyTorch is required; install the optional dependency with: pip install -e ".[train]"'
        ) from exc
    return torch, F, DataLoader, Dataset


def _select_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise HumanRecordingError(f"requested device is unavailable: {requested}")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _limit_by_combat(samples: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Create a deterministic smoke subset without taking only early combats/Acts."""
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    by_combat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_combat[sample["combat_id"]].append(sample)
    generator = random.Random(seed)
    combats = list(by_combat)
    generator.shuffle(combats)
    for rows in by_combat.values():
        generator.shuffle(rows)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for combat_id in combats:
            rows = by_combat[combat_id]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _load_split(
    tensorizer: CombatTensorizerV0,
    split: str,
    *,
    limit: int | None,
    seed: int,
    required_sources: dict[str, str] | None = None,
    value_targets: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    decoded = list(iter_combat_model_samples(split))
    if required_sources is not None:
        decoded_by_id = {sample["transition_id"]: sample for sample in decoded}
        missing = sorted(set(required_sources) - set(decoded_by_id))
        if missing:
            raise HumanRecordingError(f"checkpoint evaluation source is missing {len(missing)} transitions")
        decoded = [decoded_by_id[transition_id] for transition_id in required_sources]
        changed = [
            transition_id for transition_id, expected_sha256 in required_sources.items()
            if decoded_by_id[transition_id].get("source_transition_sha256") != expected_sha256
        ]
        if changed:
            raise HumanRecordingError(f"checkpoint evaluation source changed for {len(changed)} transitions")
    decoded = _limit_by_combat(decoded, limit, seed)
    if value_targets is not None:
        missing_targets = [row["transition_id"] for row in decoded if row["transition_id"] not in value_targets]
        if missing_targets:
            raise HumanRecordingError(f"missing {len(missing_targets)} combat resource targets")
        decoded = [{**row, "value_target": value_targets[row["transition_id"]]} for row in decoded]
    return [tensorizer.tensorize(sample) for sample in decoded]


def _make_loader(
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    torch, _, DataLoader, Dataset = _require_torch()

    class Rows(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return samples[index]

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Rows(),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate_combat_numpy,
    )


def _accuracy_block(correct: int, total: int) -> dict[str, Any]:
    return {"correct": correct, "samples": total, "accuracy": correct / total if total else None}


def _binary_ece(probabilities: list[float], targets: list[float], *, bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            row for row, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probabilities[row] for row in members) / len(members)
        frequency = sum(targets[row] for row in members) / len(members)
        error += len(members) / total * abs(confidence - frequency)
    return error


def evaluate_combat_policy(model: Any, loader: Any, *, device: str) -> dict[str, Any]:
    torch, F, _, _ = _require_torch()
    from .combat_model import numpy_batch_to_torch

    model.eval()
    total_nll = 0.0
    total = 0
    correct = 0
    top3_correct = 0
    legal_predictions = 0
    random_legal_expected = 0.0
    policy_only_correct = 0
    resource_samples = 0
    hp_loss_absolute_error = 0.0
    immediate_hp_loss_absolute_error = 0.0
    positive_immediate_samples = 0
    positive_immediate_absolute_error = 0.0
    end_turn_immediate_samples = 0
    end_turn_immediate_absolute_error = 0.0
    death_brier = 0.0
    potion_absolute_error = 0.0
    max_hp_absolute_error = 0.0
    state_value_samples = 0
    state_hp_loss_absolute_error = 0.0
    state_death_brier = 0.0
    state_potion_absolute_error = 0.0
    state_max_hp_absolute_error = 0.0
    outcome_samples = 0
    outcome_nll = 0.0
    outcome_raw_nll = 0.0
    outcome_correct = 0
    outcome_within_one = 0
    outcome_end_hp_absolute_error = 0.0
    outcome_raw_end_hp_absolute_error = 0.0
    outcome_death_brier = 0.0
    outcome_death_samples = 0
    outcome_death_probabilities: list[float] = []
    outcome_death_targets: list[float] = []
    grouped: dict[str, dict[Any, list[int]]] = {
        "act": defaultdict(lambda: [0, 0]),
        "combat_difficulty_tier": defaultdict(lambda: [0, 0]),
        "act_combat_difficulty_tier": defaultdict(lambda: [0, 0]),
        "room_type": defaultdict(lambda: [0, 0]),
        "label_action_type": defaultdict(lambda: [0, 0]),
        "encounter_support": defaultdict(lambda: [0, 0]),
        "combat": defaultdict(lambda: [0, 0]),
    }
    with torch.no_grad():
        for numpy_batch in loader:
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            if model.resource_value_head is not None:
                policy_logits, resource, logits = model.policy_resource_outputs(batch)
            else:
                policy_logits = model(batch)
                resource = None
                logits = policy_logits
            labels = batch["label"].long()
            losses = F.cross_entropy(logits, labels, reduction="none")
            predictions = logits.argmax(dim=1)
            matches = predictions.eq(labels)
            label_scores = logits.gather(1, labels[:, None])
            ranks = logits.gt(label_scores).sum(dim=1) + 1
            action_counts = batch["action_mask"].sum(dim=1)

            batch_total = labels.shape[0]
            total += batch_total
            total_nll += float(losses.sum().item())
            correct += int(matches.sum().item())
            top3_correct += int(ranks.le(3).sum().item())
            legal_predictions += int(batch["action_mask"].gather(1, predictions[:, None]).sum().item())
            random_legal_expected += float((1.0 / action_counts.float()).sum().item())
            policy_only_correct += int(policy_logits.argmax(dim=1).eq(labels).sum().item())

            if resource is not None:
                selected = {
                    key: value.gather(1, labels[:, None]).squeeze(1)
                    for key, value in resource.items()
                }
                mask = batch["resource_target_mask"].bool()
                resource_samples += int(mask.sum().item())
                hp_loss_absolute_error += float(
                    (selected["hp_loss_fraction"][mask] - batch["target_hp_loss_fraction"][mask]).abs().sum().item()
                )
                immediate_hp_loss_absolute_error += float(
                    (
                        selected["immediate_hp_loss_fraction"][mask]
                        - batch["target_immediate_hp_loss_fraction"][mask]
                    ).abs().sum().item()
                )
                immediate_target = batch["target_immediate_hp_loss_fraction"][mask]
                immediate_prediction = selected["immediate_hp_loss_fraction"][mask]
                positive_mask = immediate_target.gt(0)
                positive_immediate_samples += int(positive_mask.sum().item())
                positive_immediate_absolute_error += float(
                    (immediate_prediction[positive_mask] - immediate_target[positive_mask]).abs().sum().item()
                )
                end_turn_mask = torch.as_tensor(
                    [name == "end_turn" for name in numpy_batch["label_action_type"]],
                    device=mask.device,
                    dtype=torch.bool,
                )[mask]
                end_turn_immediate_samples += int(end_turn_mask.sum().item())
                end_turn_immediate_absolute_error += float(
                    (immediate_prediction[end_turn_mask] - immediate_target[end_turn_mask]).abs().sum().item()
                )
                death_probability = torch.sigmoid(selected["death_logit"][mask])
                death_brier += float(
                    ((death_probability - batch["target_death"][mask]) ** 2).sum().item()
                )
                potion_absolute_error += float(
                    (selected["potion_spent"][mask] - batch["target_potion_spent"][mask]).abs().sum().item()
                )
                max_hp_absolute_error += float(
                    (selected["max_hp_delta"][mask] - batch["target_max_hp_delta"][mask]).abs().sum().item()
                )
            if model.state_value_head is not None:
                state_predictions = model.state_value_predictions(batch)
                state_mask = batch["resource_target_mask"].bool()
                state_value_samples += int(state_mask.sum().item())
                state_hp_loss_absolute_error += float(
                    (
                        state_predictions["hp_loss_fraction"][state_mask]
                        - batch["target_hp_loss_fraction"][state_mask]
                    ).abs().sum().item()
                )
                state_death_probability = torch.sigmoid(
                    state_predictions["death_logit"][state_mask]
                )
                state_death_brier += float(
                    (
                        state_death_probability
                        - batch["target_death"][state_mask]
                    ).square().sum().item()
                )
                state_potion_absolute_error += float(
                    (
                        state_predictions["potion_spent"][state_mask]
                        - batch["target_potion_spent"][state_mask]
                    ).abs().sum().item()
                )
                state_max_hp_absolute_error += float(
                    (
                        state_predictions["max_hp_delta"][state_mask]
                        - batch["target_max_hp_delta"][state_mask]
                    ).abs().sum().item()
                )
            if model.state_outcome_head is not None:
                outcome_mask = batch["resource_target_mask"].bool()
                outcome_targets = model.state_outcome_targets(batch)[outcome_mask]
                calibrated = model.state_outcome_predictions(batch)
                calibrated_logits = calibrated["logits"][outcome_mask]
                raw_logits = model.state_outcome_logits(batch, calibrated=False)[outcome_mask]
                outcome_samples += int(outcome_mask.sum().item())
                outcome_nll += float(
                    F.cross_entropy(calibrated_logits, outcome_targets, reduction="sum").item()
                )
                outcome_raw_nll += float(
                    F.cross_entropy(raw_logits, outcome_targets, reduction="sum").item()
                )
                outcome_predictions = calibrated_logits.argmax(dim=1)
                outcome_correct += int(outcome_predictions.eq(outcome_targets).sum().item())
                outcome_within_one += int(
                    outcome_predictions.sub(outcome_targets).abs().le(1).sum().item()
                )
                expected_hp = calibrated["expected_end_hp_fraction"][outcome_mask]
                terminal_hp = batch["target_terminal_hp_fraction"][outcome_mask]
                outcome_end_hp_absolute_error += float(
                    expected_hp.sub(terminal_hp).abs().sum().item()
                )
                survivor_bins = model.config.state_outcome_bins - 1
                centers = (
                    torch.arange(
                        survivor_bins, device=raw_logits.device, dtype=raw_logits.dtype
                    )
                    + 0.5
                ) / survivor_bins
                raw_expected_hp = (raw_logits.softmax(dim=-1)[:, 1:] * centers).sum(dim=-1)
                outcome_raw_end_hp_absolute_error += float(
                    raw_expected_hp.sub(terminal_hp).abs().sum().item()
                )
                death_probability = calibrated["death_probability"][outcome_mask]
                death_target = batch["target_death"][outcome_mask]
                outcome_death_samples += int(death_target.sum().item())
                outcome_death_brier += float(
                    death_probability.sub(death_target).square().sum().item()
                )
                outcome_death_probabilities.extend(death_probability.cpu().tolist())
                outcome_death_targets.extend(death_target.cpu().tolist())

            match_values = matches.cpu().tolist()
            act_values = batch["act"].cpu().tolist()
            encounter_values = batch.get("encounter_identity")
            encounter_values = (
                encounter_values.cpu().tolist()
                if encounter_values is not None
                else [1] * len(match_values)
            )
            for index, matched in enumerate(match_values):
                keys = {
                    "act": str(act_values[index]),
                    "combat_difficulty_tier": numpy_batch["combat_difficulty_tier"][index],
                    "act_combat_difficulty_tier": (
                        f"act{act_values[index]}:"
                        f"{numpy_batch['combat_difficulty_tier'][index]}"
                    ),
                    "room_type": numpy_batch["room_type"][index],
                    "label_action_type": numpy_batch["label_action_type"][index],
                    "encounter_support": (
                        "known_frequent" if int(encounter_values[index]) >= 2
                        else "rare_or_unseen"
                    ),
                    "combat": numpy_batch["combat_id"][index],
                }
                for group_name, key in keys.items():
                    grouped[group_name][key][1] += 1
                    grouped[group_name][key][0] += int(matched)
    if not total:
        raise HumanRecordingError("cannot evaluate an empty combat split")
    per_combat = grouped.pop("combat")
    macro_combat_accuracy = sum(values[0] / values[1] for values in per_combat.values()) / len(per_combat)
    result = {
        "samples": total,
        "combats": len(per_combat),
        "nll": total_nll / total,
        "perplexity": math.exp(min(total_nll / total, 20.0)),
        "top1_accuracy": correct / total,
        "policy_only_top1_accuracy": policy_only_correct / total,
        "top3_accuracy": top3_correct / total,
        "macro_combat_accuracy": macro_combat_accuracy,
        "legal_prediction_rate": legal_predictions / total,
        "random_legal_expected_accuracy": random_legal_expected / total,
        "per_act": {
            key: _accuracy_block(values[0], values[1]) for key, values in sorted(grouped["act"].items())
        },
        "per_combat_difficulty_tier": {
            key: _accuracy_block(values[0], values[1])
            for key, values in sorted(grouped["combat_difficulty_tier"].items())
        },
        "per_act_combat_difficulty_tier": {
            key: _accuracy_block(values[0], values[1])
            for key, values in sorted(
                grouped["act_combat_difficulty_tier"].items()
            )
        },
        "per_room_type": {
            key: _accuracy_block(values[0], values[1])
            for key, values in sorted(grouped["room_type"].items())
        },
        "per_label_action_type": {
            key: _accuracy_block(values[0], values[1])
            for key, values in sorted(grouped["label_action_type"].items())
        },
        "per_encounter_support": {
            key: _accuracy_block(values[0], values[1])
            for key, values in sorted(grouped["encounter_support"].items())
        },
    }
    if resource_samples:
        result["resource_value"] = {
            "samples": resource_samples,
            "hp_loss_fraction_mae": hp_loss_absolute_error / resource_samples,
            "immediate_hp_loss_fraction_mae": immediate_hp_loss_absolute_error / resource_samples,
            "positive_immediate_hp_loss_fraction_mae": (
                positive_immediate_absolute_error / positive_immediate_samples
                if positive_immediate_samples else None
            ),
            "end_turn_immediate_hp_loss_fraction_mae": (
                end_turn_immediate_absolute_error / end_turn_immediate_samples
                if end_turn_immediate_samples else None
            ),
            "death_brier": death_brier / resource_samples,
            "potion_spent_mae": potion_absolute_error / resource_samples,
            "max_hp_delta_mae": max_hp_absolute_error / resource_samples,
        }
    if state_value_samples:
        result["state_value"] = {
            "samples": state_value_samples,
            "hp_loss_fraction_mae": state_hp_loss_absolute_error / state_value_samples,
            "death_brier": state_death_brier / state_value_samples,
            "potion_spent_mae": state_potion_absolute_error / state_value_samples,
            "max_hp_delta_mae": state_max_hp_absolute_error / state_value_samples,
        }
    if outcome_samples:
        result["state_outcome"] = {
            "samples": outcome_samples,
            "bins": model.config.state_outcome_bins,
            "temperature": float(model.state_outcome_temperature.item()),
            "nll": outcome_nll / outcome_samples,
            "raw_nll": outcome_raw_nll / outcome_samples,
            "bin_accuracy": outcome_correct / outcome_samples,
            "within_one_bin_accuracy": outcome_within_one / outcome_samples,
            "expected_end_hp_fraction_mae": outcome_end_hp_absolute_error / outcome_samples,
            "raw_expected_end_hp_fraction_mae": (
                outcome_raw_end_hp_absolute_error / outcome_samples
            ),
            "death_samples": outcome_death_samples,
            "death_brier": outcome_death_brier / outcome_samples,
            "death_ece": _binary_ece(
                outcome_death_probabilities, outcome_death_targets
            ),
        }
    return result


def _fit_state_outcome_temperature(model: Any, loader: Any, *, device: str) -> dict[str, float]:
    torch, F, _, _ = _require_torch()
    from .combat_model import numpy_batch_to_torch

    if model.state_outcome_head is None:
        return {"temperature": 1.0, "validation_nll": 0.0}
    logits_rows = []
    target_rows = []
    model.eval()
    with torch.no_grad():
        for numpy_batch in loader:
            batch = numpy_batch_to_torch(numpy_batch, device=device)
            mask = batch["resource_target_mask"].bool()
            logits_rows.append(model.state_outcome_logits(batch, calibrated=False)[mask])
            target_rows.append(model.state_outcome_targets(batch)[mask])
    logits = torch.cat(logits_rows)
    targets = torch.cat(target_rows)
    candidates = torch.logspace(
        math.log10(0.25), math.log10(4.0), 161, device=logits.device
    )
    losses = torch.stack([
        F.cross_entropy(logits / temperature, targets) for temperature in candidates
    ])
    best = int(losses.argmin().item())
    temperature = float(candidates[best].item())
    model.state_outcome_temperature.fill_(temperature)
    return {
        "temperature": temperature,
        "validation_nll": float(losses[best].item()),
    }


def _validate_training_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise HumanRecordingError("unsupported combat policy training config")
    model = config.get("model", {})
    training = config.get("training", {})
    d_model = int(model.get("d_model", 0))
    nhead = int(model.get("nhead", 0))
    if d_model <= 0 or int(model.get("num_layers", 0)) <= 0 or nhead <= 0:
        raise HumanRecordingError("combat policy model dimensions must be positive")
    if d_model % nhead:
        raise HumanRecordingError("combat policy d_model must be divisible by nhead")
    if int(training.get("batch_size", 0)) <= 0 or int(training.get("epochs", 0)) <= 0:
        raise HumanRecordingError("combat policy batch size and epochs must be positive")
    if bool(model.get("resource_value_heads")) and float(model.get("decision_value_scale", 0.0)) < 0:
        raise HumanRecordingError("combat policy decision value scale cannot be negative")
    outcome_bins = int(model.get("state_outcome_bins", 0))
    if outcome_bins == 1 or outcome_bins < 0:
        raise HumanRecordingError("combat state outcome bins must be zero or at least two")
    candidate_feature_dim = int(model.get("candidate_engine_feature_dim", 0))
    if candidate_feature_dim not in {0, CANDIDATE_ENGINE_FEATURE_DIM}:
        raise HumanRecordingError(
            "combat candidate engine feature dimension does not match the tensorizer"
        )
    selection_metric = str(training.get("selection_metric", "auto"))
    if selection_metric not in {
        "auto",
        "nll",
        "state_value.hp_loss_fraction_mae",
        "state_outcome.raw_nll",
    }:
        raise HumanRecordingError(f"unsupported combat selection metric: {selection_metric}")


def _selection_score(
    metrics: dict[str, Any],
    *,
    requested: str,
    train_state_value: bool,
    train_state_outcome: bool,
) -> tuple[str, float]:
    metric = requested
    if metric == "auto":
        metric = (
            "state_outcome.raw_nll"
            if train_state_outcome
            else "state_value.hp_loss_fraction_mae"
            if train_state_value
            else "nll"
        )
    if metric == "nll":
        return metric, float(metrics["nll"])
    if metric == "state_value.hp_loss_fraction_mae":
        return metric, float(metrics["state_value"]["hp_loss_fraction_mae"])
    if metric == "state_outcome.raw_nll":
        return metric, float(metrics["state_outcome"]["raw_nll"])
    raise AssertionError(metric)


def train_combat_policy(
    *,
    config_path: Path = DEFAULT_TRAINING_CONFIG,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    device: str = "auto",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    initialize_checkpoint: Path | None = None,
    freeze_base: bool = False,
) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    from .combat_model import CombatPolicyConfig, CombatPolicyTransformer, numpy_batch_to_torch

    validate_combat_model_examples()
    config = load_json(config_path)
    _validate_training_config(config)
    vocabulary = load_json(VOCAB_PATH)
    tensorizer = CombatTensorizerV0(vocabulary)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = _select_device(torch, device)

    use_resource_targets = bool(config["model"].get("resource_value_heads", False))
    use_state_value_targets = bool(config["model"].get("state_value_head", False))
    use_state_outcome_targets = int(config["model"].get("state_outcome_bins", 0)) >= 2
    train_policy_loss = bool(config["training"].get("policy_loss_enabled", True))
    train_state_value_targets = use_state_value_targets and bool(
        config["training"].get("state_value_loss_enabled", True)
    )
    use_value_targets = use_resource_targets or use_state_value_targets or use_state_outcome_targets
    value_targets = load_combat_value_targets() if use_value_targets else None
    train_rows = _load_split(
        tensorizer, "train", limit=max_train_samples, seed=seed, value_targets=value_targets
    )
    validation_rows = _load_split(
        tensorizer, "validation", limit=max_eval_samples, seed=seed + 1, value_targets=value_targets
    )
    test_rows = _load_split(
        tensorizer, "test", limit=max_eval_samples, seed=seed + 2, value_targets=value_targets
    )
    training = config["training"]
    batch_size = int(training["batch_size"])
    train_loader = _make_loader(train_rows, batch_size=batch_size, shuffle=True, seed=seed)
    validation_loader = _make_loader(validation_rows, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = _make_loader(test_rows, batch_size=batch_size, shuffle=False, seed=seed)

    model_config = CombatPolicyConfig.from_vocabulary(vocabulary, **config["model"])
    model = CombatPolicyTransformer(model_config).to(selected_device)
    initialization: dict[str, Any] | None = None
    if initialize_checkpoint is not None:
        initialize_checkpoint = initialize_checkpoint.resolve()
        initial = torch.load(initialize_checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(initial["state_dict"], strict=False)
        allowed_missing = {
            name for name in model.state_dict()
            if name.startswith("resource_value_head.")
            or name.startswith("state_value_head.")
            or name.startswith("state_outcome_head.")
            or name.startswith("candidate_engine_projection.")
            or name.startswith("encounter_embedding.")
            or name.startswith("encounter_candidate_projection.")
            or name == "encounter_residual_scale"
            or name == "candidate_engine_scale"
            or name == "state_outcome_temperature"
        }
        allowed_unexpected = {
            name for name in unexpected
            if name.startswith("resource_value_head.")
            or name.startswith("state_value_head.")
            or name.startswith("state_outcome_head.")
            or name == "state_outcome_temperature"
        }
        if set(missing) - allowed_missing or set(unexpected) - allowed_unexpected:
            raise HumanRecordingError(
                f"incompatible initialization checkpoint; missing={missing}, unexpected={unexpected}"
            )
        initialization = {
            "checkpoint": str(initialize_checkpoint),
            "checkpoint_sha256": sha256_file(initialize_checkpoint),
            "missing_resource_parameters": sorted(missing),
            "freeze_base": freeze_base,
        }
    elif freeze_base:
        raise HumanRecordingError("freeze_base requires an initialization checkpoint")
    if freeze_base:
        freeze_action_resource_head = bool(training.get("freeze_action_resource_head", False))
        train_candidate_engine_adapter = bool(
            training.get("train_candidate_engine_adapter", False)
        )
        train_existing_state_value_head = bool(
            training.get("train_existing_state_value_head", True)
        )
        train_encounter_adapter = bool(
            training.get("train_encounter_adapter", False)
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                (name.startswith("resource_value_head.") and not freeze_action_resource_head)
                or (name.startswith("state_value_head.") and train_existing_state_value_head)
                or name.startswith("state_outcome_head.")
                or (
                    train_candidate_engine_adapter
                    and (
                        name.startswith("candidate_engine_projection.")
                        or name == "candidate_engine_scale"
                    )
                )
                or (
                    train_encounter_adapter
                    and (
                        name.startswith("encounter_embedding.")
                        or name.startswith("encounter_candidate_projection.")
                        or name == "encounter_residual_scale"
                    )
                )
            )
    optimized_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not optimized_parameters:
        raise HumanRecordingError("combat policy training has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_validation_score = float("inf")
    best_state: dict[str, Any] | None = None
    patience = 0
    history: list[dict[str, Any]] = []
    epochs_completed = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        # Adapter-only runs freeze the entire pretrained network. Keeping the
        # frozen Transformer and scorers in train mode would still activate
        # their Dropout layers, so the small adapter would optimize against a
        # stochastic target that differs from validation inference. Eval mode
        # preserves gradients while making the frozen base deterministic.
        model.train(not freeze_base)
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_components: dict[str, float] = defaultdict(float)
        for numpy_batch in train_loader:
            batch = numpy_batch_to_torch(numpy_batch, device=selected_device)
            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, Any] = {}
            loss = None
            if train_policy_loss and use_resource_targets:
                losses = model.policy_resource_loss(
                    batch,
                    label_smoothing=float(training.get("label_smoothing", 0.0)),
                    hp_loss_coefficient=float(training.get("hp_loss_coefficient", 1.0)),
                    immediate_hp_loss_coefficient=float(
                        training.get("immediate_hp_loss_coefficient", 1.0)
                    ),
                    immediate_hp_loss_positive_weight=float(
                        training.get("immediate_hp_loss_positive_weight", 8.0)
                    ),
                    death_coefficient=float(training.get("death_coefficient", 0.5)),
                    potion_coefficient=float(training.get("potion_coefficient", 0.25)),
                    max_hp_coefficient=float(training.get("max_hp_coefficient", 0.25)),
                    death_positive_weight=float(training.get("death_positive_weight", 4.0)),
                )
                loss = losses["total"]
            elif train_policy_loss:
                loss = model.behavior_cloning_loss(
                    batch,
                    label_smoothing=float(training.get("label_smoothing", 0.0)),
                )
                losses = {"total": loss, "policy": loss}
            if train_state_value_targets:
                state_losses = model.state_value_loss(
                    batch,
                    hp_loss_coefficient=float(training.get("state_hp_loss_coefficient", 1.0)),
                    death_coefficient=float(training.get("state_death_coefficient", 0.5)),
                    potion_coefficient=float(training.get("state_potion_coefficient", 0.25)),
                    max_hp_coefficient=float(training.get("state_max_hp_coefficient", 0.25)),
                    death_positive_weight=float(training.get("death_positive_weight", 4.0)),
                )
                losses = {
                    **losses,
                    **{f"state_{name}": value for name, value in state_losses.items() if name != "total"},
                }
                loss = state_losses["total"] if loss is None else loss + state_losses["total"]
                losses["total"] = loss
            if use_state_outcome_targets:
                outcome_loss = model.state_outcome_loss(
                    batch,
                    death_weight=float(training.get("state_outcome_death_weight", 4.0)),
                )
                weighted_outcome_loss = (
                    float(training.get("state_outcome_coefficient", 1.0)) * outcome_loss
                )
                loss = weighted_outcome_loss if loss is None else loss + weighted_outcome_loss
                losses["state_outcome"] = outcome_loss
                losses["total"] = loss
            if loss is None:
                raise HumanRecordingError("combat policy training has no enabled loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            count = int(batch["label"].shape[0])
            epoch_loss += float(loss.item()) * count
            for name, component in losses.items():
                epoch_components[name] += float(component.item()) * count
            epoch_samples += count
        validation_metrics = evaluate_combat_policy(model, validation_loader, device=selected_device)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / epoch_samples,
            "train_components": {
                name: value / epoch_samples for name, value in sorted(epoch_components.items())
            },
            "validation": validation_metrics,
        })
        epochs_completed = epoch
        selection_metric, validation_score = _selection_score(
            validation_metrics,
            requested=str(training.get("selection_metric", "auto")),
            train_state_value=train_state_value_targets,
            train_state_outcome=use_state_outcome_targets,
        )
        if validation_score < best_validation_score - 1e-6:
            best_validation_score = validation_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(training["early_stopping_patience"]):
                break
        print(
            f"epoch={epoch} train_loss={epoch_loss / epoch_samples:.6f} "
            f"validation_{selection_metric.replace('.', '_')}="
            f"{validation_score:.6f}",
            file=__import__("sys").stderr,
            flush=True,
        )
    if best_state is None:
        raise HumanRecordingError("combat policy training did not produce a checkpoint")
    model.load_state_dict(best_state)
    outcome_calibration = (
        _fit_state_outcome_temperature(model, validation_loader, device=selected_device)
        if use_state_outcome_targets
        else None
    )
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    validation_metrics = evaluate_combat_policy(model, validation_loader, device=selected_device)
    test_metrics = evaluate_combat_policy(model, test_loader, device=selected_device)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    dataset_index_path = run_root / "dataset_index.json"
    write_json_atomic(dataset_index_path, {
        "source_manifest_sha256": sha256_file(COMBAT_MODEL_ROOT / "manifest.json"),
            "value_target_manifest_sha256": sha256_file(VALUE_MANIFEST_PATH) if use_value_targets else None,
        "splits": {
            "train": [
                {"transition_id": row["transition_id"], "source_transition_sha256": row["source_transition_sha256"]}
                for row in train_rows
            ],
            "validation": [
                {"transition_id": row["transition_id"], "source_transition_sha256": row["source_transition_sha256"]}
                for row in validation_rows
            ],
            "test": [
                {"transition_id": row["transition_id"], "source_transition_sha256": row["source_transition_sha256"]}
                for row in test_rows
            ],
        },
    })
    checkpoint_path = run_root / "model.pt"
    torch.save({
        "model_config": model_config.to_dict(),
        "state_dict": best_state,
        "vocabulary_sha256": sha256_file(VOCAB_PATH),
        "source_manifest_sha256": sha256_file(COMBAT_MODEL_ROOT / "manifest.json"),
        "dataset_index_sha256": sha256_file(dataset_index_path),
        "value_target_manifest_sha256": sha256_file(VALUE_MANIFEST_PATH) if use_value_targets else None,
    }, checkpoint_path)
    shutil.copy2(VOCAB_PATH, run_root / "vocab.json")
    shutil.copy2(config_path, run_root / "training_config.json")
    report = {
        "status": "PASS",
        "run_id": run_id,
        "device": selected_device,
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "epochs_completed": epochs_completed,
        # Retain the original report key so existing analysis scripts can load
        # both policy-only and state-value runs without a schema branch.
        "best_validation_nll": validation_metrics["nll"],
        "best_validation_score": best_validation_score,
        "best_validation_metric": _selection_score(
            validation_metrics,
            requested=str(training.get("selection_metric", "auto")),
            train_state_value=train_state_value_targets,
            train_state_outcome=use_state_outcome_targets,
        )[0],
        "outcome_calibration": outcome_calibration,
        "dataset": {
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "test_samples": len(test_rows),
            "source_manifest_sha256": sha256_file(COMBAT_MODEL_ROOT / "manifest.json"),
        },
        "model": {
            **model_config.to_dict(),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "optimized_parameter_count": sum(parameter.numel() for parameter in optimized_parameters),
        },
        "initialization": initialization,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "vocabulary": str(run_root / "vocab.json"),
            "dataset_index": str(dataset_index_path),
        },
    }
    write_json_atomic(run_root / "metrics.json", report)
    write_json_atomic(artifact_root / "latest.json", {
        "run_id": run_id,
        "metrics": str(run_root / "metrics.json"),
        "checkpoint": str(checkpoint_path),
    })
    return report


def evaluate_combat_checkpoint(
    checkpoint_path: Path,
    *,
    split: str = "test",
    device: str = "auto",
    max_samples: int | None = None,
) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    from .combat_model import SUPPORTED_MODEL_VERSIONS, CombatPolicyConfig, CombatPolicyTransformer

    if split not in {"train", "validation", "test"}:
        raise HumanRecordingError(f"unsupported combat evaluation split: {split}")
    checkpoint_path = checkpoint_path.resolve()
    vocabulary_path = checkpoint_path.parent / "vocab.json"
    dataset_index_path = checkpoint_path.parent / "dataset_index.json"
    if not checkpoint_path.exists() or not vocabulary_path.exists() or not dataset_index_path.exists():
        raise HumanRecordingError("checkpoint, vocab.json and dataset_index.json must all exist")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("vocabulary_sha256") != sha256_file(vocabulary_path):
        raise HumanRecordingError("checkpoint vocabulary fingerprint mismatch")
    if checkpoint.get("dataset_index_sha256") != sha256_file(dataset_index_path):
        raise HumanRecordingError("checkpoint dataset index fingerprint mismatch")
    dataset_index = load_json(dataset_index_path)
    raw_model_config = dict(checkpoint["model_config"])
    if raw_model_config.pop("model_version", None) not in SUPPORTED_MODEL_VERSIONS:
        raise HumanRecordingError("unsupported combat policy checkpoint version")
    model_config = CombatPolicyConfig(**raw_model_config)
    vocabulary = load_json(vocabulary_path)
    tensorizer = CombatTensorizerV0(vocabulary)
    required_sources = {
        row["transition_id"]: row["source_transition_sha256"]
        for row in dataset_index["splits"][split]
    }
    uses_value_targets = (
        model_config.resource_value_heads
        or model_config.state_value_head
        or model_config.state_outcome_bins >= 2
    )
    value_targets = load_combat_value_targets() if uses_value_targets else None
    if uses_value_targets:
        expected_value_sha256 = checkpoint.get("value_target_manifest_sha256")
        if expected_value_sha256 != sha256_file(VALUE_MANIFEST_PATH):
            raise HumanRecordingError("checkpoint combat resource targets changed")
    rows = _load_split(
        tensorizer,
        split,
        limit=max_samples,
        seed=0,
        required_sources=required_sources,
        value_targets=value_targets,
    )
    loader = _make_loader(rows, batch_size=128, shuffle=False, seed=0)
    selected_device = _select_device(torch, device)
    model = CombatPolicyTransformer(model_config).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": split,
        "device": selected_device,
        "current_source_manifest_matches": (
            checkpoint.get("source_manifest_sha256") == sha256_file(COMBAT_MODEL_ROOT / "manifest.json")
        ),
        "metrics": evaluate_combat_policy(model, loader, device=selected_device),
    }
