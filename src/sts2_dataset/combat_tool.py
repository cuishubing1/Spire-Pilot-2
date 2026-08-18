from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .combat_directive import (
    COMBAT_DIRECTIVE_VERSION,
    CandidateMechanicFactV0,
    CombatDirectiveV0,
)
from .combat_engine_features import candidate_preview_features, ground_future_max_hp_delta
from .combat_model import (
    SUPPORTED_MODEL_VERSIONS,
    CombatObjective,
    CombatPolicyConfig,
    CombatPolicyTransformer,
    numpy_batch_to_torch,
)
from .combat_online import visible_intent_end_turn_hp_loss
from .combat_tensorizer import CombatTensorizerV0, collate_combat_numpy
from .human import HumanRecordingError
from .util import load_json, sha256_file


COMBAT_TOOL_VERSION = "combat-tool-0.2.0"


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise HumanRecordingError("CUDA was requested but is unavailable")
    return device


def load_combat_tool_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "auto",
) -> "CombatToolV0":
    """Load a fingerprint-checked checkpoint as an auditable Combat Tool."""

    checkpoint_path = checkpoint_path.resolve()
    vocabulary_path = checkpoint_path.parent / "vocab.json"
    dataset_index_path = checkpoint_path.parent / "dataset_index.json"
    for required in (checkpoint_path, vocabulary_path, dataset_index_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocabulary_sha256 = sha256_file(vocabulary_path)
    dataset_index_sha256 = sha256_file(dataset_index_path)
    if checkpoint.get("vocabulary_sha256") != vocabulary_sha256:
        raise HumanRecordingError("checkpoint vocabulary fingerprint mismatch")
    if checkpoint.get("dataset_index_sha256") != dataset_index_sha256:
        raise HumanRecordingError("checkpoint dataset index fingerprint mismatch")
    raw_config = dict(checkpoint["model_config"])
    model_version = raw_config.pop("model_version", None)
    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise HumanRecordingError("unsupported combat policy checkpoint version")
    selected_device = _select_device(device)
    model = CombatPolicyTransformer(CombatPolicyConfig(**raw_config)).to(selected_device)
    model.checkpoint_model_version = str(model_version)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return CombatToolV0(
        model,
        CombatTensorizerV0(load_json(vocabulary_path)),
        device=selected_device,
        model_provenance={
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_version": str(model_version),
            "vocabulary_sha256": vocabulary_sha256,
            "dataset_index_sha256": dataset_index_sha256,
            "source_manifest_sha256": checkpoint.get("source_manifest_sha256"),
            "device": selected_device,
        },
    )


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    values = [max(0.0, float(value)) for value in probabilities]
    total = sum(values)
    if len(values) <= 1 or total <= 0.0:
        return 0.0
    normalized = [value / total for value in values if value > 0.0]
    return -sum(value * math.log(value) for value in normalized) / math.log(len(values))


def _candidate_source_id(sample: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    direct = candidate.get("source_id")
    if direct:
        return str(direct)
    source_ref = candidate.get("source_ref")
    if source_ref is None:
        return None
    observation = sample.get("observation") or {}
    for key in ("hand", "potions"):
        for row in observation.get(key) or []:
            if isinstance(row, dict) and str(row.get("entity_ref")) == str(source_ref):
                return str(row.get("id")) if row.get("id") else None
    return None


def _mechanic_facts_by_candidate(
    facts: Sequence[CandidateMechanicFactV0 | dict[str, Any]],
    candidate_ids: set[str],
) -> dict[str, list[CandidateMechanicFactV0]]:
    result: dict[str, list[CandidateMechanicFactV0]] = defaultdict(list)
    for raw in facts:
        fact = raw if isinstance(raw, CandidateMechanicFactV0) else CandidateMechanicFactV0.from_dict(raw)
        if fact.candidate_id not in candidate_ids:
            raise ValueError(f"mechanic fact references unknown candidate: {fact.candidate_id}")
        result[fact.candidate_id].append(fact)
    return result


def _directive_adjustment(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    directive: CombatDirectiveV0,
    facts: Sequence[CandidateMechanicFactV0],
) -> tuple[float, list[str], list[str], dict[str, float]]:
    candidate_id = str(candidate["candidate_id"])
    action_type = str(candidate.get("action_type") or "")
    target_ref = str(candidate.get("target_ref")) if candidate.get("target_ref") is not None else None
    source_id = _candidate_source_id(sample, candidate)
    exclusions: list[str] = []
    if action_type in directive.forbidden_action_types:
        exclusions.append("directive_forbidden_action_type")
    if candidate_id in directive.forbidden_candidate_ids:
        exclusions.append("directive_forbidden_candidate")
    if action_type == "use_potion":
        if (
            directive.max_potion_uses is not None
            and directive.potion_uses_so_far >= directive.max_potion_uses
        ):
            exclusions.append("directive_potion_budget_exhausted")
        if source_id is not None and source_id in directive.preserve_potion_ids:
            exclusions.append("directive_preserve_potion")

    components = {
        "action_type_bias": float(directive.action_type_biases.get(action_type, 0.0)),
        "target_bias": float(directive.target_biases.get(target_ref, 0.0))
        if target_ref is not None else 0.0,
        "candidate_bias": float(directive.candidate_biases.get(candidate_id, 0.0)),
        "mechanic_bias": sum(float(row.score_delta) for row in facts),
    }
    mechanic_ids = [row.mechanic_id for row in facts]
    if any(row.hard_forbidden for row in facts):
        exclusions.append("engine_mechanic_forbidden")
    return sum(components.values()), exclusions, mechanic_ids, components


def rank_combat_actions(
    model: CombatPolicyTransformer,
    tensorizer: CombatTensorizerV0,
    sample: dict[str, Any],
    *,
    device: str,
    objective: CombatObjective | None = None,
    directive: CombatDirectiveV0 | dict[str, Any] | None = None,
    mechanic_facts: Sequence[CandidateMechanicFactV0 | dict[str, Any]] = (),
    reuse_entity_encoding: bool = True,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Rank legal combat actions and expose every runtime score component."""

    parsed_directive = (
        directive
        if isinstance(directive, CombatDirectiveV0)
        else CombatDirectiveV0.from_dict(directive)
    )
    if directive is not None and objective is not None:
        raise ValueError("pass either objective or directive, not both")
    runtime_objective = parsed_directive.objective(model.config) if directive is not None else objective
    candidates = sample.get("candidates") or []
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if not candidates or any(not identity for identity in candidate_ids):
        raise ValueError("Combat Tool requires non-empty candidates with candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Combat Tool candidate ids must be unique")
    facts_by_candidate = _mechanic_facts_by_candidate(mechanic_facts, set(candidate_ids))

    tensorized = tensorizer.tensorize(sample)
    batch = numpy_batch_to_torch(collate_combat_numpy([tensorized]), device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        shared_state_predictions = None
        if model.resource_value_head is not None:
            if reuse_entity_encoding and model.state_value_head is not None:
                (
                    policy_logits,
                    raw_resource,
                    decision_logits,
                    shared_state_predictions,
                ) = model.policy_resource_state_outputs(
                    batch, objective=runtime_objective
                )
            else:
                policy_logits, raw_resource, decision_logits = model.policy_resource_outputs(
                    batch, objective=runtime_objective
                )
            model_immediate = raw_resource["immediate_hp_loss_fraction"].clone()
            model_max_hp = raw_resource["max_hp_delta"].clone()
            immediate = model_immediate.clone()
            grounded_max_hp = model_max_hp.clone()
            immediate_sources = ["learned"] * len(candidates)
            max_hp_grounding: list[dict[str, Any]] = []
            grounded_end_turn = visible_intent_end_turn_hp_loss(sample["observation"])
            if runtime_objective is not None:
                decision_logits = decision_logits.clone()
                for index, candidate in enumerate(candidates):
                    max_hp_fact = ground_future_max_hp_delta(
                        float(model_max_hp[0, index].item()), sample["observation"]
                    )
                    max_hp_grounding.append(max_hp_fact)
                    max_hp_value = grounded_max_hp.new_tensor(max_hp_fact["grounded_prediction"])
                    decision_logits[0, index] += (
                        runtime_objective.decision_value_scale
                        * runtime_objective.max_hp_gain_weight
                        * (max_hp_value - model_max_hp[0, index])
                    )
                    grounded_max_hp[0, index] = max_hp_value
                    if grounded_end_turn is not None and candidate["action_type"] == "end_turn":
                        learned = immediate[0, index]
                        grounded = learned.new_tensor(grounded_end_turn["hp_loss_fraction"])
                        decision_logits[0, index] += (
                            runtime_objective.decision_value_scale
                            * runtime_objective.immediate_hp_loss_weight
                            * (learned - grounded)
                        )
                        immediate[0, index] = grounded
                        immediate_sources[index] = "visible_intent_rule"
                raw_resource = {
                    **raw_resource,
                    "immediate_hp_loss_fraction": immediate,
                    "max_hp_delta": grounded_max_hp,
                }
            else:
                max_hp_grounding = [
                    ground_future_max_hp_delta(
                        float(model_max_hp[0, index].item()), sample["observation"]
                    )
                    for index in range(len(candidates))
                ]
        else:
            policy_logits = model(batch)
            decision_logits = policy_logits.clone()
            raw_resource = None
            model_immediate = None
            model_max_hp = None
            max_hp_grounding = []
            immediate_sources = ["unavailable"] * len(candidates)
            grounded_end_turn = None

        state_risk = None
        if model.state_value_head is not None:
            state_predictions = (
                shared_state_predictions
                if shared_state_predictions is not None
                else model.state_value_predictions(batch)
            )
            state_max_hp = float(state_predictions["max_hp_delta"][0].item())
            grounded_state_max_hp = ground_future_max_hp_delta(
                state_max_hp, sample["observation"]
            )
            visible_end_turn = visible_intent_end_turn_hp_loss(sample["observation"])
            state_risk = {
                "death_probability": float(
                    torch.sigmoid(state_predictions["death_logit"][0]).item()
                ),
                "hp_loss_fraction": float(
                    state_predictions["hp_loss_fraction"][0].item()
                ),
                "immediate_hp_loss_fraction": (
                    float(visible_end_turn["hp_loss_fraction"])
                    if visible_end_turn is not None else None
                ),
                "immediate_hp_loss_source": (
                    "visible_intent_rule" if visible_end_turn is not None else "unavailable"
                ),
                "potion_spent": float(state_predictions["potion_spent"][0].item()),
                "max_hp_delta": float(grounded_state_max_hp["grounded_prediction"]),
                "max_hp_delta_model": state_max_hp,
                "max_hp_delta_source": grounded_state_max_hp["source"],
            }

        base_decision_logits = decision_logits.clone()
        directive_biases: list[float] = []
        exclusion_reasons: list[list[str]] = []
        mechanic_ids: list[list[str]] = []
        adjustment_components: list[dict[str, float]] = []
        for index, candidate in enumerate(candidates):
            adjustment, exclusions, ids, components = _directive_adjustment(
                sample,
                candidate,
                parsed_directive,
                facts_by_candidate.get(candidate_ids[index], ()),
            )
            directive_biases.append(adjustment)
            exclusion_reasons.append(exclusions)
            mechanic_ids.append(ids)
            adjustment_components.append(components)
            decision_logits[0, index] += adjustment
        eligible = torch.tensor(
            [not reasons for reasons in exclusion_reasons],
            dtype=torch.bool,
            device=decision_logits.device,
        )
        directive_conflict = not bool(eligible.any().item())
        if directive_conflict:
            eligible = torch.ones_like(eligible)
        decision_logits = decision_logits.masked_fill(~eligible.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(decision_logits, dim=1)[0].detach().cpu().tolist()
        policy_probabilities = torch.softmax(policy_logits, dim=1)[0].detach().cpu().tolist()
        policy_scores = policy_logits[0].detach().cpu().tolist()
        base_scores = base_decision_logits[0].detach().cpu().tolist()
        final_scores = decision_logits[0].detach().cpu().tolist()
        resource = (
            {key: value[0].detach().cpu().tolist() for key, value in raw_resource.items()}
            if raw_resource is not None else None
        )
        model_immediate_values = (
            model_immediate[0].detach().cpu().tolist() if model_immediate is not None else None
        )
        model_max_hp_values = (
            model_max_hp[0].detach().cpu().tolist() if model_max_hp is not None else None
        )
    inference_ms = (time.perf_counter() - started) * 1000.0

    ranked: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        resource_prediction = None
        if resource is not None:
            resource_prediction = {
                "hp_loss_fraction": round(float(resource["hp_loss_fraction"][index]), 6),
                "immediate_hp_loss_fraction": round(
                    float(resource["immediate_hp_loss_fraction"][index]), 6
                ),
                "immediate_hp_loss_fraction_model": round(
                    float(model_immediate_values[index]), 6
                ),
                "immediate_hp_loss_source": immediate_sources[index],
                "visible_intent": (
                    {key: round(float(value), 6) for key, value in grounded_end_turn.items()}
                    if immediate_sources[index] == "visible_intent_rule" else None
                ),
                "death_probability": round(
                    float(torch.sigmoid(torch.tensor(resource["death_logit"][index])).item()), 6
                ),
                "potion_spent": round(float(resource["potion_spent"][index]), 6),
                "max_hp_delta": round(float(resource["max_hp_delta"][index]), 6),
                "max_hp_delta_model": round(float(model_max_hp_values[index]), 6),
                "max_hp_delta_source": max_hp_grounding[index]["source"],
                "max_hp_positive_growth_cap": round(
                    float(max_hp_grounding[index]["positive_growth_cap"]), 6
                ),
                "max_hp_negative_loss_cap": round(
                    float(max_hp_grounding[index]["negative_loss_cap"]), 6
                ),
            }
        base_score = float(base_scores[index])
        policy_score = float(policy_scores[index])
        final_score = float(final_scores[index])
        ranked.append({
            "candidate_index": index,
            "probability": round(float(probabilities[index]), 8),
            "policy_probability": round(float(policy_probabilities[index]), 8),
            "eligible": bool(eligible[index].item()),
            "exclusion_reasons": exclusion_reasons[index],
            "candidate": candidate,
            "engine_preview": candidate_preview_features(sample["observation"], candidate),
            "mechanic_ids": mechanic_ids[index],
            "score_breakdown": {
                "policy_logit": round(policy_score, 6),
                "resource_utility": round(base_score - policy_score, 6),
                **{key: round(value, 6) for key, value in adjustment_components[index].items()},
                "final_logit": round(final_score, 6) if math.isfinite(final_score) else None,
            },
            "resource_prediction": resource_prediction,
        })
    ranked.sort(
        key=lambda row: (bool(row["eligible"]), float(row["probability"]), -int(row["candidate_index"])),
        reverse=True,
    )
    diagnostics = {
        "directive": parsed_directive.to_dict(),
        "directive_conflict_fallback": directive_conflict,
        "eligible_candidate_count": sum(bool(row["eligible"]) for row in ranked),
        "state_risk": state_risk,
    }
    return ranked, inference_ms, diagnostics


class CombatToolV0:
    def __init__(
        self,
        model: CombatPolicyTransformer,
        tensorizer: CombatTensorizerV0,
        *,
        device: str,
        model_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tensorizer = tensorizer
        self.device = device
        self.model_provenance = model_provenance or {
            "checkpoint_sha256": None,
            "model_version": getattr(model, "checkpoint_model_version", None),
            "vocabulary_sha256": None,
            "dataset_index_sha256": None,
            "source_manifest_sha256": None,
            "device": device,
        }

    def decide(
        self,
        sample: dict[str, Any],
        *,
        directive: CombatDirectiveV0 | dict[str, Any] | None = None,
        mechanic_facts: Sequence[CandidateMechanicFactV0 | dict[str, Any]] = (),
        top_k: int = 3,
    ) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        parsed_directive = (
            directive
            if isinstance(directive, CombatDirectiveV0)
            else CombatDirectiveV0.from_dict(directive)
        )
        ranked, inference_ms, diagnostics = rank_combat_actions(
            self.model,
            self.tensorizer,
            sample,
            device=self.device,
            directive=parsed_directive,
            mechanic_facts=mechanic_facts,
        )
        selected = ranked[0]
        probabilities = [float(row["probability"]) for row in ranked if row["eligible"]]
        entropy = _normalized_entropy(probabilities)
        ordered_probabilities = sorted(probabilities, reverse=True)
        top_gap = (
            ordered_probabilities[0] - ordered_probabilities[1]
            if len(ordered_probabilities) > 1 else 1.0
        )
        reasons: list[str] = []
        if diagnostics["directive_conflict_fallback"]:
            reasons.append("directive_conflict_fallback")
        if entropy >= parsed_directive.replan_normalized_entropy:
            reasons.append("high_action_entropy")
        if top_gap <= parsed_directive.replan_top_probability_gap:
            reasons.append("small_top_probability_gap")
        resource = selected.get("resource_prediction")
        state_risk = diagnostics.get("state_risk")
        # State risk is the safer replan signal: it is supervised for every
        # observed state. Candidate resource values are supervised only on the
        # human-selected action and remain diagnostic for counterfactual
        # candidates until exact successor labels are available.
        risk = state_risk or resource or {}
        risk_source = (
            "state_value_head"
            if state_risk is not None else
            "candidate_resource_head"
            if resource is not None else
            "unavailable"
        )
        death = float(risk.get("death_probability") or 0.0)
        hp_loss = float(risk.get("hp_loss_fraction") or 0.0)
        if death >= parsed_directive.replan_death_probability:
            reasons.append("predicted_death_risk")
        if (
            parsed_directive.acceptable_hp_loss_fraction is not None
            and hp_loss > parsed_directive.acceptable_hp_loss_fraction
        ):
            reasons.append("predicted_hp_loss_above_budget")
        requested_search = parsed_directive.search_policy
        recommended_mode = requested_search.mode
        if requested_search.mode == "policy_only" and reasons:
            recommended_mode = "one_step"
        objective_fields = sorted(parsed_directive.objective_overrides)
        objective_scale = float(
            parsed_directive.objective(self.model.config).decision_value_scale
        )
        candidate_resource_available = self.model.resource_value_head is not None
        objective_reranking_active = bool(
            candidate_resource_available and objective_scale > 0.0
        )
        ignored_objective_fields = (
            objective_fields if objective_fields and not objective_reranking_active else []
        )
        if not ignored_objective_fields:
            ignored_objective_reason = None
        elif not candidate_resource_available:
            ignored_objective_reason = "candidate_resource_head_unavailable"
        else:
            ignored_objective_reason = "decision_value_scale_zero"
        return {
            "schema_version": COMBAT_TOOL_VERSION,
            "directive_version": COMBAT_DIRECTIVE_VERSION,
            "status": "directive_conflict_fallback"
            if diagnostics["directive_conflict_fallback"] else "ok",
            "model_provenance": dict(self.model_provenance),
            "transition_id": sample.get("transition_id"),
            "combat_id": sample.get("combat_id"),
            "chosen_action": selected["candidate"],
            "chosen": selected,
            "top_k": ranked[: min(top_k, len(ranked))],
            "ranked_actions": ranked,
            "uncertainty": {
                "normalized_entropy": round(entropy, 6),
                "top_probability_gap": round(top_gap, 6),
            },
            "predicted_risk": {
                "source": risk_source,
                "death_probability": round(death, 6),
                "hp_loss_fraction": round(hp_loss, 6),
                "immediate_hp_loss_fraction": risk.get("immediate_hp_loss_fraction"),
                "immediate_hp_loss_source": risk.get("immediate_hp_loss_source"),
                "potion_spent": risk.get("potion_spent"),
                "max_hp_delta": risk.get("max_hp_delta"),
            },
            "capabilities": {
                "policy_ranking": "active",
                "candidate_resource_prediction": (
                    "diagnostic_on_policy" if candidate_resource_available else "unavailable"
                ),
                "state_risk_prediction": (
                    "diagnostic_on_policy" if state_risk is not None else "unavailable"
                ),
                "visible_end_turn_damage": (
                    "exact_rule" if risk.get("immediate_hp_loss_source") == "visible_intent_rule"
                    else "unavailable"
                ),
                "objective_reranking": (
                    "experimental_on_policy" if objective_reranking_active else "inactive"
                ),
                "hard_constraints": "active",
                "explicit_biases": "active",
                "search_execution": "unavailable",
            },
            "directive_effects": {
                "objective_overrides_requested": objective_fields,
                "objective_reranking_applied": objective_reranking_active,
                "ignored_objective_overrides": ignored_objective_fields,
                "ignored_objective_reason": ignored_objective_reason,
            },
            "request_replan": {
                "required": bool(reasons),
                "reasons": reasons,
            },
            "search_request": {
                **requested_search.to_dict(),
                "recommended_mode": recommended_mode,
                "execute_search": requested_search.mode != "policy_only",
                "search_executed": False,
                "trigger_reasons": reasons,
            },
            "applied_directive": parsed_directive.to_dict(),
            "mechanic_fact_count": len(mechanic_facts),
            "inference_ms": round(inference_ms, 3),
        }
