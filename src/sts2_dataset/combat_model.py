from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODEL_VERSION = "combat-policy-transformer-0.6.0"
SUPPORTED_MODEL_VERSIONS = {
    "combat-policy-transformer-0.2.0",
    "combat-policy-transformer-0.3.0",
    "combat-policy-transformer-0.4.0",
    "combat-policy-transformer-0.5.0",
    MODEL_VERSION,
}


@dataclass(frozen=True)
class CombatPolicyConfig:
    entity_type_count: int
    entity_identity_count: int
    action_type_count: int
    target_kind_count: int
    encounter_identity_count: int = 0
    encounter_embedding_adapter: bool = False
    encounter_residual_adapter: bool = False
    numeric_feature_dim: int = 64
    categorical_feature_dim: int = 64
    candidate_engine_feature_dim: int = 0
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    resource_value_heads: bool = False
    state_value_head: bool = False
    state_outcome_bins: int = 0
    decision_value_scale: float = 0.0
    hp_loss_weight: float = 2.0
    immediate_hp_loss_weight: float = 4.0
    death_penalty: float = 5.0
    potion_cost: float = 0.25
    max_hp_gain_weight: float = 0.25

    @classmethod
    def from_vocabulary(cls, vocabulary: dict[str, Any], **overrides: Any) -> "CombatPolicyConfig":
        use_encounter_identity = bool(
            overrides.get("encounter_embedding_adapter", False)
            or overrides.get("encounter_residual_adapter", False)
        )
        values = {
            "entity_type_count": len(vocabulary["entity_types"]),
            "entity_identity_count": len(vocabulary["entity_identity"]),
            "action_type_count": len(vocabulary["action_types"]),
            "target_kind_count": len(vocabulary["target_kinds"]),
            "encounter_identity_count": (
                len(vocabulary.get("encounter_identity", []))
                if use_encounter_identity else 0
            ),
            "numeric_feature_dim": int(vocabulary["numeric_feature_dim"]),
            "categorical_feature_dim": int(vocabulary["categorical_feature_dim"]),
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {"model_version": MODEL_VERSION, **asdict(self)}


@dataclass(frozen=True)
class CombatObjective:
    """Runtime resource preferences supplied by the future upper-level planner."""

    decision_value_scale: float
    hp_loss_weight: float
    immediate_hp_loss_weight: float
    death_penalty: float
    potion_cost: float
    max_hp_gain_weight: float

    @classmethod
    def from_config(cls, config: CombatPolicyConfig, **overrides: float | None) -> "CombatObjective":
        values = {
            "decision_value_scale": config.decision_value_scale,
            "hp_loss_weight": config.hp_loss_weight,
            "immediate_hp_loss_weight": config.immediate_hp_loss_weight,
            "death_penalty": config.death_penalty,
            "potion_cost": config.potion_cost,
            "max_hp_gain_weight": config.max_hp_gain_weight,
        }
        values.update({key: float(value) for key, value in overrides.items() if value is not None})
        return cls(**values)


class CombatPolicyTransformer(nn.Module):
    """Encode visible entities, then score only the supplied legal actions.

    Entity position zero is the global/player entity. Candidate actions bind to
    source and target indices produced by CombatTensorizerV0. The network does
    not use a fixed card-action output vocabulary.
    """

    def __init__(self, config: CombatPolicyConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.entity_type_embedding = nn.Embedding(config.entity_type_count, d_model, padding_idx=0)
        self.entity_identity_embedding = nn.Embedding(config.entity_identity_count, d_model, padding_idx=0)
        self.entity_numeric_projection = nn.Linear(config.numeric_feature_dim, d_model)
        self.entity_categorical_projection = nn.Linear(config.categorical_feature_dim, d_model, bias=False)
        self.entity_input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.entity_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.entity_output_norm = nn.LayerNorm(d_model)
        self.encounter_embedding = (
            nn.Embedding(config.encounter_identity_count, d_model, padding_idx=0)
            if config.encounter_identity_count > 0
            else None
        )
        if self.encounter_embedding is not None:
            nn.init.zeros_(self.encounter_embedding.weight)
        self.encounter_candidate_projection = (
            nn.Linear(4 * d_model, d_model, bias=False)
            if config.encounter_residual_adapter and self.encounter_embedding is not None
            else None
        )
        if self.encounter_candidate_projection is not None:
            self.encounter_residual_scale = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("encounter_residual_scale", None)

        self.action_type_embedding = nn.Embedding(config.action_type_count, d_model, padding_idx=0)
        self.target_kind_embedding = nn.Embedding(config.target_kind_count, d_model, padding_idx=0)
        self.candidate_engine_projection = (
            nn.Sequential(
                nn.Linear(config.candidate_engine_feature_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if config.candidate_engine_feature_dim > 0
            else None
        )
        if self.candidate_engine_projection is not None:
            self.candidate_engine_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("candidate_engine_scale", None)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 1),
        )
        self.resource_value_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 5),
        ) if config.resource_value_heads else None
        self.state_value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 4),
        ) if config.state_value_head else None
        self.state_outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, config.state_outcome_bins),
        ) if config.state_outcome_bins >= 2 else None
        if self.state_outcome_head is not None:
            self.register_buffer("state_outcome_temperature", torch.ones(()))

    @staticmethod
    def _gather_entities(encoded: Tensor, indices: Tensor) -> Tensor:
        gather_index = indices.long().unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
        return encoded.gather(1, gather_index)

    def _encode_entities(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        entity_mask = batch["entity_mask"].bool()
        if entity_mask.ndim != 2:
            raise ValueError("entity_mask must be [batch, length]")
        if not torch.all(entity_mask[:, 0]):
            raise ValueError("every combat sample must contain the global entity at index zero")
        entity = (
            self.entity_type_embedding(batch["entity_type"])
            + self.entity_identity_embedding(batch["entity_identity"])
            + self.entity_numeric_projection(batch["entity_numeric"])
            + self.entity_categorical_projection(batch["entity_categorical"])
        )
        entity = self.entity_input_norm(entity)
        encoded = self.entity_encoder(entity, src_key_padding_mask=~entity_mask)
        encoded = self.entity_output_norm(encoded)
        return encoded, entity_mask

    def _candidate_features_from_encoded(
        self,
        batch: dict[str, Tensor],
        encoded: Tensor,
    ) -> tuple[Tensor, Tensor]:
        action_mask = batch["action_mask"].bool()
        if action_mask.ndim != 2:
            raise ValueError("action_mask must be [batch, length]")
        if not torch.all(action_mask.any(dim=1)):
            raise ValueError("every combat sample must contain at least one legal action")

        state = encoded[:, :1, :].expand(-1, action_mask.shape[1], -1)
        if self.encounter_embedding is not None and not self.config.encounter_residual_adapter:
            encounter_identity = batch.get("encounter_identity")
            if encounter_identity is None:
                raise ValueError("encounter_identity is required by this checkpoint")
            state = state + self.encounter_embedding(
                encounter_identity.long()
            )[:, None, :]
        source = self._gather_entities(encoded, batch["action_source"])
        target = self._gather_entities(encoded, batch["action_target"])
        action = (
            self.action_type_embedding(batch["action_type"])
            + self.target_kind_embedding(batch["action_target_kind"])
        )
        if self.candidate_engine_projection is not None:
            engine_features = batch.get("candidate_engine_numeric")
            if engine_features is None:
                raise ValueError("candidate engine features are required by this checkpoint")
            if engine_features.ndim != 3:
                raise ValueError("candidate_engine_numeric must be [batch, actions, features]")
            if engine_features.shape[:2] != action.shape[:2]:
                raise ValueError("candidate engine feature action dimensions do not match")
            if engine_features.shape[-1] != self.config.candidate_engine_feature_dim:
                raise ValueError("candidate engine feature width does not match checkpoint")
            action = action + self.candidate_engine_scale * self.candidate_engine_projection(
                engine_features.float()
            )
        return torch.cat((state, source, target, action), dim=-1), action_mask

    def _policy_logits_from_features(
        self,
        batch: dict[str, Tensor],
        features: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        logits = self.candidate_scorer(features).squeeze(-1)
        if self.encounter_candidate_projection is not None:
            encounter_identity = batch.get("encounter_identity")
            if encounter_identity is None:
                raise ValueError("encounter_identity is required by this checkpoint")
            assert self.encounter_embedding is not None
            encounter = self.encounter_embedding(encounter_identity.long())
            candidate = self.encounter_candidate_projection(features)
            residual = torch.einsum("bad,bd->ba", candidate, encounter) / math.sqrt(
                self.config.d_model
            )
            logits = logits + self.encounter_residual_scale * residual
        return logits.masked_fill(~action_mask, float("-inf"))

    def _candidate_features(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        encoded, _ = self._encode_entities(batch)
        return self._candidate_features_from_encoded(batch, encoded)

    def _state_value_predictions_from_encoded(
        self, encoded: Tensor
    ) -> dict[str, Tensor]:
        if self.state_value_head is None:
            raise ValueError("state value head is disabled")
        raw = self.state_value_head(encoded[:, 0, :])
        return {
            "hp_loss_fraction": torch.sigmoid(raw[..., 0]),
            "death_logit": raw[..., 1],
            "potion_spent": F.softplus(raw[..., 2]),
            "max_hp_delta": raw[..., 3],
        }

    def state_value_predictions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        encoded, _ = self._encode_entities(batch)
        return self._state_value_predictions_from_encoded(encoded)

    def state_outcome_logits(
        self, batch: dict[str, Tensor], *, calibrated: bool = True
    ) -> Tensor:
        if self.state_outcome_head is None:
            raise ValueError("state outcome distribution head is disabled")
        encoded, _ = self._encode_entities(batch)
        logits = self.state_outcome_head(encoded[:, 0, :])
        if calibrated:
            logits = logits / self.state_outcome_temperature.clamp_min(1e-3)
        return logits

    def state_outcome_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.state_outcome_bins < 2:
            raise ValueError("state outcome distribution head is disabled")
        survivor_bins = self.config.state_outcome_bins - 1
        survivor = (
            batch["target_terminal_hp_fraction"].float()
            .clamp(0.0, 1.0)
            .mul(survivor_bins)
            .floor()
            .long()
            .clamp(max=survivor_bins - 1)
            .add(1)
        )
        return torch.where(batch["target_death"].bool(), torch.zeros_like(survivor), survivor)

    def state_outcome_predictions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = self.state_outcome_logits(batch)
        probabilities = logits.softmax(dim=-1)
        survivor_bins = self.config.state_outcome_bins - 1
        centers = (
            torch.arange(survivor_bins, device=logits.device, dtype=logits.dtype) + 0.5
        ) / survivor_bins
        expected_end_hp_fraction = (probabilities[..., 1:] * centers).sum(dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "death_probability": probabilities[..., 0],
            "expected_end_hp_fraction": expected_end_hp_fraction,
        }

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features, action_mask = self._candidate_features(batch)
        return self._policy_logits_from_features(batch, features, action_mask)

    def resource_predictions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.resource_value_head is None:
            raise ValueError("resource value heads are disabled")
        features, action_mask = self._candidate_features(batch)
        return self._resource_predictions_from_features(features, action_mask)

    def _resource_predictions_from_features(
        self, features: Tensor, action_mask: Tensor
    ) -> dict[str, Tensor]:
        assert self.resource_value_head is not None
        raw = self.resource_value_head(features)
        return {
            "hp_loss_fraction": torch.sigmoid(raw[..., 0]).masked_fill(~action_mask, 0.0),
            "immediate_hp_loss_fraction": torch.sigmoid(raw[..., 1]).masked_fill(~action_mask, 0.0),
            "death_logit": raw[..., 2].masked_fill(~action_mask, 0.0),
            "potion_spent": F.softplus(raw[..., 3]).masked_fill(~action_mask, 0.0),
            "max_hp_delta": raw[..., 4].masked_fill(~action_mask, 0.0),
        }

    def decision_logits(
        self, batch: dict[str, Tensor], objective: CombatObjective | None = None
    ) -> Tensor:
        if self.resource_value_head is None:
            return self(batch)
        logits, _, decision = self.policy_resource_outputs(batch, objective=objective)
        return decision

    def policy_resource_outputs(
        self, batch: dict[str, Tensor], *, objective: CombatObjective | None = None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        if self.resource_value_head is None:
            raise ValueError("resource value heads are disabled")
        encoded, _ = self._encode_entities(batch)
        return self._policy_resource_outputs_from_encoded(
            batch, encoded, objective=objective
        )

    def _policy_resource_outputs_from_encoded(
        self,
        batch: dict[str, Tensor],
        encoded: Tensor,
        *,
        objective: CombatObjective | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        if self.resource_value_head is None:
            raise ValueError("resource value heads are disabled")
        features, action_mask = self._candidate_features_from_encoded(batch, encoded)
        logits = self._policy_logits_from_features(batch, features, action_mask)
        values = self._resource_predictions_from_features(features, action_mask)
        objective = objective or CombatObjective.from_config(self.config)
        if objective.decision_value_scale == 0.0:
            return logits, values, logits
        utility = (
            -objective.hp_loss_weight * values["hp_loss_fraction"]
            -objective.immediate_hp_loss_weight * values["immediate_hp_loss_fraction"]
            -objective.death_penalty * torch.sigmoid(values["death_logit"])
            -objective.potion_cost * values["potion_spent"]
            +objective.max_hp_gain_weight * values["max_hp_delta"]
        )
        return logits, values, logits + objective.decision_value_scale * utility

    def policy_resource_state_outputs(
        self, batch: dict[str, Tensor], *, objective: CombatObjective | None = None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, dict[str, Tensor]]:
        """Evaluate policy, action resources, and state value with one encoder pass.

        This is an inference optimization for checkpoints that expose both
        resource-value and state-value heads.  It deliberately reuses only the
        immutable entity encoding; every output head still runs normally.
        """

        if self.resource_value_head is None:
            raise ValueError("resource value heads are disabled")
        if self.state_value_head is None:
            raise ValueError("state value head is disabled")
        encoded, _ = self._encode_entities(batch)
        logits, values, decision = self._policy_resource_outputs_from_encoded(
            batch, encoded, objective=objective
        )
        state = self._state_value_predictions_from_encoded(encoded)
        return logits, values, decision, state

    def behavior_cloning_loss(
        self,
        batch: dict[str, Tensor],
        *,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        logits = self(batch)
        labels = batch["label"].long()
        action_mask = batch["action_mask"].bool()
        if not torch.all(action_mask.gather(1, labels[:, None]).squeeze(1)):
            raise ValueError("a behavior-cloning label points to a padded action")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if label_smoothing == 0.0:
            return F.cross_entropy(logits, labels)
        log_probabilities = F.log_softmax(logits, dim=1)
        negative_log_likelihood = -log_probabilities.gather(1, labels[:, None]).squeeze(1)
        legal_log_probabilities = log_probabilities.masked_fill(~action_mask, 0.0)
        legal_action_count = action_mask.sum(dim=1).clamp_min(1)
        legal_smoothing_loss = -legal_log_probabilities.sum(dim=1) / legal_action_count
        return (
            (1.0 - label_smoothing) * negative_log_likelihood
            + label_smoothing * legal_smoothing_loss
        ).mean()

    def policy_resource_loss(
        self,
        batch: dict[str, Tensor],
        *,
        label_smoothing: float = 0.0,
        hp_loss_coefficient: float = 1.0,
        immediate_hp_loss_coefficient: float = 1.0,
        immediate_hp_loss_positive_weight: float = 8.0,
        death_coefficient: float = 0.5,
        potion_coefficient: float = 0.25,
        max_hp_coefficient: float = 0.25,
        death_positive_weight: float = 4.0,
    ) -> dict[str, Tensor]:
        policy_logits, predictions, _ = self.policy_resource_outputs(batch)
        labels = batch["label"].long()
        action_mask = batch["action_mask"].bool()
        if label_smoothing == 0.0:
            policy = F.cross_entropy(policy_logits, labels)
        else:
            log_probabilities = F.log_softmax(policy_logits, dim=1)
            negative_log_likelihood = -log_probabilities.gather(1, labels[:, None]).squeeze(1)
            legal_log_probabilities = log_probabilities.masked_fill(~action_mask, 0.0)
            legal_action_count = action_mask.sum(dim=1).clamp_min(1)
            policy = (
                (1.0 - label_smoothing) * negative_log_likelihood
                + label_smoothing * (-legal_log_probabilities.sum(dim=1) / legal_action_count)
            ).mean()
        target_mask = batch["resource_target_mask"].bool()
        if not torch.all(target_mask):
            raise ValueError("resource-value training requires a target for every sample")

        def selected(name: str) -> Tensor:
            return predictions[name].gather(1, labels[:, None]).squeeze(1)

        hp_loss = F.smooth_l1_loss(selected("hp_loss_fraction"), batch["target_hp_loss_fraction"].float())
        immediate_target = batch["target_immediate_hp_loss_fraction"].float()
        immediate_error = F.smooth_l1_loss(
            selected("immediate_hp_loss_fraction"), immediate_target, reduction="none"
        )
        immediate_weights = torch.where(
            immediate_target.gt(0),
            torch.as_tensor(immediate_hp_loss_positive_weight, device=labels.device),
            torch.ones((), device=labels.device),
        )
        immediate_hp_loss = (immediate_error * immediate_weights).sum() / immediate_weights.sum()
        death = F.binary_cross_entropy_with_logits(
            selected("death_logit"),
            batch["target_death"].float(),
            pos_weight=torch.as_tensor(death_positive_weight, device=labels.device),
        )
        potion = F.smooth_l1_loss(selected("potion_spent"), batch["target_potion_spent"].float())
        max_hp = F.smooth_l1_loss(selected("max_hp_delta"), batch["target_max_hp_delta"].float())
        total = (
            policy
            + hp_loss_coefficient * hp_loss
            + immediate_hp_loss_coefficient * immediate_hp_loss
            + death_coefficient * death
            + potion_coefficient * potion
            + max_hp_coefficient * max_hp
        )
        return {
            "total": total,
            "policy": policy,
            "hp_loss": hp_loss,
            "immediate_hp_loss": immediate_hp_loss,
            "death": death,
            "potion": potion,
            "max_hp": max_hp,
        }

    def state_value_loss(
        self,
        batch: dict[str, Tensor],
        *,
        hp_loss_coefficient: float = 1.0,
        death_coefficient: float = 0.5,
        potion_coefficient: float = 0.25,
        max_hp_coefficient: float = 0.25,
        death_positive_weight: float = 4.0,
    ) -> dict[str, Tensor]:
        predictions = self.state_value_predictions(batch)
        target_mask = batch["resource_target_mask"].bool()
        if not torch.all(target_mask):
            raise ValueError("state-value training requires a target for every sample")
        hp_loss = F.smooth_l1_loss(
            predictions["hp_loss_fraction"], batch["target_hp_loss_fraction"].float()
        )
        death = F.binary_cross_entropy_with_logits(
            predictions["death_logit"],
            batch["target_death"].float(),
            pos_weight=torch.as_tensor(
                death_positive_weight, device=predictions["death_logit"].device
            ),
        )
        potion = F.smooth_l1_loss(
            predictions["potion_spent"], batch["target_potion_spent"].float()
        )
        max_hp = F.smooth_l1_loss(
            predictions["max_hp_delta"], batch["target_max_hp_delta"].float()
        )
        total = (
            hp_loss_coefficient * hp_loss
            + death_coefficient * death
            + potion_coefficient * potion
            + max_hp_coefficient * max_hp
        )
        return {
            "total": total,
            "hp_loss": hp_loss,
            "death": death,
            "potion": potion,
            "max_hp": max_hp,
        }

    def state_outcome_loss(
        self,
        batch: dict[str, Tensor],
        *,
        death_weight: float = 1.0,
    ) -> Tensor:
        if death_weight <= 0.0:
            raise ValueError("state outcome death weight must be positive")
        target_mask = batch["resource_target_mask"].bool()
        if not torch.all(target_mask):
            raise ValueError("state-outcome training requires a target for every sample")
        logits = self.state_outcome_logits(batch, calibrated=False)
        targets = self.state_outcome_targets(batch)
        weights = torch.ones(
            self.config.state_outcome_bins,
            device=logits.device,
            dtype=logits.dtype,
        )
        weights[0] = float(death_weight)
        return F.cross_entropy(logits, targets, weight=weights)


def numpy_batch_to_torch(
    batch: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if key in {
            "transition_id",
            "combat_id",
            "room_type",
            "label_action_type",
            "combat_difficulty_tier",
            "encounter_signature",
        }:
            result[key] = value
        else:
            result[key] = torch.as_tensor(value, device=device)
    return result
