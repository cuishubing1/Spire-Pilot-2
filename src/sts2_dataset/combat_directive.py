from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .combat_model import CombatObjective, CombatPolicyConfig


COMBAT_DIRECTIVE_VERSION = "combat-directive-0.2.0"
LEGACY_COMBAT_DIRECTIVE_VERSIONS = {"combat-directive-0.1.0"}
SUPPORTED_ACTION_TYPES = {
    "play_card",
    "use_potion",
    "discard_potion",
    "end_turn",
}
SUPPORTED_SEARCH_MODES = {"policy_only", "one_step", "turn_boundary"}
SUPPORTED_SEARCH_BUDGET_CLASSES = {"low", "medium", "high"}


def _finite_number(name: str, value: Any, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(row, str) or not row for row in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(value)


def _bias_map(name: str, value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        score = _finite_number(f"{name}.{key}", raw)
        if abs(score) > 20.0:
            raise ValueError(f"{name}.{key} magnitude must be <= 20")
        result[key] = score
    return result


@dataclass(frozen=True)
class CombatSearchPolicyV0:
    """Bounded search request controlled by the upper-level agent.

    The upper layer selects a semantic mode and budget class.  Concrete beam
    widths or MCTS simulations remain an implementation detail of the combat
    executor so an LLM cannot request an unbounded search.
    """

    mode: str = "policy_only"
    budget_class: str = "low"
    max_wall_ms: int = 750
    determinizations: int = 2
    allow_policy_override: bool = True
    mechanic_plan_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CombatSearchPolicyV0":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("search_policy must be an object")
        allowed = {
            "mode",
            "budget_class",
            "max_wall_ms",
            "determinizations",
            "allow_policy_override",
            "mechanic_plan_id",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown search policy fields: {sorted(unknown)}")
        mode = str(value.get("mode", "policy_only"))
        if mode not in SUPPORTED_SEARCH_MODES:
            raise ValueError(f"unsupported search mode: {mode!r}")
        budget_class = str(value.get("budget_class", "low"))
        if budget_class not in SUPPORTED_SEARCH_BUDGET_CLASSES:
            raise ValueError(f"unsupported search budget class: {budget_class!r}")
        max_wall_ms = value.get("max_wall_ms", 750)
        if (
            not isinstance(max_wall_ms, int)
            or isinstance(max_wall_ms, bool)
            or not 50 <= max_wall_ms <= 10_000
        ):
            raise ValueError("search_policy.max_wall_ms must be an integer in [50, 10000]")
        determinizations = value.get("determinizations", 2)
        if (
            not isinstance(determinizations, int)
            or isinstance(determinizations, bool)
            or not 1 <= determinizations <= 8
        ):
            raise ValueError("search_policy.determinizations must be an integer in [1, 8]")
        allow_policy_override = value.get("allow_policy_override", True)
        if not isinstance(allow_policy_override, bool):
            raise ValueError("search_policy.allow_policy_override must be boolean")
        mechanic_plan_id = value.get("mechanic_plan_id")
        if mechanic_plan_id is not None and (
            not isinstance(mechanic_plan_id, str) or not mechanic_plan_id
        ):
            raise ValueError("search_policy.mechanic_plan_id must be null or a non-empty string")
        return cls(
            mode=mode,
            budget_class=budget_class,
            max_wall_ms=max_wall_ms,
            determinizations=determinizations,
            allow_policy_override=allow_policy_override,
            mechanic_plan_id=mechanic_plan_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "budget_class": self.budget_class,
            "max_wall_ms": self.max_wall_ms,
            "determinizations": self.determinizations,
            "allow_policy_override": self.allow_policy_override,
            "mechanic_plan_id": self.mechanic_plan_id,
        }


@dataclass(frozen=True)
class CombatDirectiveV0:
    """Typed upper-to-lower combat objective.

    The directive contains goals and preferences only. Enemy mechanics remain
    a separate trusted rules-engine input to Combat Tool V0.
    """

    objective_overrides: dict[str, float] = field(default_factory=dict)
    max_potion_uses: int | None = None
    potion_uses_so_far: int = 0
    preserve_potion_ids: tuple[str, ...] = ()
    forbidden_action_types: tuple[str, ...] = ()
    forbidden_candidate_ids: tuple[str, ...] = ()
    action_type_biases: dict[str, float] = field(default_factory=dict)
    target_biases: dict[str, float] = field(default_factory=dict)
    candidate_biases: dict[str, float] = field(default_factory=dict)
    acceptable_hp_loss_fraction: float | None = None
    replan_death_probability: float = 0.20
    replan_normalized_entropy: float = 0.80
    replan_top_probability_gap: float = 0.08
    search_policy: CombatSearchPolicyV0 = field(default_factory=CombatSearchPolicyV0)
    scope: str = "current_combat"
    expires_at: str = "combat_end"

    @classmethod
    def default(cls) -> "CombatDirectiveV0":
        return cls()

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CombatDirectiveV0":
        if value is None:
            return cls.default()
        if not isinstance(value, dict):
            raise ValueError("combat directive must be an object")
        version = value.get("schema_version", COMBAT_DIRECTIVE_VERSION)
        if version != COMBAT_DIRECTIVE_VERSION and version not in LEGACY_COMBAT_DIRECTIVE_VERSIONS:
            raise ValueError(f"unsupported combat directive version: {version!r}")

        objective = value.get("objective") or {}
        resource = value.get("resource_policy") or {}
        preferences = value.get("action_preferences") or {}
        replan = value.get("replan_policy") or {}
        for name, row in (
            ("objective", objective),
            ("resource_policy", resource),
            ("action_preferences", preferences),
            ("replan_policy", replan),
        ):
            if not isinstance(row, dict):
                raise ValueError(f"{name} must be an object")

        objective_names = {
            "decision_value_scale",
            "hp_loss_weight",
            "immediate_hp_loss_weight",
            "death_penalty",
            "potion_cost",
            "max_hp_gain_weight",
        }
        unknown_objectives = set(objective) - objective_names
        if unknown_objectives:
            raise ValueError(f"unknown combat objective fields: {sorted(unknown_objectives)}")
        objective_overrides = {
            name: _finite_number(f"objective.{name}", raw, minimum=0.0)
            for name, raw in objective.items()
            if raw is not None
        }

        max_potion_uses = resource.get("max_potion_uses")
        if max_potion_uses is not None:
            if not isinstance(max_potion_uses, int) or isinstance(max_potion_uses, bool) or max_potion_uses < 0:
                raise ValueError("resource_policy.max_potion_uses must be a non-negative integer")
        potion_uses_so_far = resource.get("potion_uses_so_far", 0)
        if not isinstance(potion_uses_so_far, int) or isinstance(potion_uses_so_far, bool) or potion_uses_so_far < 0:
            raise ValueError("resource_policy.potion_uses_so_far must be a non-negative integer")

        forbidden_action_types = _string_tuple(
            "action_preferences.forbidden_action_types",
            preferences.get("forbidden_action_types"),
        )
        unknown_actions = set(forbidden_action_types) - SUPPORTED_ACTION_TYPES
        if unknown_actions:
            raise ValueError(f"unsupported forbidden action types: {sorted(unknown_actions)}")
        action_type_biases = _bias_map(
            "action_preferences.action_type_biases",
            preferences.get("action_type_biases"),
        )
        unknown_actions = set(action_type_biases) - SUPPORTED_ACTION_TYPES
        if unknown_actions:
            raise ValueError(f"unsupported biased action types: {sorted(unknown_actions)}")

        acceptable = resource.get("acceptable_hp_loss_fraction")
        if acceptable is not None:
            acceptable = _finite_number(
                "resource_policy.acceptable_hp_loss_fraction", acceptable, minimum=0.0
            )
            if acceptable > 1.0:
                raise ValueError("acceptable_hp_loss_fraction must be <= 1")

        death_threshold = _finite_number(
            "replan_policy.death_probability",
            replan.get("death_probability", 0.20),
            minimum=0.0,
        )
        entropy_threshold = _finite_number(
            "replan_policy.normalized_entropy",
            replan.get("normalized_entropy", 0.80),
            minimum=0.0,
        )
        gap_threshold = _finite_number(
            "replan_policy.top_probability_gap",
            replan.get("top_probability_gap", 0.08),
            minimum=0.0,
        )
        if death_threshold > 1.0 or entropy_threshold > 1.0 or gap_threshold > 1.0:
            raise ValueError("replan thresholds must be in [0, 1]")

        scope = str(value.get("scope", "current_combat"))
        expires_at = str(value.get("expires_at", "combat_end"))
        if scope != "current_combat" or expires_at != "combat_end":
            raise ValueError("Combat Directive V0 only supports current_combat/combat_end scope")
        return cls(
            objective_overrides=objective_overrides,
            max_potion_uses=max_potion_uses,
            potion_uses_so_far=potion_uses_so_far,
            preserve_potion_ids=_string_tuple(
                "resource_policy.preserve_potion_ids", resource.get("preserve_potion_ids")
            ),
            forbidden_action_types=forbidden_action_types,
            forbidden_candidate_ids=_string_tuple(
                "action_preferences.forbidden_candidate_ids",
                preferences.get("forbidden_candidate_ids"),
            ),
            action_type_biases=action_type_biases,
            target_biases=_bias_map(
                "action_preferences.target_biases", preferences.get("target_biases")
            ),
            candidate_biases=_bias_map(
                "action_preferences.candidate_biases", preferences.get("candidate_biases")
            ),
            acceptable_hp_loss_fraction=acceptable,
            replan_death_probability=death_threshold,
            replan_normalized_entropy=entropy_threshold,
            replan_top_probability_gap=gap_threshold,
            search_policy=CombatSearchPolicyV0.from_dict(value.get("search_policy")),
            scope=scope,
            expires_at=expires_at,
        )

    def objective(self, config: CombatPolicyConfig) -> CombatObjective:
        return CombatObjective.from_config(config, **self.objective_overrides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMBAT_DIRECTIVE_VERSION,
            "scope": self.scope,
            "expires_at": self.expires_at,
            "objective": dict(sorted(self.objective_overrides.items())),
            "resource_policy": {
                "max_potion_uses": self.max_potion_uses,
                "potion_uses_so_far": self.potion_uses_so_far,
                "preserve_potion_ids": list(self.preserve_potion_ids),
                "acceptable_hp_loss_fraction": self.acceptable_hp_loss_fraction,
            },
            "action_preferences": {
                "forbidden_action_types": list(self.forbidden_action_types),
                "forbidden_candidate_ids": list(self.forbidden_candidate_ids),
                "action_type_biases": dict(sorted(self.action_type_biases.items())),
                "target_biases": dict(sorted(self.target_biases.items())),
                "candidate_biases": dict(sorted(self.candidate_biases.items())),
            },
            "replan_policy": {
                "death_probability": self.replan_death_probability,
                "normalized_entropy": self.replan_normalized_entropy,
                "top_probability_gap": self.replan_top_probability_gap,
            },
            "search_policy": self.search_policy.to_dict(),
        }


@dataclass(frozen=True)
class CandidateMechanicFactV0:
    """Trusted candidate-level fact emitted by a deterministic rule adapter."""

    candidate_id: str
    mechanic_id: str
    score_delta: float = 0.0
    hard_forbidden: bool = False
    source: str = "engine_rule"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateMechanicFactV0":
        if not isinstance(value, dict):
            raise ValueError("candidate mechanic fact must be an object")
        candidate_id = value.get("candidate_id")
        mechanic_id = value.get("mechanic_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate mechanic fact requires candidate_id")
        if not isinstance(mechanic_id, str) or not mechanic_id:
            raise ValueError("candidate mechanic fact requires mechanic_id")
        source = value.get("source", "engine_rule")
        if source != "engine_rule":
            raise ValueError("candidate mechanic facts must come from engine_rule")
        score_delta = _finite_number("candidate mechanic score_delta", value.get("score_delta", 0.0))
        if abs(score_delta) > 20.0:
            raise ValueError("candidate mechanic score_delta magnitude must be <= 20")
        hard_forbidden = value.get("hard_forbidden", False)
        if not isinstance(hard_forbidden, bool):
            raise ValueError("candidate mechanic hard_forbidden must be boolean")
        return cls(
            candidate_id=candidate_id,
            mechanic_id=mechanic_id,
            score_delta=score_delta,
            hard_forbidden=hard_forbidden,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mechanic_id": self.mechanic_id,
            "score_delta": self.score_delta,
            "hard_forbidden": self.hard_forbidden,
            "source": self.source,
        }
