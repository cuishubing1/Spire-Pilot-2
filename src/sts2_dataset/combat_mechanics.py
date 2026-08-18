from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .combat_directive import CombatDirectiveV0
from .combat_engine_features import candidate_preview_features
from .combat_online import visible_intent_end_turn_hp_loss


MECHANIC_GUIDANCE_VERSION = "combat-mechanic-guidance-0.1.0"


def _has_power(enemy: dict[str, Any], power_id: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in enemy.get("powers") or []
            if isinstance(row, dict) and row.get("id") == power_id
        ),
        None,
    )


def _living_enemy(observation: dict[str, Any], enemy_id: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in observation.get("enemies") or []
            if isinstance(row, dict)
            and row.get("id") == enemy_id
            and float(row.get("hp") or 0.0) > 0.0
        ),
        None,
    )


def _candidate_previews(sample: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    observation = sample["observation"]
    return [
        (candidate, candidate_preview_features(observation, candidate))
        for candidate in sample.get("candidates") or []
    ]


def _max_immediate_block(
    previews: list[tuple[dict[str, Any], dict[str, Any]]], energy: int
) -> float:
    by_source: dict[str, tuple[int, float]] = {}
    for candidate, preview in previews:
        if candidate.get("action_type") != "play_card":
            continue
        source_ref = candidate.get("source_ref")
        block = max(0.0, float(preview.get("block_gain") or 0.0))
        raw_cost = float(preview.get("energy_cost") or 0.0)
        if not source_ref or block <= 0.0 or raw_cost < 0.0 or not raw_cost.is_integer():
            continue
        cost = int(raw_cost)
        previous = by_source.get(str(source_ref))
        if previous is None or block > previous[1]:
            by_source[str(source_ref)] = (cost, block)
    best = [0.0] * (energy + 1)
    for cost, block in by_source.values():
        if cost > energy:
            continue
        for budget in range(energy, cost - 1, -1):
            best[budget] = max(best[budget], best[budget - cost] + block)
    return max(best, default=0.0)


@dataclass
class MechanicDirectiveControllerV0:
    mechanic: str
    terror_setup_rounds: int = 2
    terror_trigger_round: int | None = None
    bowlbug_committed_round: int | None = None

    def __post_init__(self) -> None:
        if self.mechanic not in {
            "bowlbug_rock_full_block",
            "terror_eel_threshold_burst",
            "overgrowth_shrinker_priority",
        }:
            raise ValueError(f"unsupported mechanic guidance: {self.mechanic}")
        if self.terror_setup_rounds < 0:
            raise ValueError("terror_setup_rounds must be non-negative")

    def directive_for(
        self, sample: dict[str, Any]
    ) -> tuple[CombatDirectiveV0, dict[str, Any]]:
        if self.mechanic == "bowlbug_rock_full_block":
            return self._bowlbug_directive(sample)
        if self.mechanic == "terror_eel_threshold_burst":
            return self._terror_eel_directive(sample)
        return self._overgrowth_directive(sample)

    def _bowlbug_directive(
        self, sample: dict[str, Any]
    ) -> tuple[CombatDirectiveV0, dict[str, Any]]:
        observation = sample["observation"]
        rock = _living_enemy(observation, "MONSTER.BOWLBUG_ROCK")
        power = _has_power(rock, "POWER.IMBALANCED_POWER") if rock else None
        rock_attacks = bool(rock and rock.get("intends_attack"))
        incoming = visible_intent_end_turn_hp_loss(observation)
        if not rock or not power or not rock_attacks or incoming is None:
            return CombatDirectiveV0.default(), {
                "guidance_version": MECHANIC_GUIDANCE_VERSION,
                "mechanic": self.mechanic,
                "phase": "inactive",
                "reason": "rock_not_attacking_or_rule_unavailable",
            }

        current_block = float((observation.get("global") or {}).get("block") or 0.0)
        incoming_damage = float(incoming["incoming_damage"])
        deficit = max(0.0, incoming_damage - current_block)
        previews = _candidate_previews(sample)
        energy = max(0, int((observation.get("global") or {}).get("energy") or 0))
        round_number = int((observation.get("global") or {}).get("round") or 0)
        if self.bowlbug_committed_round != round_number:
            self.bowlbug_committed_round = None
        maximum_immediate_block = _max_immediate_block(previews, energy)
        if deficit > maximum_immediate_block:
            self.bowlbug_committed_round = None
            return CombatDirectiveV0.default(), {
                "guidance_version": MECHANIC_GUIDANCE_VERSION,
                "mechanic": self.mechanic,
                "phase": "block_plan_infeasible",
                "enemy_ref": rock.get("entity_ref"),
                "power": power,
                "incoming_damage": incoming_damage,
                "current_block": current_block,
                "block_deficit": deficit,
                "maximum_immediate_block": maximum_immediate_block,
                "reason": "visible_hand_cannot_complete_full_block",
            }
        if deficit > 0.0 and self.bowlbug_committed_round is None:
            self.bowlbug_committed_round = round_number
        candidate_biases: dict[str, float] = {}
        forbidden: list[str] = []
        for candidate, preview in previews:
            candidate_id = str(candidate["candidate_id"])
            block_gain = max(0.0, float(preview.get("block_gain") or 0.0))
            if deficit > 0.0 and block_gain > 0.0:
                coverage = min(1.0, block_gain / deficit)
                candidate_biases[candidate_id] = 0.75 + 1.5 * coverage
                if current_block + block_gain >= incoming_damage:
                    candidate_biases[candidate_id] += 2.0
            elif deficit > 0.0 and self.bowlbug_committed_round == round_number:
                forbidden.append(candidate_id)
            elif deficit <= 0.0:
                if candidate.get("action_type") == "end_turn":
                    candidate_biases[candidate_id] = 1.5
                elif candidate.get("target_ref") == rock.get("entity_ref"):
                    candidate_biases[candidate_id] = -1.0
        directive = CombatDirectiveV0.from_dict({
            "action_preferences": {
                "forbidden_candidate_ids": forbidden,
                "candidate_biases": candidate_biases,
            },
            "replan_policy": {"top_probability_gap": 0.12},
        })
        return directive, {
            "guidance_version": MECHANIC_GUIDANCE_VERSION,
            "mechanic": self.mechanic,
            "phase": "complete_visible_block",
            "enemy_ref": rock.get("entity_ref"),
            "power": power,
            "incoming_damage": incoming_damage,
            "current_block": current_block,
            "block_deficit": deficit,
            "maximum_immediate_block": maximum_immediate_block,
            "committed_round": self.bowlbug_committed_round,
            "forbidden_non_block_candidates": len(forbidden),
            "biased_candidate_count": len(candidate_biases),
        }

    def _terror_eel_directive(
        self, sample: dict[str, Any]
    ) -> tuple[CombatDirectiveV0, dict[str, Any]]:
        observation = sample["observation"]
        eel = _living_enemy(observation, "MONSTER.TERROR_EEL")
        if not eel:
            return CombatDirectiveV0.default(), {
                "guidance_version": MECHANIC_GUIDANCE_VERSION,
                "mechanic": self.mechanic,
                "phase": "inactive",
                "reason": "terror_eel_not_living",
            }
        power = _has_power(eel, "POWER.SHRIEK_POWER")
        threshold = float((power or {}).get("amount") or (float(eel.get("max_hp") or 0.0) * 0.5))
        hp = float(eel.get("hp") or 0.0)
        round_number = int((observation.get("global") or {}).get("round") or 0)
        if (hp <= threshold or power is None) and self.terror_trigger_round is None:
            self.terror_trigger_round = round_number
        setup_ready = round_number > self.terror_setup_rounds
        triggered = self.terror_trigger_round is not None
        candidate_biases: dict[str, float] = {}
        forbidden: list[str] = []
        for candidate, preview in _candidate_previews(sample):
            candidate_id = str(candidate["candidate_id"])
            damage = max(0.0, float(preview.get("total_damage") or 0.0))
            block = max(0.0, float(preview.get("block_gain") or 0.0))
            if not triggered and not setup_ready:
                if damage > 0.0 and hp - damage <= threshold:
                    forbidden.append(candidate_id)
                elif damage > 0.0:
                    candidate_biases[candidate_id] = -0.35
                elif block > 0.0:
                    candidate_biases[candidate_id] = 0.35
            else:
                if damage > 0.0:
                    candidate_biases[candidate_id] = 1.0 + min(1.5, damage / max(1.0, hp) * 4.0)
                elif candidate.get("action_type") == "end_turn":
                    candidate_biases[candidate_id] = -1.0
                else:
                    candidate_biases[candidate_id] = -0.2
        phase = (
            "post_threshold_burst"
            if triggered
            else "cross_threshold_now"
            if setup_ready
            else "pre_threshold_setup"
        )
        directive = CombatDirectiveV0.from_dict({
            "action_preferences": {
                "forbidden_candidate_ids": forbidden,
                "candidate_biases": candidate_biases,
            },
            "replan_policy": {"top_probability_gap": 0.12},
        })
        return directive, {
            "guidance_version": MECHANIC_GUIDANCE_VERSION,
            "mechanic": self.mechanic,
            "phase": phase,
            "enemy_ref": eel.get("entity_ref"),
            "power": power,
            "threshold_hp": threshold,
            "enemy_hp": hp,
            "round": round_number,
            "setup_rounds": self.terror_setup_rounds,
            "trigger_round": self.terror_trigger_round,
            "forbidden_threshold_crossings": len(forbidden),
            "biased_candidate_count": len(candidate_biases),
        }

    def _overgrowth_directive(
        self, sample: dict[str, Any]
    ) -> tuple[CombatDirectiveV0, dict[str, Any]]:
        observation = sample["observation"]
        shrinker = _living_enemy(observation, "MONSTER.SHRINKER_BEETLE")
        fuzzy = _living_enemy(observation, "MONSTER.FUZZY_WURM_CRAWLER")
        if not shrinker or not fuzzy:
            return CombatDirectiveV0.default(), {
                "guidance_version": MECHANIC_GUIDANCE_VERSION,
                "mechanic": self.mechanic,
                "phase": "inactive",
                "reason": "priority_pair_not_both_living",
            }
        target_biases = {
            str(shrinker["entity_ref"]): 1.75,
            str(fuzzy["entity_ref"]): -0.5,
        }
        directive = CombatDirectiveV0.from_dict({
            "action_preferences": {"target_biases": target_biases},
            "replan_policy": {"top_probability_gap": 0.12},
        })
        return directive, {
            "guidance_version": MECHANIC_GUIDANCE_VERSION,
            "mechanic": self.mechanic,
            "phase": "focus_shrinker",
            "priority_target_ref": shrinker.get("entity_ref"),
            "priority_target_hp": shrinker.get("hp"),
            "secondary_target_ref": fuzzy.get("entity_ref"),
        }
