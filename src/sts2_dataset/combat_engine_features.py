from __future__ import annotations

import math
from typing import Any

from .combat_online import visible_intent_end_turn_hp_loss


ENGINE_FEATURE_VERSION = "combat-engine-features-0.1.0"
CANDIDATE_ENGINE_FEATURE_VERSION = "candidate-engine-features-0.1.0"
CANDIDATE_ENGINE_FEATURE_NAMES = (
    "energy_fraction",
    "energy_cost_fraction",
    "costs_x",
    "block_gain_fraction",
    "block_after_fraction",
    "damage_per_hit_target_fraction",
    "hit_count_log",
    "total_damage_target_fraction",
    "target_hp_fraction",
    "target_block_fraction",
    "target_remaining_fraction",
    "preview_lethal",
    "visible_incoming_damage_fraction",
    "visible_end_turn_hp_loss_fraction",
    "block_adjusted_end_turn_hp_loss_fraction",
    "uses_potion",
    "discards_potion",
    "ends_turn",
)
CANDIDATE_ENGINE_FEATURE_DIM = len(CANDIDATE_ENGINE_FEATURE_NAMES)
COMBAT_END_MAX_HP_RELICS = {"RELIC.CHOSEN_CHEESE"}


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _find_entity(rows: list[dict[str, Any]], reference: Any) -> dict[str, Any] | None:
    wanted = str(reference)
    return next((row for row in rows if str(row.get("entity_ref")) == wanted), None)


def _nested_number(value: Any, *paths: tuple[str, ...]) -> float:
    for path in paths:
        current = value
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)):
            return float(current)
    return 0.0


def combat_future_max_hp_growth_cap(
    observation: dict[str, Any],
) -> dict[str, float | list[dict[str, float | str]]]:
    """Return a public-rule upper bound for remaining in-combat Max HP gain.

    Card growth is discovered from the engine-exported ``maxhp`` dynamic var
    in playable piles. Relics are intentionally allow-listed because static
    pickup relics such as Strawberry also expose a ``maxhp`` var even though
    their gain was already resolved before combat.
    """

    sources: list[dict[str, float | str]] = []
    piles = observation.get("piles") or {}
    zones = {
        "hand": observation.get("hand") or [],
        "draw": piles.get("draw") or [],
        "discard": piles.get("discard") or [],
    }
    for zone, cards in zones.items():
        for card in cards:
            if not isinstance(card, dict):
                continue
            amount = _nested_number(
                card,
                ("stats", "maxhp"),
                ("dynamic_vars", "preview", "maxhp"),
                ("dynamic_vars", "effective", "maxhp"),
            )
            if amount <= 0.0:
                continue
            count = max(1.0, _number(card.get("count"), 1.0))
            sources.append({
                "kind": "card",
                "id": str(card.get("id") or ""),
                "zone": zone,
                "amount": amount * count,
            })
    for relic in observation.get("relics") or []:
        if not isinstance(relic, dict) or relic.get("id") not in COMBAT_END_MAX_HP_RELICS:
            continue
        visible = relic.get("visible_state") or {}
        if visible.get("is_used_up") or visible.get("is_melted"):
            continue
        amount = _nested_number(relic, ("dynamic_vars", "maxhp"))
        if amount > 0.0:
            sources.append({
                "kind": "relic",
                "id": str(relic.get("id")),
                "zone": "relic",
                "amount": amount,
            })
    loss_sources: list[dict[str, float | str]] = []
    for entity in [
        *(observation.get("hand") or []),
        *(piles.get("draw") or []),
        *(piles.get("discard") or []),
        *(observation.get("relics") or []),
    ]:
        if not isinstance(entity, dict):
            continue
        amount = _nested_number(
            entity,
            ("stats", "maxhploss"),
            ("dynamic_vars", "maxhploss"),
            ("dynamic_vars", "preview", "maxhploss"),
            ("dynamic_vars", "effective", "maxhploss"),
        )
        if amount > 0.0:
            loss_sources.append({
                "kind": "max_hp_loss",
                "id": str(entity.get("id") or ""),
                "zone": "visible_entity",
                "amount": amount,
            })
    return {
        "positive_growth_cap": sum(float(row["amount"]) for row in sources),
        "negative_loss_cap": sum(float(row["amount"]) for row in loss_sources),
        "sources": sources,
        "loss_sources": loss_sources,
    }


def ground_future_max_hp_delta(
    prediction: float, observation: dict[str, Any]
) -> dict[str, float | str | list[dict[str, float | str]]]:
    """Clamp learned positive growth to what public engine facts permit."""

    capability = combat_future_max_hp_growth_cap(observation)
    cap = float(capability["positive_growth_cap"])
    loss_cap = float(capability["negative_loss_cap"])
    raw = float(prediction)
    grounded = max(-loss_cap, min(raw, cap))
    return {
        "raw_prediction": raw,
        "grounded_prediction": grounded,
        "positive_growth_cap": cap,
        "negative_loss_cap": loss_cap,
        "source": "engine_public_growth_cap",
        "sources": capability["sources"],
        "loss_sources": capability["loss_sources"],
    }


def candidate_preview_features(
    observation: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | bool | str]:
    """Return deterministic, player-visible action preview features.

    These are engine-exported facts available before executing the action. They
    are not learned transition targets and never inspect audit-only pile order.
    """
    global_state = observation.get("global") or {}
    action_type = str(candidate.get("action_type") or "")
    source = _find_entity(observation.get("hand") or [], candidate.get("source_ref"))
    target = _find_entity(observation.get("enemies") or [], candidate.get("target_ref"))
    features: dict[str, float | bool | str] = {
        "feature_version": ENGINE_FEATURE_VERSION,
        "source": "engine_public_preview",
        "energy_before": _number(global_state.get("energy")),
        "energy_cost": 0.0,
        "costs_x": False,
        "block_gain": 0.0,
        "damage_per_hit": 0.0,
        "hit_count": 0.0,
        "total_damage": 0.0,
        "target_hp_before": _number((target or {}).get("hp")),
        "target_block_before": _number((target or {}).get("block")),
        "preview_lethal": False,
        "uses_potion": action_type == "use_potion",
        "discards_potion": action_type == "discard_potion",
        "ends_turn": action_type == "end_turn",
        "visible_end_turn_hp_loss": 0.0,
        "visible_end_turn_hp_loss_fraction": 0.0,
    }
    if source is not None:
        energy_cost = source.get("energy_cost") or {}
        features["energy_cost"] = _number(
            energy_cost.get("current", source.get("cost"))
            if isinstance(energy_cost, dict) else source.get("cost")
        )
        features["costs_x"] = bool(
            energy_cost.get("costs_x") if isinstance(energy_cost, dict) else False
        )
        stats = source.get("stats") if isinstance(source.get("stats"), dict) else {}
        features["block_gain"] = _number(stats.get("block"))
        if target is not None:
            target_ref = str(target.get("entity_ref"))
            damage_row = next(
                (
                    row for row in source.get("damage_by_target") or []
                    if isinstance(row, dict)
                    and str(row.get("target_combat_id")) == target_ref
                ),
                None,
            )
            if damage_row is None:
                target_index = target.get("index")
                damage_row = next(
                    (
                        row for row in source.get("damage_by_target") or []
                        if isinstance(row, dict) and row.get("target_index") == target_index
                    ),
                    {},
                )
            per_hit = _number(damage_row.get("damage", stats.get("damage")))
            hits = _number(damage_row.get("hits", damage_row.get("repeats", 1)), 1.0)
            total = _number(damage_row.get("total_damage"), per_hit * hits)
            features["damage_per_hit"] = per_hit
            features["hit_count"] = hits
            features["total_damage"] = total
            features["preview_lethal"] = total >= (
                _number(target.get("hp")) + _number(target.get("block"))
            )
    if action_type == "end_turn":
        estimate = visible_intent_end_turn_hp_loss(observation)
        if estimate is not None:
            features["visible_end_turn_hp_loss"] = estimate["hp_loss"]
            features["visible_end_turn_hp_loss_fraction"] = estimate["hp_loss_fraction"]
    return features


def candidate_engine_feature_vector(
    observation: dict[str, Any], candidate: dict[str, Any]
) -> list[float]:
    """Encode explicit public engine previews for one legal candidate.

    The values intentionally have stable, named positions rather than a hash.
    Damage and block are normalized by the relevant public maximum HP so the
    same feature scale transfers across Acts and ascension levels.  The
    block-adjusted loss is exact only for the block already present plus the
    engine-exported immediate block on this candidate; it does not speculate
    about chained powers, future draws, or hidden RNG.
    """

    preview = candidate_preview_features(observation, candidate)
    global_state = observation.get("global") or {}
    player_max_hp = max(1.0, _number(global_state.get("max_hp"), 1.0))
    max_energy = max(1.0, _number(global_state.get("max_energy"), 1.0))
    target_max_hp = 1.0
    target = _find_entity(observation.get("enemies") or [], candidate.get("target_ref"))
    if target is not None:
        target_max_hp = max(1.0, _number(target.get("max_hp"), 1.0))

    visible = visible_intent_end_turn_hp_loss(observation)
    incoming = float(visible["incoming_damage"]) if visible is not None else 0.0
    current_block = max(0.0, _number(global_state.get("block")))
    block_gain = max(0.0, float(preview["block_gain"]))
    block_adjusted_loss = max(0.0, incoming - current_block - block_gain)
    target_hp = max(0.0, float(preview["target_hp_before"]))
    target_block = max(0.0, float(preview["target_block_before"]))
    total_damage = max(0.0, float(preview["total_damage"]))
    target_remaining = max(0.0, target_hp + target_block - total_damage)

    values = {
        "energy_fraction": max(0.0, float(preview["energy_before"])) / max_energy,
        "energy_cost_fraction": max(0.0, float(preview["energy_cost"])) / max_energy,
        "costs_x": float(bool(preview["costs_x"])),
        "block_gain_fraction": block_gain / player_max_hp,
        "block_after_fraction": (current_block + block_gain) / player_max_hp,
        "damage_per_hit_target_fraction": max(
            0.0, float(preview["damage_per_hit"])
        ) / target_max_hp,
        "hit_count_log": math.log1p(max(0.0, float(preview["hit_count"]))),
        "total_damage_target_fraction": total_damage / target_max_hp,
        "target_hp_fraction": target_hp / target_max_hp if target is not None else 0.0,
        "target_block_fraction": target_block / target_max_hp if target is not None else 0.0,
        "target_remaining_fraction": target_remaining / target_max_hp
        if target is not None else 0.0,
        "preview_lethal": float(bool(preview["preview_lethal"])),
        "visible_incoming_damage_fraction": incoming / player_max_hp,
        "visible_end_turn_hp_loss_fraction": float(visible["hp_loss_fraction"])
        if visible is not None else 0.0,
        "block_adjusted_end_turn_hp_loss_fraction": block_adjusted_loss / player_max_hp,
        "uses_potion": float(bool(preview["uses_potion"])),
        "discards_potion": float(bool(preview["discards_potion"])),
        "ends_turn": float(bool(preview["ends_turn"])),
    }
    if tuple(values) != CANDIDATE_ENGINE_FEATURE_NAMES:
        raise AssertionError("candidate engine feature order changed")
    return [float(values[name]) for name in CANDIDATE_ENGINE_FEATURE_NAMES]


def exact_transition_features(
    before: dict[str, Any], after: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | bool | str]:
    """Summarize one action using actual engine successor states.

    This function consumes the result of a branch execution. It intentionally
    does not predict effects from card text. Complex powers, relics and chained
    effects are therefore reflected in the measured deltas.
    """
    before_player = before.get("player") or {}
    after_player = after.get("player") or {}
    before_enemies = before.get("enemies") or []
    after_enemies = after.get("enemies") or []
    decision_after = str(after.get("decision") or "")
    terminal = decision_after == "game_over"
    victory = bool(after.get("victory")) if terminal else False
    enemy_state_observed = (
        decision_after == "combat_play"
        or bool(after_enemies)
        or (terminal and victory)
    )
    block_observed = isinstance(after_player.get("block"), (int, float))
    energy_observed = isinstance(after.get("energy"), (int, float))
    hand_observed = isinstance(after.get("hand"), list)
    round_observed = isinstance(after.get("round"), (int, float))
    before_enemy_hp = sum(_number(row.get("hp")) for row in before_enemies if isinstance(row, dict))
    after_enemy_hp = (
        sum(_number(row.get("hp")) for row in after_enemies if isinstance(row, dict))
        if enemy_state_observed else before_enemy_hp
    )
    before_alive = sum(
        _number(row.get("hp")) > 0 for row in before_enemies if isinstance(row, dict)
    )
    after_alive = (
        sum(_number(row.get("hp")) > 0 for row in after_enemies if isinstance(row, dict))
        if enemy_state_observed else before_alive
    )
    hp_before = _number(before_player.get("hp"))
    hp_after = _number(after_player.get("hp"), hp_before)
    max_hp_before = _number(before_player.get("max_hp"))
    max_hp_after = _number(after_player.get("max_hp"), max_hp_before)
    return {
        "feature_version": ENGINE_FEATURE_VERSION,
        "source": "engine_executed_transition",
        "action_type": str(candidate.get("action_type") or ""),
        "enemy_state_observed": enemy_state_observed,
        "block_observed": block_observed,
        "energy_observed": energy_observed,
        "hand_observed": hand_observed,
        "round_observed": round_observed,
        "hp_loss": max(0.0, hp_before - hp_after),
        "hp_delta": hp_after - hp_before,
        "max_hp_delta": max_hp_after - max_hp_before,
        "block_delta": (
            _number(after_player.get("block")) - _number(before_player.get("block"))
            if block_observed else 0.0
        ),
        "energy_delta": (
            _number(after.get("energy")) - _number(before.get("energy"))
            if energy_observed else 0.0
        ),
        "enemy_hp_loss": max(0.0, before_enemy_hp - after_enemy_hp),
        "enemies_killed": float(max(0, before_alive - after_alive)),
        "potion_count_delta": float(len(after_player.get("potions") or []) - len(before_player.get("potions") or [])),
        "round_delta": (
            _number(after.get("round")) - _number(before.get("round"))
            if round_observed else 0.0
        ),
        "hand_count_delta": (
            float(len(after.get("hand") or []) - len(before.get("hand") or []))
            if hand_observed else 0.0
        ),
        "combat_continues": decision_after == "combat_play",
        "terminal": terminal,
        "victory": victory,
    }
