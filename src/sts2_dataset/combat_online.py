from __future__ import annotations

from typing import Any

from .combat_contract import ACTION_VERSION, OBSERVATION_VERSION, build_action_candidates_v0, project_observation_v0
from .combat_encounter import encounter_signature_from_enemies
from .human import HumanRecordingError
from .util import canonical_json


def _search_card_category(card: dict[str, Any]) -> str:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    effective = (
        (card.get("dynamic_vars") or {}).get("effective")
        if isinstance(card.get("dynamic_vars"), dict)
        else {}
    )
    effective = effective if isinstance(effective, dict) else {}
    if float(stats.get("block", effective.get("block", 0)) or 0) > 0:
        return "card_block"
    if str(card.get("type") or "").lower() == "attack" or card.get("damage_by_target"):
        return "card_attack"
    if str(card.get("type") or "").lower() == "power":
        return "card_power"
    return "card_skill"


def _attach_search_metadata(
    candidates: list[dict[str, Any]], observation: dict[str, Any]
) -> None:
    """Add instance-agnostic branch keys without changing executable actions."""

    hand = {
        str(card.get("instance_id")): card
        for card in (observation.get("combat") or {}).get("hand") or []
        if isinstance(card, dict) and card.get("instance_id")
    }
    potions = {
        str(potion.get("instance_id")): potion
        for potion in (observation.get("player") or {}).get("potions") or []
        if isinstance(potion, dict) and potion.get("instance_id")
    }
    for candidate in candidates:
        action_type = str(candidate.get("action_type") or "")
        source_ref = str(candidate.get("source_ref") or "")
        if action_type == "play_card" and source_ref in hand:
            card = hand[source_ref]
            source_signature: dict[str, Any] | None = {
                "kind": "card",
                "id": card.get("id"),
                "upgrade_level": card.get("upgrade_level"),
                "cost": card.get("cost"),
                "energy_cost": card.get("energy_cost"),
                "enchantment": card.get("enchantment"),
                "affliction": card.get("affliction"),
                "stats": card.get("stats"),
                "effective_vars": (card.get("dynamic_vars") or {}).get("effective"),
                "runtime_flags": card.get("runtime_flags"),
                "runtime_state": card.get("runtime_state"),
                "persistent_state": card.get("persistent_state"),
                "target_type": card.get("target_type"),
            }
            category = _search_card_category(card)
        elif action_type in {"use_potion", "discard_potion"} and source_ref in potions:
            potion = potions[source_ref]
            source_signature = {
                "kind": "potion",
                "id": potion.get("id"),
                "target_type": potion.get("target_type"),
                "vars": potion.get("vars"),
            }
            category = "potion_use" if action_type == "use_potion" else "potion_discard"
        else:
            source_signature = None
            category = action_type or "other"
        candidate["search_category"] = category
        candidate["search_equivalence_key"] = canonical_json({
            "action_type": action_type,
            "source": source_signature,
            "target_kind": candidate.get("target_kind"),
            "target_ref": candidate.get("target_ref"),
        })


def visible_intent_end_turn_hp_loss(observation: dict[str, Any]) -> dict[str, float] | None:
    """Estimate immediate end-turn HP loss from player-visible attack intents.

    This deliberately uses only the public observation contract. If an attack
    is visible but its damage is unavailable, the estimate is unavailable
    instead of silently treating unknown damage as zero.
    """
    global_state = observation.get("global") or {}
    max_hp = global_state.get("max_hp")
    block = global_state.get("block")
    if not isinstance(max_hp, (int, float)) or float(max_hp) <= 0:
        return None
    if not isinstance(block, (int, float)):
        return None

    total_incoming_damage = 0.0
    saw_attack = False
    for enemy in observation.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        enemy_hp = enemy.get("hp")
        if isinstance(enemy_hp, (int, float)) and float(enemy_hp) <= 0:
            continue
        for intent in enemy.get("intent") or []:
            if not isinstance(intent, dict):
                continue
            damage = intent.get("damage")
            is_attack = bool(intent.get("is_attack")) or damage is not None
            if not is_attack:
                continue
            saw_attack = True
            total_damage = intent.get("total_damage")
            if isinstance(total_damage, (int, float)):
                total_incoming_damage += max(0.0, float(total_damage))
                continue
            if not isinstance(damage, (int, float)):
                return None
            repetitions = intent.get("hits", intent.get("repeats", 1))
            if not isinstance(repetitions, (int, float)):
                return None
            total_incoming_damage += max(0.0, float(damage)) * max(0.0, float(repetitions))

    if not saw_attack:
        total_incoming_damage = 0.0
    hp_loss = max(0.0, total_incoming_damage - max(0.0, float(block)))
    return {
        "incoming_damage": total_incoming_damage,
        "block": max(0.0, float(block)),
        "hp_loss": hp_loss,
        "hp_loss_fraction": hp_loss / float(max_hp),
    }


def _card_for_recorder(card: dict[str, Any], *, instance_id: str | None = None) -> dict[str, Any]:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    result = {
        "id": card.get("id"),
        "index": card.get("index"),
        "count": card.get("count"),
        "instance_id": instance_id,
        "lineage_id": None,
        "upgrade_level": int(card.get("upgrade_level") or bool(card.get("upgraded"))),
        "max_upgrade_level": 1,
        "type": card.get("type"),
        "rarity": card.get("rarity"),
        "target_type": card.get("target_type"),
        "tags": card.get("tags") or [],
        "keywords": card.get("keywords") or [],
        "cost": card.get("cost"),
        "can_play": card.get("can_play"),
        "stats": stats or None,
        "dynamic_vars": {"preview": stats, "effective": stats},
        "persistent_state": [],
        "runtime_state": [],
    }
    damage_rows = []
    for damage in card.get("damage_by_target") or []:
        if not isinstance(damage, dict):
            continue
        target_index = damage.get("target_index")
        damage_rows.append({
            "target_index": target_index,
            "target_combat_id": f"headless:{target_index}" if target_index is not None else None,
            "damage": damage.get("damage"),
            "hits": damage.get("hits", damage.get("repeat", 1)),
            "total_damage": damage.get("total_damage", damage.get("damage")),
        })
    if damage_rows:
        result["damage_by_target"] = damage_rows
    return result


def _legal_actions(state: dict[str, Any], recorder_observation: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    enemies = recorder_observation["combat"]["enemies"]
    for card in recorder_observation["combat"]["hand"]:
        if not card.get("can_play"):
            continue
        target_type = str(card.get("target_type") or "").lower()
        if target_type == "anyenemy":
            for enemy in enemies:
                actions.append({
                    "action_id": "play_card",
                    "args": {
                        "card_instance_id": card["instance_id"],
                        "target_combat_id": enemy["combat_id"],
                    },
                })
        else:
            actions.append({
                "action_id": "play_card",
                "args": {"card_instance_id": card["instance_id"]},
            })
    for potion in recorder_observation["player"]["potions"]:
        target_type = str(potion.get("target_type") or "").lower()
        if target_type == "anyenemy":
            for enemy in enemies:
                actions.append({
                    "action_id": "use_potion",
                    "args": {
                        "potion_instance_id": potion["instance_id"],
                        "target_combat_id": enemy["combat_id"],
                    },
                })
        else:
            actions.append({
                "action_id": "use_potion",
                "args": {"potion_instance_id": potion["instance_id"]},
            })
        actions.append({
            "action_id": "discard_potion",
            "args": {"potion_instance_id": potion["instance_id"]},
        })
    actions.append({"action_id": "end_turn", "args": {}})
    return actions


def headless_state_to_model_sample(
    state: dict[str, Any],
    *,
    transition_id: str,
    combat_id: str,
    encounter_signature: str | None = None,
) -> dict[str, Any]:
    if state.get("decision") != "combat_play":
        raise HumanRecordingError(f"online policy requires combat_play, got {state.get('decision')!r}")
    context = state.get("context") or {}
    headless_player = state.get("player") or {}
    enemies = []
    for enemy in state.get("enemies") or []:
        enemy_index = int(enemy.get("index") or 0)
        enemies.append({
            "id": enemy.get("id"),
            "index": enemy_index,
            # Combat Action V0 reserves bare 0 as a missing-target sentinel in
            # legacy recordings, so namespace the headless engine's integer ID.
            "combat_id": f"headless:{enemy.get('combat_id', enemy_index)}",
            "hp": enemy.get("hp"),
            "max_hp": enemy.get("max_hp"),
            "block": enemy.get("block"),
            "intends_attack": enemy.get("intends_attack"),
            "intent": [
                {
                    **intent,
                    "is_attack": bool(intent.get("damage") is not None),
                }
                for intent in enemy.get("intents") or [] if isinstance(intent, dict)
            ],
            "powers": enemy.get("powers") or [],
        })
    hand = [
        _card_for_recorder(card, instance_id=f"hand:{card['index']}")
        for card in state.get("hand") or [] if isinstance(card, dict)
    ]
    piles = state.get("piles") or {}
    potions = []
    for potion in headless_player.get("potions") or []:
        if not isinstance(potion, dict):
            continue
        index = int(potion.get("index") or 0)
        potions.append({
            "id": potion.get("id"),
            "index": index,
            "instance_id": f"potion:{index}",
            "target_type": potion.get("target_type"),
            "vars": potion.get("vars") or {},
        })
    recorder_observation = {
        "phase": "combat_play",
        "run": {
            "act": int(context.get("act") or 0),
            "total_floor": int(context.get("total_floor", context.get("floor")) or 0),
            "ascension": int(context.get("ascension") or 0),
            "room_type": context.get("room_type"),
        },
        "player": {
            "character_id": headless_player.get("id"),
            "hp": headless_player.get("hp"),
            "max_hp": headless_player.get("max_hp"),
            "block": headless_player.get("block"),
            "gold": headless_player.get("gold"),
            "relics": [
                {
                    "id": relic.get("id"),
                    "index": relic.get("index"),
                    "stack_count": 1,
                    "status": "Normal",
                    "dynamic_vars": relic.get("vars") or {},
                    "persistent_state": [],
                    "runtime_state": [],
                }
                for relic in headless_player.get("relics") or [] if isinstance(relic, dict)
            ],
            "potions": potions,
        },
        "combat": {
            "turn": int(state.get("round") or 0),
            "round": int(state.get("round") or 0),
            "turn_phase": "Play",
            "energy": state.get("energy"),
            "max_energy": state.get("max_energy"),
            "stars": state.get("stars", 0),
            "orb_slots": state.get("orb_slots"),
            "draw_pile_count": state.get("draw_pile_count"),
            "discard_pile_count": state.get("discard_pile_count"),
            "exhaust_pile_count": state.get("exhaust_pile_count"),
            "hand": hand,
            "draw_pile": [_card_for_recorder(card) for card in piles.get("draw") or []],
            "discard_pile": [_card_for_recorder(card) for card in piles.get("discard") or []],
            "exhaust_pile": [_card_for_recorder(card) for card in piles.get("exhaust") or []],
            "enemies": enemies,
            "player_powers": state.get("player_powers") or [],
            "orbs": state.get("orbs") or [],
        },
    }
    legal_actions = _legal_actions(state, recorder_observation)
    recorder_observation["legal_actions"] = legal_actions
    candidates, _ = build_action_candidates_v0(recorder_observation, legal_actions[0])
    _attach_search_metadata(candidates, recorder_observation)
    projected = project_observation_v0(recorder_observation)
    # Online callers should cache this value from the first combat decision in
    # state.context.encounter_signature. Falling back to the current enemy set
    # keeps older callers usable, but a kill, summon, or split can then change
    # the adapter identity during the combat.
    stable_encounter_signature = str(
        encounter_signature
        or context.get("encounter_signature")
        or encounter_signature_from_enemies(enemies)
    )
    return {
        "transition_id": transition_id,
        "combat_id": combat_id,
        "split": "online",
        "act": int(context.get("act") or 0),
        "floor": int(context.get("total_floor", context.get("floor")) or 0),
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "encounter_signature": stable_encounter_signature,
        "observation": projected,
        "candidates": candidates,
        "label_index": 0,
        "label_action_type": candidates[0]["action_type"],
        "source_transition_sha256": None,
    }


def candidate_to_headless_command(candidate: dict[str, Any]) -> dict[str, Any]:
    action_type = candidate["action_type"]
    if action_type == "end_turn":
        return {"cmd": "action", "action": "end_turn"}
    if action_type in {"select_cards", "skip_select"}:
        engine_action = candidate.get("engine_action") or {}
        action_id = str(engine_action.get("action_id") or action_type)
        args = dict(engine_action.get("args") or {})
        return {"cmd": "action", "action": action_id, "args": args}
    if action_type == "play_card":
        args: dict[str, Any] = {"card_index": int(candidate["source_index"])}
    elif action_type in {"use_potion", "discard_potion"}:
        source_ref = str(candidate.get("source_ref") or "")
        try:
            potion_index = int(source_ref.rsplit(":", 1)[-1])
        except ValueError:
            potion_index = int(candidate["source_index"])
        args = {"potion_index": potion_index}
    else:
        raise HumanRecordingError(f"unsupported online combat action: {action_type}")
    if candidate.get("target_kind") == "enemy":
        args["target_index"] = int(candidate["target_index"])
    return {"cmd": "action", "action": action_type, "args": args}


def first_card_select_candidate(
    state: dict[str, Any], *, preferred_card_ids: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Choose one explicit combat card-selection option.

    Some enemy-turn selectors advertise ``min_select=0`` even though sending
    an empty selection makes sts2-cli implicitly choose index zero.  Returning
    the concrete card index keeps execution traces faithful to the action that
    the engine actually takes.  A caller may provide human-selected card IDs;
    otherwise the first visible option is used.
    """
    cards = list(state.get("cards") or [])
    if not cards:
        if int(state.get("min_select") or 0) == 0:
            return {
                "candidate_id": "skip_select",
                "candidate_index": 0,
                "action_type": "skip_select",
                "source_type": "card_selection_fallback",
                "source_id": None,
                "target_id": None,
                "engine_action": {"action_id": "skip_select", "args": {}},
            }
        raise HumanRecordingError("combat card selection exposed no cards")

    preferred = set(preferred_card_ids)
    selected = next(
        (card for card in cards if str(card.get("id")) in preferred),
        cards[0],
    )
    selected_index = int(selected.get("index", 0))
    selected_id = str(selected.get("id") or "")
    args = {"indices": str(selected_index)}
    return {
        "candidate_id": f"select_cards:{selected_index}:{selected_id}",
        "candidate_index": selected_index,
        "action_type": "select_cards",
        "source_type": "card_selection_human_match" if selected_id in preferred else "card_selection_first",
        "source_id": selected_id or None,
        "target_id": None,
        "engine_action": {"action_id": "select_cards", "args": args},
    }
