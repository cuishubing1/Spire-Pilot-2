from __future__ import annotations

import itertools
from typing import Any

from .types import Action
from .util import canonical_json


def _action(name: str, args: dict[str, Any] | None = None, label: str = "", source=None) -> Action:
    args = args or {}
    action_id = f"{name}:{canonical_json(args)}"
    return Action(action_id=action_id, action=name, args=args, label=label or name, source=source)


def enumerate_legal_actions(state: dict[str, Any]) -> list[Action]:
    phase = state.get("decision")
    player = state.get("player") or {}
    gold = int(player.get("gold") or 0)

    if phase == "game_over":
        return []
    if phase == "map_select":
        return [
            _action("select_map_node", {"col": c["col"], "row": c["row"]}, c.get("type", "map"), c)
            for c in state.get("choices", [])
        ]
    if phase == "combat_play":
        actions: list[Action] = []
        enemies = state.get("enemies", [])
        for card in state.get("hand", []):
            if not card.get("can_play", False):
                continue
            base = {"card_index": card["index"]}
            if card.get("target_type") == "AnyEnemy":
                for enemy in enemies:
                    args = dict(base, target_index=enemy["index"])
                    actions.append(_action("play_card", args, card.get("name", card.get("id", "card")), card))
            else:
                actions.append(_action("play_card", base, card.get("name", card.get("id", "card")), card))
        for potion in player.get("potions", []):
            target_type = potion.get("target_type")
            if target_type == "AnyEnemy":
                for enemy in enemies:
                    args = {"potion_index": potion["index"], "target_index": enemy["index"]}
                    actions.append(_action("use_potion", args, potion.get("name", "potion"), potion))
            else:
                actions.append(_action("use_potion", {"potion_index": potion["index"]}, potion.get("name", "potion"), potion))
            actions.append(_action("discard_potion", {"potion_index": potion["index"]}, "discard potion", potion))
        actions.append(_action("end_turn"))
        return actions
    if phase in {"event_choice", "rest_site"}:
        result = []
        for option in state.get("options", []):
            enabled = option.get("is_enabled", True) and not option.get("is_locked", False)
            if enabled:
                result.append(_action("choose_option", {"option_index": option["index"]}, option.get("title") or option.get("name") or option.get("option_id", "option"), option))
        return result or [_action("leave_room")]
    if phase == "card_reward":
        result = [
            _action("select_card_reward", {"card_index": c["index"]}, c.get("name", c.get("id", "card")), c)
            for c in state.get("cards", [])
        ]
        if state.get("can_skip", True):
            result.append(_action("skip_card_reward"))
        return result
    if phase == "bundle_select":
        return [_action("select_bundle", {"bundle_index": b["index"]}, f"bundle {b['index']}", b) for b in state.get("bundles", [])]
    if phase == "card_select":
        cards = state.get("cards", [])
        minimum = int(state.get("min_select", 1))
        maximum = min(int(state.get("max_select", minimum)), len(cards))
        result = []
        for count in range(minimum, maximum + 1):
            for combo in itertools.combinations([c["index"] for c in cards], count):
                encoded = ",".join(map(str, combo))
                result.append(_action("select_cards", {"indices": encoded}, f"select {encoded}"))
        if minimum == 0:
            result.append(_action("skip_select"))
        return result
    if phase == "shop":
        result = []
        for kind, action in (("cards", "buy_card"), ("relics", "buy_relic"), ("potions", "buy_potion")):
            singular = kind[:-1] if kind != "relics" else "relic"
            for item in state.get(kind, []):
                if item.get("is_stocked", True) and int(item.get("cost") or 0) <= gold:
                    result.append(_action(action, {f"{singular}_index": item["index"]}, item.get("name", item.get("id", action)), item))
        removal_cost = state.get("card_removal_cost")
        if removal_cost is not None and int(removal_cost) <= gold:
            result.append(_action("remove_card", label="remove card"))
        result.append(_action("leave_room"))
        return result
    raise ValueError(f"Unsupported decision phase: {phase!r}")

