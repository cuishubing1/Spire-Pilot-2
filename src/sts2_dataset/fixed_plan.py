from __future__ import annotations

import json
from typing import Any, Iterable

from .human import HumanRecordingError


IGNORED_HEADLESS_PHASES = {"reward_select", "treasure", "potion_manage"}


def build_fixed_noncombat_plan(transitions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract an ordered, canonical non-combat plan from HumanRecorder rows.

    Headless automatically resolves ordinary gold/potion/relic rewards and
    treasure rooms, so their UI-only decisions are omitted.  Multi-click card
    selection records are collapsed to the final confirmed selection.
    """
    result: list[dict[str, Any]] = []
    closed_shops: set[tuple[int, int]] = set()
    for row in sorted(transitions, key=lambda value: int(value["record_sequence"])):
        if not row.get("is_canonical") or not row.get("is_training_eligible"):
            continue
        phase = str(row.get("phase") or "")
        if phase == "combat_play" or phase in IGNORED_HEADLESS_PHASES:
            continue
        action = json.loads(row["action_json"]) if isinstance(row.get("action_json"), str) else row["action_json"]
        action_id = action.get("action_id")
        position = (int(row.get("act") or 0), int(row.get("floor") or 0))
        if phase == "shop" and position in closed_shops:
            # A shop cannot be reopened after leave_shop in the same room.
            # Delayed hooks and old recordings can contain trailing purchases;
            # they are not executable components of a fixed room plan.
            continue
        if phase == "card_select" and action_id == "choose_card":
            continue
        entry = {
            "record_sequence": int(row["record_sequence"]),
            "source_act": int(row.get("act") or 0),
            "source_floor": int(row.get("floor") or 0),
            "phase": phase,
            "action": action,
        }
        result.append(entry)
        if phase == "shop" and action_id == "leave_shop":
            closed_shops.add(position)
    return result


def _indexed(values: Any, index: int, *, field: str) -> dict[str, Any]:
    rows = [value for value in values or [] if isinstance(value, dict)]
    matches = [value for value in rows if int(value.get("index", -1)) == index]
    if len(matches) != 1:
        raise HumanRecordingError(f"fixed plan {field} index {index} is not currently available")
    return matches[0]


def _find_identity(values: Any, identity: str, *, field: str) -> dict[str, Any]:
    rows = [value for value in values or [] if isinstance(value, dict)]
    matches = [value for value in rows if value.get("id") == identity]
    if not matches:
        offered = [value.get("id") for value in rows]
        raise HumanRecordingError(
            f"fixed plan {field} {identity!r} is not offered; current identities={offered!r}"
        )
    return matches[0]


def fixed_plan_command(state: dict[str, Any], plan_entry: dict[str, Any]) -> dict[str, Any]:
    decision = str(state.get("decision") or "")
    phase = str(plan_entry.get("phase") or "")
    if decision != phase:
        raise HumanRecordingError(f"fixed plan phase {phase!r} does not match engine decision {decision!r}")
    action = plan_entry["action"]
    action_id = str(action.get("action_id") or "")
    args = action.get("args") or {}

    if action_id == "select_map_node":
        coord = args.get("coord") or {}
        col, row = int(coord["col"]), int(coord["row"])
        matches = [
            value for value in state.get("choices") or []
            if int(value.get("col", -1)) == col and int(value.get("row", -1)) == row
        ]
        if len(matches) != 1:
            raise HumanRecordingError(f"fixed plan map coordinate ({col}, {row}) is not reachable")
        return {"cmd": "action", "action": "select_map_node", "args": {"col": col, "row": row}}

    if action_id in {"choose_event_option", "choose_rest_option"}:
        index = int(args["index"])
        option = _indexed(state.get("options"), index, field="option")
        if option.get("is_locked") or option.get("is_enabled") is False:
            raise HumanRecordingError(f"fixed plan option {index} is currently unavailable")
        return {"cmd": "action", "action": "choose_option", "args": {"option_index": index}}

    if action_id == "choose_card_reward":
        card = _find_identity(state.get("cards"), str(args["card_id"]), field="card reward")
        return {"cmd": "action", "action": "select_card_reward", "args": {"card_index": int(card["index"])}}
    if action_id == "choose_reward_alternative":
        if not state.get("can_skip", True):
            raise HumanRecordingError("fixed plan wants the card-reward alternative but skipping is unavailable")
        return {"cmd": "action", "action": "skip_card_reward"}

    if action_id == "buy_shop_item":
        identity = str(args.get("id") or "")
        kind = str(args.get("kind") or "").lower()
        if "card" in kind:
            item, command, index_name = _find_identity(state.get("cards"), identity, field="shop card"), "buy_card", "card_index"
        elif "relic" in kind:
            item, command, index_name = _find_identity(state.get("relics"), identity, field="shop relic"), "buy_relic", "relic_index"
        elif "potion" in kind:
            item, command, index_name = _find_identity(state.get("potions"), identity, field="shop potion"), "buy_potion", "potion_index"
        else:
            raise HumanRecordingError(f"unsupported fixed-plan shop kind: {args.get('kind')!r}")
        if not item.get("is_stocked", True):
            raise HumanRecordingError(f"fixed plan shop item {identity!r} is no longer stocked")
        if int((state.get("player") or {}).get("gold") or 0) < int(item.get("cost") or 0):
            raise HumanRecordingError(f"fixed plan shop item {identity!r} is no longer affordable")
        return {"cmd": "action", "action": command, "args": {index_name: int(item["index"])}}
    if action_id == "remove_card":
        cost = state.get("card_removal_cost")
        if cost is None or int((state.get("player") or {}).get("gold") or 0) < int(cost):
            raise HumanRecordingError("fixed plan card removal is no longer affordable")
        return {"cmd": "action", "action": "remove_card"}
    if action_id == "leave_shop":
        return {"cmd": "action", "action": "leave_room"}

    if action_id == "confirm_card_selection":
        selected_ids = [str(value) for value in args.get("selected_card_ids") or []]
        available = [value for value in state.get("cards") or [] if isinstance(value, dict)]
        used: set[int] = set()
        indices: list[int] = []
        for identity in selected_ids:
            match = next(
                (value for value in available if value.get("id") == identity and int(value["index"]) not in used),
                None,
            )
            if match is None:
                raise HumanRecordingError(f"fixed plan selected card {identity!r} is not currently available")
            index = int(match["index"])
            used.add(index)
            indices.append(index)
        minimum = int(state.get("min_select") or 0)
        maximum = int(state.get("max_select") or 0)
        if not minimum <= len(indices) <= maximum:
            raise HumanRecordingError(
                f"fixed plan selects {len(indices)} cards but engine requires {minimum}-{maximum}"
            )
        return {
            "cmd": "action",
            "action": "select_cards",
            "args": {"indices": ",".join(str(value) for value in indices)},
        }
    if action_id == "skip_card_selection":
        if int(state.get("min_select") or 0) > 0:
            raise HumanRecordingError("fixed plan cancels a mandatory card selection")
        return {"cmd": "action", "action": "skip_select"}

    if action_id == "proceed":
        return {"cmd": "action", "action": "proceed"}
    raise HumanRecordingError(f"unsupported fixed-plan action: {action_id!r}")
