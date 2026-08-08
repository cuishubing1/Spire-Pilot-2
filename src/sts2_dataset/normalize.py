from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from . import SCHEMA_VERSION
from .constants import KNOWN_PHASES
from .legal_actions import enumerate_legal_actions
from .types import AuditRef, ObservationEnvelope
from .util import state_hash


VISIBLE_TOP_LEVEL = {
    "act",
    "act_name",
    "floor",
    "round",
    "energy",
    "max_energy",
    "hand",
    "enemies",
    "player_powers",
    "draw_pile_count",
    "discard_pile_count",
    "orbs",
    "orb_slots",
    "stars",
    "osty",
    "choices",
    "cards",
    "bundles",
    "options",
    "can_skip",
    "from_event",
    "min_select",
    "max_select",
    "event_id",
    "event_name",
    "description",
    "relics",
    "potions",
    "card_removal_cost",
    "gold_earned",
    "victory",
}


def _normalize_id(value: Any, namespace: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing stable {namespace} id")
    text = str(value).strip()
    if text.endswith(".UNKNOWN") or text == "?":
        raise ValueError(f"Unknown stable {namespace} id: {text}")
    return text


def _display_names(value: Any) -> Any:
    if isinstance(value, list):
        return [_display_names(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        result["display_name" if key == "name" else key] = _display_names(item)
    return result


def _assign_instance_ids(items: list[dict[str, Any]], scope: str) -> None:
    counts: defaultdict[str, int] = defaultdict(int)
    for item in items:
        model_id = _normalize_id(item.get("id"), scope)
        occurrence = counts[model_id]
        counts[model_id] += 1
        # Deterministic within an observation. It deliberately does not claim engine identity.
        item["instance_id"] = f"{scope}/{model_id}/{occurrence}"


def _validate_and_enrich_entities(obs: dict[str, Any]) -> None:
    player = obs.get("player") or {}
    _normalize_id(player.get("id"), "character")
    for key, namespace in (("deck", "deck"), ("relics", "relic"), ("potions", "potion")):
        items = [x for x in player.get(key, []) if isinstance(x, dict)]
        _assign_instance_ids(items, namespace)
    for key, namespace in (("hand", "hand"), ("enemies", "enemy"), ("cards", "card"), ("relics", "relic"), ("potions", "potion")):
        items = [x for x in obs.get("screen", {}).get(key, []) if isinstance(x, dict)]
        if items:
            _assign_instance_ids(items, namespace)
    for bundle in obs.get("screen", {}).get("bundles", []):
        cards = [x for x in bundle.get("cards", []) if isinstance(x, dict)]
        _assign_instance_ids(cards, f"bundle-{bundle.get('index', 0)}")
    event_id = obs.get("screen", {}).get("event_id")
    if obs.get("phase") == "event_choice":
        _normalize_id(event_id, "event")


def normalize_observation(
    raw: dict[str, Any],
    *,
    config: dict[str, Any],
    run_id: str,
    step_id: int,
    audit_ref: AuditRef | None,
    visible_map: dict[str, Any] | None = None,
) -> ObservationEnvelope:
    if raw.get("type") == "error":
        raise RuntimeError(f"Engine error: {raw.get('message')}")
    if raw.get("type") != "decision":
        raise ValueError(f"Expected decision response, got {raw.get('type')!r}")
    phase = raw.get("decision")
    if phase not in KNOWN_PHASES:
        raise ValueError(f"Unknown decision phase: {phase!r}")

    context = copy.deepcopy(raw.get("context") or {})
    context.setdefault("act", raw.get("act"))
    context.setdefault("floor", raw.get("floor"))
    screen = {key: copy.deepcopy(raw[key]) for key in VISIBLE_TOP_LEVEL if key in raw}
    if phase == "shop":
        # v0.107.1 clears a purchased entry's Model while retaining an empty
        # unstocked slot. It is not an entity or legal action; preserve it only
        # in raw engine JSON and omit it from the training view.
        for inventory_key in ("cards", "relics", "potions"):
            if inventory_key in screen:
                screen[inventory_key] = [
                    item for item in screen[inventory_key] if item.get("is_stocked", True)
                ]
    player = copy.deepcopy(raw.get("player") or {})
    if isinstance(player.get("deck"), list):
        # Deck list order is not player-visible or strategically meaningful, and
        # RunState load reconstructs this collection in a different insertion
        # order. Canonicalize it as a multiset before assigning occurrence IDs.
        player["deck"].sort(
            key=lambda card: (
                str(card.get("id", "")),
                bool(card.get("upgraded", False)),
                str(card.get("enchantment", "")),
                str(card.get("affliction", "")),
            )
        )
    agent_observation = _display_names(
        {
            "phase": phase,
            "context": context,
            "player": player,
            "visible_map": copy.deepcopy(visible_map),
            "screen": screen,
        }
    )
    _validate_and_enrich_entities(agent_observation)
    legal = [a.to_dict() for a in enumerate_legal_actions(raw)]
    fingerprint = {
        "game_version": config["game_version"],
        "steam_build_id": config["steam_build_id"],
        "sts2_dll_sha256": config["sts2_dll_sha256"],
        "sts2_cli_commit": config["sts2_cli_commit"],
        "sts2_cli_protocol": config["sts2_cli_protocol"],
    }
    digest = state_hash(agent_observation)
    terminal = phase == "game_over"
    reason = None
    if terminal:
        reason = "victory" if screen.get("victory") else "defeat"
    return ObservationEnvelope(
        schema_version=SCHEMA_VERSION,
        dataset_version=config["dataset_version"],
        run_id=run_id,
        step_id=step_id,
        game_fingerprint=fingerprint,
        phase=phase,
        context=context,
        agent_observation=agent_observation,
        legal_actions=legal,
        audit_ref=audit_ref.to_dict() if audit_ref else None,
        state_hash=digest,
        terminal=terminal,
        terminal_reason=reason,
    )


def outcome_delta(before: ObservationEnvelope, after: ObservationEnvelope) -> dict[str, Any]:
    a = before.agent_observation.get("player", {})
    b = after.agent_observation.get("player", {})

    def ids(player: dict[str, Any], key: str) -> list[str]:
        return [x.get("id") for x in player.get(key, []) if isinstance(x, dict)]

    before_deck, after_deck = ids(a, "deck"), ids(b, "deck")
    before_relics, after_relics = ids(a, "relics"), ids(b, "relics")
    before_potions, after_potions = ids(a, "potions"), ids(b, "potions")
    return {
        "hp_delta": int(b.get("hp") or 0) - int(a.get("hp") or 0),
        "max_hp_delta": int(b.get("max_hp") or 0) - int(a.get("max_hp") or 0),
        "gold_delta": int(b.get("gold") or 0) - int(a.get("gold") or 0),
        "floor_delta": int(after.context.get("floor") or 0) - int(before.context.get("floor") or 0),
        "deck_added": _multiset_added(before_deck, after_deck),
        "deck_removed": _multiset_added(after_deck, before_deck),
        "relics_added": _multiset_added(before_relics, after_relics),
        "potions_added": _multiset_added(before_potions, after_potions),
        "potions_removed": _multiset_added(after_potions, before_potions),
        "terminal": after.terminal,
        "victory": after.agent_observation.get("screen", {}).get("victory") if after.terminal else None,
    }


def _multiset_added(before: list[str], after: list[str]) -> list[str]:
    remaining = list(before)
    result = []
    for item in after:
        if item in remaining:
            remaining.remove(item)
        else:
            result.append(item)
    return result
