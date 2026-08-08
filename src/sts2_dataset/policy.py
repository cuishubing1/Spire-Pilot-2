from __future__ import annotations

import hashlib
import random
from typing import Any


class HeuristicPolicy:
    policy_id = "heuristic_v1"

    def __init__(self, seed: str):
        numeric = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
        self.random = random.Random(numeric)

    def choose(self, envelope: dict[str, Any]) -> dict[str, Any]:
        actions = envelope["legal_actions"]
        if not actions:
            raise ValueError(f"No legal action for non-terminal phase {envelope['phase']}")
        phase = envelope["phase"]
        observation = envelope["agent_observation"]
        player = observation.get("player", {})
        screen = observation.get("screen", {})

        if phase == "map_select":
            return self._map(actions, player)
        if phase == "combat_play":
            return self._combat(actions, player, screen)
        if phase == "rest_site":
            return self._rest(actions, player)
        if phase == "shop":
            return self._shop(actions, player)
        if phase == "card_reward":
            return self._card_reward(actions, player)
        if phase == "event_choice":
            return self._stable_tie(actions)
        if phase in {"bundle_select", "card_select"}:
            return self._stable_tie(actions)
        return self._stable_tie(actions)

    def _stable_tie(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(actions, key=lambda a: a["action_id"])
        return ordered[self.random.randrange(len(ordered))]

    def _map(self, actions: list[dict[str, Any]], player: dict[str, Any]) -> dict[str, Any]:
        hp_ratio = int(player.get("hp", 0)) / max(1, int(player.get("max_hp", 1)))
        gold = int(player.get("gold", 0))
        base = {"RestSite": 35, "Merchant": 25, "Elite": 15, "Monster": 10, "Unknown": 12, "Treasure": 30}
        scored = []
        for action in actions:
            room_type = str((action.get("source") or {}).get("type", "Unknown"))
            score = base.get(room_type, 10)
            if "Rest" in room_type and hp_ratio < 0.55:
                score += 100
            if "Elite" in room_type:
                score += 45 if hp_ratio > 0.72 else -80
            if "Merchant" in room_type:
                score += 45 if gold >= 150 else -15
            scored.append((score, action))
        best = max(score for score, _ in scored)
        return self._stable_tie([a for score, a in scored if score == best])

    def _combat(self, actions: list[dict[str, Any]], player: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
        enemies = {int(e.get("index", -1)): e for e in screen.get("enemies", [])}
        incoming = sum(
            int(intent.get("total_damage", intent.get("damage", 0)) or 0)
            for enemy in enemies.values()
            for intent in enemy.get("intents", []) or []
        )
        current_block = int(player.get("block", 0))
        scored = []
        for action in actions:
            name = action["action"]
            source = action.get("source") or {}
            stats = source.get("stats") or {}
            score = 0.0
            if name == "play_card":
                target = enemies.get(int(action.get("args", {}).get("target_index", -1)))
                damage = self._damage_for(source, target)
                block = int(stats.get("block", 0) or 0)
                if target and damage >= int(target.get("hp", 0)):
                    score += 10000
                if incoming > current_block:
                    score += min(block, incoming - current_block) * 20
                score += damage * 5 + block * 2
                if str(source.get("type")) == "Power":
                    score += 8
                score -= max(0, int(source.get("cost", 0) or 0)) * 0.1
            elif name == "use_potion":
                score = 250 if incoming >= int(player.get("hp", 0)) else 5
            elif name == "discard_potion":
                score = -1000
            elif name == "end_turn":
                score = -50 if any(a["action"] == "play_card" for a in actions) else 0
            scored.append((score, action))
        best = max(score for score, _ in scored)
        return self._stable_tie([a for score, a in scored if score == best])

    @staticmethod
    def _damage_for(card: dict[str, Any], target: dict[str, Any] | None) -> int:
        if target is not None:
            for row in card.get("damage_by_target", []) or []:
                if row.get("target_index") == target.get("index"):
                    return int(row.get("total_damage", row.get("damage", 0)) or 0)
        stats = card.get("stats") or {}
        damage = int(stats.get("calculateddamage", stats.get("damage", 0)) or 0)
        repeat = int(stats.get("repeat", 1) or 1)
        return damage * repeat

    def _rest(self, actions: list[dict[str, Any]], player: dict[str, Any]) -> dict[str, Any]:
        hp_ratio = int(player.get("hp", 0)) / max(1, int(player.get("max_hp", 1)))
        if hp_ratio < 0.60:
            heals = [a for a in actions if str((a.get("source") or {}).get("option_id", "")).upper() == "HEAL"]
            if heals:
                return self._stable_tie(heals)
        upgrades = [a for a in actions if str((a.get("source") or {}).get("option_id", "")).upper() in {"SMITH", "UPGRADE"}]
        return self._stable_tie(upgrades or actions)

    def _shop(self, actions: list[dict[str, Any]], player: dict[str, Any]) -> dict[str, Any]:
        priority = {"buy_relic": 4, "remove_card": 3, "buy_card": 2, "buy_potion": 1, "leave_room": 0}
        scored = []
        for action in actions:
            source = action.get("source") or {}
            score = priority.get(action["action"], 0) * 100
            rarity = str(source.get("rarity", ""))
            score += {"Rare": 30, "Uncommon": 15, "Common": 5}.get(rarity, 0)
            score -= int(source.get("cost", 0) or 0) / 1000
            scored.append((score, action))
        best = max(score for score, _ in scored)
        return self._stable_tie([a for score, a in scored if score == best])

    def _card_reward(self, actions: list[dict[str, Any]], player: dict[str, Any]) -> dict[str, Any]:
        deck_size = len(player.get("deck", []))
        scored = []
        for action in actions:
            if action["action"] == "skip_card_reward":
                score = 12 if deck_size >= 30 else -5
            else:
                source = action.get("source") or {}
                score = {"Rare": 30, "Uncommon": 18, "Common": 8}.get(str(source.get("rarity", "")), 5)
                if str(source.get("type")) == "Power":
                    score += 3
            scored.append((score, action))
        best = max(score for score, _ in scored)
        return self._stable_tie([a for score, a in scored if score == best])

