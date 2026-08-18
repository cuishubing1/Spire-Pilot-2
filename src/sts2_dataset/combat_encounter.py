from __future__ import annotations

from typing import Any, Iterable


ENCOUNTER_SIGNATURE_VERSION = "initial-enemy-signature-0.1.0"
UNKNOWN_ENCOUNTER_SIGNATURE = "encounter:<UNKNOWN>"


def encounter_signature_from_enemies(enemies: Iterable[dict[str, Any]]) -> str:
    identities = sorted(
        str(enemy.get("id"))
        for enemy in enemies
        if isinstance(enemy, dict) and enemy.get("id")
    )
    if not identities:
        return UNKNOWN_ENCOUNTER_SIGNATURE
    return "encounter:" + "+".join(identities)


def encounter_signature_from_observation(observation: dict[str, Any]) -> str:
    combat = observation.get("combat") or {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if enemies is None:
        enemies = observation.get("enemies") or []
    return encounter_signature_from_enemies(enemies)
