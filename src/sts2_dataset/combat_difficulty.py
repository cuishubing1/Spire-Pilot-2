from __future__ import annotations

from typing import Any

from .human import HumanRecordingError


COMBAT_DIFFICULTY_POLICY = "sts2-v0.107.1-enemy-stats"
COMBAT_DIFFICULTY_TIERS: tuple[dict[str, Any], ...] = (
    {
        "id": "base_a0_a7",
        "minimum_ascension": 0,
        "maximum_ascension": 7,
        "enemy_stat_modifiers": [],
    },
    {
        "id": "tough_a8",
        "minimum_ascension": 8,
        "maximum_ascension": 8,
        "enemy_stat_modifiers": ["tough_enemies"],
    },
    {
        "id": "deadly_a9_a10",
        "minimum_ascension": 9,
        "maximum_ascension": 10,
        "enemy_stat_modifiers": ["tough_enemies", "deadly_enemies"],
        "note": "A10 changes Act 3 boss progression, not a fixed combat's enemy stats.",
    },
)


def combat_difficulty_tier(ascension: int) -> str:
    value = int(ascension)
    for tier in COMBAT_DIFFICULTY_TIERS:
        if int(tier["minimum_ascension"]) <= value <= int(tier["maximum_ascension"]):
            return str(tier["id"])
    raise HumanRecordingError(
        f"unsupported ascension for {COMBAT_DIFFICULTY_POLICY}: {value}"
    )


def combat_difficulty_definitions() -> list[dict[str, Any]]:
    return [dict(value) for value in COMBAT_DIFFICULTY_TIERS]
