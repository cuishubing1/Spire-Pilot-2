from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_combat_turn_boundary_eval import (
    _best_supported_leaf,
    _choose_turn_boundary_root,
    _is_forbidden_potion_action,
    _shared_world_candidate_prefix,
)


def test_best_leaf_prefers_real_boundary_over_higher_scored_card_select():
    card_select = {
        "path_score": 10.0,
        "commands": [{"action": "use_potion"}],
        "leaf_decision": "card_select",
        "supported_boundary": False,
    }
    next_turn = {
        "path_score": 1.0,
        "commands": [{"action": "end_turn"}],
        "leaf_decision": "combat_play",
        "supported_boundary": True,
    }

    assert _best_supported_leaf([card_select, next_turn]) is next_turn


def _candidate(action: str, card_index: int | None = None) -> dict:
    args = {} if card_index is None else {"card_index": card_index}
    return {
        "action_type": action,
        "candidate_id": f"{action}:{card_index}",
        "source_index": card_index,
        "target_kind": "none",
        "engine_action": {"action_id": action, "args": args},
    }


def test_shared_world_prefix_stops_before_determinization_divergence():
    strike = _candidate("play_card", 0)
    defend = _candidate("play_card", 1)
    bash = _candidate("play_card", 2)
    worlds = [
        {"candidate_sequence": [strike, defend]},
        {"candidate_sequence": [strike, bash]},
    ]
    assert _shared_world_candidate_prefix(worlds) == [strike]


def test_shared_world_prefix_never_commits_beyond_end_turn():
    strike = _candidate("play_card", 0)
    end_turn = _candidate("end_turn")
    next_turn_a = _candidate("play_card", 2)
    next_turn_b = _candidate("play_card", 3)
    worlds = [
        {"candidate_sequence": [strike, end_turn, next_turn_a]},
        {"candidate_sequence": [strike, end_turn, next_turn_b]},
    ]
    assert _shared_world_candidate_prefix(worlds) == [strike, end_turn]


def test_all_failed_search_worlds_fall_back_to_policy_root():
    strike = _candidate("play_card", 0)
    defend = _candidate("play_card", 1)
    shortlist = [
        {"candidate": strike, "policy_probability": 0.7},
        {"candidate": defend, "policy_probability": 0.3},
    ]
    evaluations = [
        {
            **row,
            "selection_score": 0.0,
            "selection_eligible": False,
            "worlds": [{"engine_error": "unsupported simulator branch"}],
        }
        for row in shortlist
    ]

    chosen, fallback = _choose_turn_boundary_root(
        evaluations,
        shortlist,
        minimum_value_advantage=0.02,
        minimum_end_turn_advantage=0.15,
    )

    assert chosen["candidate"] == strike
    assert fallback == {
        "reason": "all_search_candidates_ineligible",
        "engine_errors": ["unsupported simulator branch"],
    }


def test_forbidden_potion_constraint_blocks_use_and_discard():
    forbidden = {"POTION.FIRE_POTION"}
    assert _is_forbidden_potion_action(
        {"action_type": "use_potion", "source_id": "POTION.FIRE_POTION"}, forbidden
    )
    assert _is_forbidden_potion_action(
        {"action_type": "discard_potion", "source_id": "POTION.FIRE_POTION"}, forbidden
    )
    assert not _is_forbidden_potion_action(
        {"action_type": "use_potion", "source_id": "POTION.BLOCK_POTION"}, forbidden
    )
    assert not _is_forbidden_potion_action(
        {"action_type": "play_card", "source_id": "CARD.STRIKE_IRONCLAD"}, forbidden
    )
