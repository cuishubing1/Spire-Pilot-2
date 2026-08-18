import json

import pytest

from sts2_dataset.fixed_plan import build_fixed_noncombat_plan, fixed_plan_command
from sts2_dataset.human import HumanRecordingError


def _row(sequence, phase, action, *, eligible=True):
    return {
        "record_sequence": sequence,
        "act": 1,
        "floor": sequence,
        "phase": phase,
        "is_canonical": True,
        "is_training_eligible": eligible,
        "action_json": json.dumps(action),
    }


def test_fixed_plan_filters_engine_automatic_and_intermediate_actions():
    rows = [
        _row(3, "reward_select", {"action_id": "select_reward", "args": {}}),
        _row(2, "card_select", {"action_id": "choose_card", "args": {"card_id": "CARD.A"}}),
        _row(1, "combat_play", {"action_id": "end_turn", "args": {}}),
        _row(4, "card_select", {
            "action_id": "confirm_card_selection", "args": {"selected_card_ids": ["CARD.A"]},
        }),
    ]
    plan = build_fixed_noncombat_plan(rows)
    assert [value["record_sequence"] for value in plan] == [4]


def test_fixed_plan_maps_identity_not_stale_reward_index():
    state = {
        "decision": "card_reward",
        "cards": [{"index": 0, "id": "CARD.B"}, {"index": 1, "id": "CARD.A"}],
        "can_skip": True,
    }
    entry = {"phase": "card_reward", "action": {
        "action_id": "choose_card_reward", "args": {"card_id": "CARD.A"},
    }}
    assert fixed_plan_command(state, entry) == {
        "cmd": "action", "action": "select_card_reward", "args": {"card_index": 1},
    }


def test_fixed_plan_refuses_unavailable_choice():
    state = {"decision": "event_choice", "options": [{"index": 0, "is_locked": True}]}
    entry = {"phase": "event_choice", "action": {
        "action_id": "choose_event_option", "args": {"index": 0},
    }}
    with pytest.raises(HumanRecordingError, match="unavailable"):
        fixed_plan_command(state, entry)


def test_fixed_plan_collapses_duplicate_shop_exit_hook():
    rows = [
        _row(10, "shop", {"action_id": "leave_shop", "args": {}}),
        _row(11, "shop", {"action_id": "leave_shop", "args": {}}),
    ]
    rows[1]["floor"] = rows[0]["floor"]
    plan = build_fixed_noncombat_plan(rows)
    assert len(plan) == 1


def test_fixed_plan_ignores_trailing_shop_actions_after_exit():
    rows = [
        _row(10, "shop", {"action_id": "buy_shop_item", "args": {"id": "CARD.A"}}),
        _row(11, "shop", {"action_id": "leave_shop", "args": {}}),
        _row(12, "shop", {"action_id": "buy_shop_item", "args": {"id": "CARD.B"}}),
        _row(13, "shop", {"action_id": "leave_shop", "args": {}}),
        _row(14, "map_select", {"action_id": "select_map_node", "args": {}}),
    ]
    for row in rows:
        row["floor"] = 31
    plan = build_fixed_noncombat_plan(rows)
    assert [value["record_sequence"] for value in plan] == [10, 11, 14]
