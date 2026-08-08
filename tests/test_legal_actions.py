from sts2_dataset.legal_actions import enumerate_legal_actions


def test_combat_actions_are_fully_grounded():
    state = {
        "decision": "combat_play",
        "player": {"gold": 0, "potions": []},
        "hand": [
            {"index": 0, "id": "CARD.STRIKE", "name": "Strike", "can_play": True, "target_type": "AnyEnemy"},
            {"index": 1, "id": "CARD.DEFEND", "name": "Defend", "can_play": True, "target_type": "Self"},
        ],
        "enemies": [{"index": 0, "id": "MONSTER.SLIME"}, {"index": 1, "id": "MONSTER.SLIME"}],
    }
    actions = enumerate_legal_actions(state)
    assert [a.action for a in actions].count("play_card") == 3
    assert actions[-1].action == "end_turn"
    assert len({a.action_id for a in actions}) == len(actions)


def test_locked_event_options_are_not_legal():
    actions = enumerate_legal_actions(
        {
            "decision": "event_choice",
            "player": {"gold": 0},
            "options": [
                {"index": 0, "title": "locked", "is_locked": True},
                {"index": 1, "title": "open", "is_locked": False},
            ],
        }
    )
    assert len(actions) == 1
    assert actions[0].args == {"option_index": 1}


def test_card_select_enumerates_combinations():
    state = {
        "decision": "card_select",
        "player": {"gold": 0},
        "cards": [{"index": i, "id": f"CARD.{i}"} for i in range(3)],
        "min_select": 1,
        "max_select": 2,
    }
    actions = enumerate_legal_actions(state)
    assert len(actions) == 6

