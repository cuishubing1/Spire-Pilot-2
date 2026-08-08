from sts2_dataset.policy import HeuristicPolicy


def test_policy_only_returns_legal_action():
    envelope = {
        "phase": "map_select",
        "agent_observation": {"player": {"hp": 20, "max_hp": 80, "gold": 50}, "screen": {}},
        "legal_actions": [
            {"action_id": "m", "action": "select_map_node", "args": {"col": 1, "row": 1}, "label": "Monster", "source": {"type": "Monster"}},
            {"action_id": "r", "action": "select_map_node", "args": {"col": 2, "row": 1}, "label": "Rest", "source": {"type": "RestSite"}},
        ],
    }
    action = HeuristicPolicy("seed").choose(envelope)
    assert action["action_id"] == "r"

