import json

from sts2_dataset.combat_value import derive_combat_value_targets


def _observation(hp, max_hp=80, potions=None, phase="combat_play", victory=None):
    value = {
        "phase": phase,
        "player": {"hp": hp, "max_hp": max_hp, "potions": potions or []},
    }
    if victory is not None:
        value["screen"] = {"victory": victory}
    return value


def _row(step, current, following, action_id="play_card"):
    return {
        "transition_id": f"t{step}",
        "combat_id": "combat-a",
        "split": "train",
        "record_sequence": step,
        "step_id": step,
        "observation_json": json.dumps(current),
        "next_observation_json": json.dumps(following),
        "action_json": json.dumps({"action_id": action_id, "args": {}}),
        "source_transition_sha256": f"sha-{step}",
    }


def test_value_targets_assign_future_cost_to_each_decision_without_editing_raw_reward():
    potion = [{"instance_id": "potion-1", "id": "POTION.FLEX"}]
    rows = [
        _row(
            1,
            _observation(50, potions=potion),
            _observation(45, potions=[]),
            action_id="use_potion",
        ),
        _row(2, _observation(45, potions=[]), _observation(46, max_hp=81, phase="card_reward")),
    ]
    targets = derive_combat_value_targets(rows)
    assert targets[0]["hp_loss_to_end"] == 4
    assert targets[0]["potion_spent_to_end"] == 1
    assert targets[0]["max_hp_delta_to_end"] == 1
    assert targets[0]["immediate_hp_loss"] == 5
    assert targets[0]["immediate_hp_loss_fraction"] == 5 / 80
    assert targets[1]["hp_loss_to_end"] == 0
    assert targets[1]["max_hp_delta_to_end"] == 1
    assert not targets[0]["death"]


def test_value_targets_count_actions_instead_of_slot_dependent_potion_ids():
    before = [
        {"instance_id": "potion:0:stable-a", "id": "POTION.FLEX"},
        {"instance_id": "potion:1:stable-b", "id": "POTION.FIRE"},
    ]
    compacted = [
        {"instance_id": "potion:0:stable-b", "id": "POTION.FIRE"},
    ]
    rows = [
        _row(
            1,
            _observation(50, potions=before),
            _observation(50, potions=compacted),
            action_id="use_potion",
        ),
        _row(2, _observation(50, potions=compacted), _observation(50, potions=compacted)),
    ]

    targets = derive_combat_value_targets(rows)

    assert targets[0]["potion_spent_to_end"] == 1
    assert targets[1]["potion_spent_to_end"] == 0


def test_value_targets_mark_terminal_death():
    rows = [_row(1, _observation(4), _observation(0, phase="game_over", victory=False))]
    target = derive_combat_value_targets(rows)[0]
    assert target["death"]
    assert target["hp_loss_to_end"] == 4
