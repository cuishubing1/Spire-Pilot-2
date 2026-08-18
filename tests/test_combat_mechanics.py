from sts2_dataset.combat_mechanics import MechanicDirectiveControllerV0


def _candidate(candidate_id, action_type, *, target_ref=None, block=0, damage=0):
    return {
        "candidate_id": candidate_id,
        "candidate_index": 0,
        "action_type": action_type,
        "source_ref": f"card:{candidate_id}" if action_type == "play_card" else None,
        "target_ref": target_ref,
        "target_kind": "enemy" if target_ref else "none",
    }


def _sample(enemies, candidates, *, round_number=1, block=0):
    hand = []
    for row in candidates:
        if row["action_type"] != "play_card":
            continue
        target = next((enemy for enemy in enemies if enemy["entity_ref"] == row["target_ref"]), None)
        hand.append({
            "id": f"CARD.{row['candidate_id']}",
            "entity_ref": row["source_ref"],
            "stats": {
                "block": row.pop("_block", 0),
                "damage": row.pop("_damage", 0),
            },
            "damage_by_target": ([{
                "target_combat_id": row["target_ref"],
                "damage": row.pop("_preview_damage", 0),
                "hits": 1,
            }] if target else []),
        })
    return {
        "observation": {
            "global": {"round": round_number, "block": block, "max_hp": 80, "energy": 3},
            "hand": hand,
            "piles": {"draw": [], "discard": [], "exhaust": []},
            "enemies": enemies,
            "potions": [],
        },
        "candidates": candidates,
    }


def test_bowlbug_guidance_rewards_candidate_completing_full_block():
    enemies = [{
        "id": "MONSTER.BOWLBUG_ROCK", "entity_ref": "enemy:rock", "hp": 40,
        "intends_attack": True,
        "intent": [{"type": "Attack", "damage": 15, "is_attack": True}],
        "powers": [{"id": "POWER.IMBALANCED_POWER", "amount": 1}],
    }]
    defend = _candidate("defend", "play_card")
    defend["_block"] = 15
    strike = _candidate("strike", "play_card", target_ref="enemy:rock")
    strike["_preview_damage"] = 6
    directive, diagnostics = MechanicDirectiveControllerV0(
        "bowlbug_rock_full_block"
    ).directive_for(_sample(enemies, [defend, strike, _candidate("end", "end_turn")]))
    assert diagnostics["phase"] == "complete_visible_block"
    assert directive.candidate_biases["defend"] > 3.0
    assert set(directive.forbidden_candidate_ids) == {"strike", "end"}


def test_bowlbug_guidance_falls_back_when_visible_hand_cannot_full_block():
    enemies = [{
        "id": "MONSTER.BOWLBUG_ROCK", "entity_ref": "enemy:rock", "hp": 40,
        "intends_attack": True,
        "intent": [{"type": "Attack", "damage": 15, "is_attack": True}],
        "powers": [{"id": "POWER.IMBALANCED_POWER", "amount": 1}],
    }]
    defend = _candidate("defend", "play_card")
    defend["_block"] = 5
    directive, diagnostics = MechanicDirectiveControllerV0(
        "bowlbug_rock_full_block"
    ).directive_for(_sample(enemies, [defend, _candidate("end", "end_turn")]))
    assert diagnostics["phase"] == "block_plan_infeasible"
    assert diagnostics["maximum_immediate_block"] == 5.0
    assert directive.candidate_biases == {}


def test_terror_guidance_delays_threshold_then_switches_to_burst():
    eel = {
        "id": "MONSTER.TERROR_EEL", "entity_ref": "enemy:eel", "hp": 75, "max_hp": 140,
        "powers": [{"id": "POWER.SHRIEK_POWER", "amount": 70}],
    }
    hit = _candidate("hit", "play_card", target_ref="enemy:eel")
    hit["_preview_damage"] = 10
    controller = MechanicDirectiveControllerV0("terror_eel_threshold_burst", terror_setup_rounds=2)
    directive, diagnostics = controller.directive_for(_sample([eel], [hit], round_number=1))
    assert diagnostics["phase"] == "pre_threshold_setup"
    assert directive.forbidden_candidate_ids == ("hit",)

    eel["hp"] = 65
    eel["powers"] = []
    hit = _candidate("hit", "play_card", target_ref="enemy:eel")
    hit["_preview_damage"] = 10
    directive, diagnostics = controller.directive_for(_sample([eel], [hit], round_number=3))
    assert diagnostics["phase"] == "post_threshold_burst"
    assert directive.candidate_biases["hit"] > 1.0


def test_overgrowth_guidance_prioritizes_shrinker_only_while_both_live():
    enemies = [
        {"id": "MONSTER.SHRINKER_BEETLE", "entity_ref": "enemy:0", "hp": 30},
        {"id": "MONSTER.FUZZY_WURM_CRAWLER", "entity_ref": "enemy:1", "hp": 50},
    ]
    controller = MechanicDirectiveControllerV0("overgrowth_shrinker_priority")
    directive, diagnostics = controller.directive_for(_sample(enemies, [], round_number=1))
    assert diagnostics["phase"] == "focus_shrinker"
    assert directive.target_biases["enemy:0"] > 0.0
    assert directive.target_biases["enemy:1"] < 0.0
