from __future__ import annotations

import pytest

from sts2_dataset.combat_potion_evaluator import build_paired_potion_proposal


def _world(world_id: str, *, value: float, death: float, hp: float, hp_loss: float,
           enemy_hp_loss: float, potion_delta: float = 0.0) -> dict:
    return {
        "determinization_id": world_id,
        "outcome": {
            "value": value,
            "death_probability": death,
            "end_hp": hp,
        },
        "exact_transition": {
            "hp_loss": hp_loss,
            "enemy_hp_loss": enemy_hp_loss,
            "enemies_killed": 0.0,
            "block_delta": 0.0,
            "energy_delta": 0.0,
            "hand_count_delta": 0.0,
            "potion_count_delta": potion_delta,
        },
    }


def _evaluation(candidate: dict, worlds: list[dict], *, value: float) -> dict:
    return {
        "candidate": candidate,
        "policy_probability": 0.2,
        "selection_eligible": True,
        "mean_value": value,
        "lower_tail_cvar": value - 0.1,
        "risk_adjusted_value": value - 0.05,
        "worlds": worlds,
    }


def test_paired_potion_proposal_reports_tactical_deltas_without_fake_confidence():
    potion_id = "POTION.BLOCK_POTION"
    use = _evaluation(
        {"candidate_id": "use:block", "action_type": "use_potion", "source_id": potion_id},
        [
            _world("a", value=0.8, death=0.0, hp=55, hp_loss=5, enemy_hp_loss=12, potion_delta=-1),
            _world("b", value=0.6, death=0.1, hp=45, hp_loss=15, enemy_hp_loss=8, potion_delta=-1),
        ],
        value=0.7,
    )
    hold = _evaluation(
        {"candidate_id": "play:defend", "action_type": "play_card", "source_id": "CARD.DEFEND"},
        [
            _world("a", value=0.4, death=0.2, hp=45, hp_loss=15, enemy_hp_loss=5),
            _world("b", value=0.2, death=0.4, hp=25, hp_loss=35, enemy_hp_loss=3),
        ],
        value=0.3,
    )
    proposal = build_paired_potion_proposal(
        potion_id=potion_id,
        use_report={"information_boundary": "visible", "evaluations": [use]},
        hold_report={"information_boundary": "visible", "evaluations": [hold]},
        state_fingerprint="state-1",
    )

    assert proposal["calibration_status"] == "uncalibrated_evidence_only"
    assert proposal["tactical_necessity"] is None
    assert proposal["estimate_confidence"] is None
    assert proposal["paired_world_count"] == 2
    assert proposal["tactical_evidence"]["combat_value_gain"] == pytest.approx(0.4)
    assert proposal["tactical_evidence"]["death_risk_reduction"] == pytest.approx(0.25)
    assert proposal["tactical_evidence"]["predicted_end_hp_gain"] == pytest.approx(15.0)
    assert proposal["tactical_evidence"]["exact_boundary_hp_saved"] == pytest.approx(15.0)
    assert proposal["tactical_evidence"]["exact_enemy_hp_loss_gain"] == pytest.approx(6.0)
    assert proposal["tactical_evidence"]["exact_potion_count_delta"] == pytest.approx(-1.0)
    assert proposal["evidence_diagnostics"] == {
        "engine_short_horizon_direction": "positive",
        "learned_leaf_value_direction": "positive",
        "direction_agreement": True,
        "requires_mechanic_or_deeper_search_review": False,
    }


def test_paired_potion_proposal_rejects_mismatched_rng_worlds():
    potion_id = "POTION.FIRE_POTION"
    use = _evaluation(
        {"candidate_id": "use:fire", "action_type": "use_potion", "source_id": potion_id},
        [_world("a", value=1, death=0, hp=50, hp_loss=0, enemy_hp_loss=20)],
        value=1.0,
    )
    hold = _evaluation(
        {"candidate_id": "end", "action_type": "end_turn", "source_id": ""},
        [_world("b", value=0, death=0, hp=50, hp_loss=0, enemy_hp_loss=0)],
        value=0.0,
    )
    with pytest.raises(ValueError, match="identical determinizations"):
        build_paired_potion_proposal(
            potion_id=potion_id,
            use_report={"evaluations": [use]},
            hold_report={"evaluations": [hold]},
            state_fingerprint="state-2",
        )


def test_paired_potion_proposal_can_select_a_controlled_target():
    potion_id = "POTION.FIRE_POTION"
    use_wrong = _evaluation(
        {
            "candidate_id": "use:fire:0", "action_type": "use_potion",
            "source_id": potion_id, "target_index": 0,
        },
        [_world("a", value=0.2, death=0, hp=50, hp_loss=0, enemy_hp_loss=10)],
        value=0.2,
    )
    use_wanted = _evaluation(
        {
            "candidate_id": "use:fire:1", "action_type": "use_potion",
            "source_id": potion_id, "target_index": 1,
        },
        [_world("a", value=0.1, death=0, hp=50, hp_loss=0, enemy_hp_loss=20)],
        value=0.1,
    )
    hold = _evaluation(
        {"candidate_id": "end", "action_type": "end_turn", "source_id": ""},
        [_world("a", value=0, death=0, hp=50, hp_loss=0, enemy_hp_loss=0)],
        value=0.0,
    )
    proposal = build_paired_potion_proposal(
        potion_id=potion_id,
        use_report={"evaluations": [use_wrong, use_wanted]},
        hold_report={"evaluations": [hold]},
        state_fingerprint="state-targeted",
        target_index=1,
    )
    assert proposal["use_candidate"]["candidate_id"] == "use:fire:1"
    assert proposal["requested_target_index"] == 1


def test_paired_evaluator_refuses_long_horizon_potion():
    with pytest.raises(ValueError, match="requires evaluator=persistent_rollout"):
        build_paired_potion_proposal(
            potion_id="POTION.STRENGTH_POTION",
            use_report={"evaluations": []},
            hold_report={"evaluations": []},
            state_fingerprint="state-3",
        )


def test_paired_potion_proposal_marks_unobservable_terminal_deltas_unknown():
    potion_id = "POTION.BLOCK_POTION"
    use_world = _world(
        "a", value=1, death=0, hp=2, hp_loss=3, enemy_hp_loss=6, potion_delta=-1
    )
    hold_world = _world("a", value=-5, death=1, hp=0, hp_loss=5, enemy_hp_loss=0)
    hold_world["exact_transition"].update({
        "enemy_state_observed": False,
        "energy_observed": False,
        "hand_observed": False,
    })
    use = _evaluation(
        {"candidate_id": "use:block", "action_type": "use_potion", "source_id": potion_id},
        [use_world],
        value=1.0,
    )
    hold = _evaluation(
        {"candidate_id": "end", "action_type": "end_turn", "source_id": ""},
        [hold_world],
        value=-5.0,
    )

    proposal = build_paired_potion_proposal(
        potion_id=potion_id,
        use_report={"evaluations": [use]},
        hold_report={"evaluations": [hold]},
        state_fingerprint="state-terminal",
    )

    evidence = proposal["tactical_evidence"]
    assert evidence["exact_boundary_hp_saved"] == pytest.approx(2.0)
    assert evidence["exact_enemy_hp_loss_gain"] is None
    assert evidence["exact_enemies_killed_gain"] is None
    assert evidence["exact_energy_delta_gain"] is None
    assert evidence["exact_hand_count_delta_gain"] is None


def test_paired_potion_proposal_flags_exact_value_direction_disagreement():
    potion_id = "POTION.FIRE_POTION"
    use = _evaluation(
        {
            "candidate_id": "use:fire", "action_type": "use_potion",
            "source_id": potion_id, "target_index": 0,
        },
        [_world("a", value=-1, death=0.1, hp=50, hp_loss=0, enemy_hp_loss=20)],
        value=-1.0,
    )
    hold = _evaluation(
        {"candidate_id": "end", "action_type": "end_turn", "source_id": ""},
        [_world("a", value=0, death=0, hp=47, hp_loss=3, enemy_hp_loss=0)],
        value=0.0,
    )
    proposal = build_paired_potion_proposal(
        potion_id=potion_id,
        use_report={"evaluations": [use]},
        hold_report={"evaluations": [hold]},
        state_fingerprint="state-conflict",
    )
    assert proposal["evidence_diagnostics"]["direction_agreement"] is False
    assert proposal["evidence_diagnostics"][
        "requires_mechanic_or_deeper_search_review"
    ] is True
