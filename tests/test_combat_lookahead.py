import math

import pytest

from sts2_dataset.combat_lookahead import (
    apply_exact_terminal_death_veto,
    apply_policy_advantage_gate,
    choose_one_step_candidate,
    one_step_takeover_ineligibility,
    policy_top_k,
    required_search_categories,
    regularized_one_step_score,
)


def _row(
    index,
    probability,
    score=0.0,
    eligible=True,
    *,
    equivalence_key=None,
    category=None,
):
    candidate = {"candidate_id": f"a{index}", "candidate_index": index}
    if equivalence_key is not None:
        candidate["search_equivalence_key"] = equivalence_key
    if category is not None:
        candidate["search_category"] = category
    return {
        "candidate": candidate,
        "policy_probability": probability,
        "selection_score": score,
        "selection_eligible": eligible,
    }


def test_policy_top_k_is_probability_ordered_and_stable_on_ties():
    rows = [_row(2, 0.2), _row(1, 0.6), _row(0, 0.2)]
    assert [row["candidate"]["candidate_id"] for row in policy_top_k(rows, 2)] == [
        "a1", "a0"
    ]


def test_policy_top_k_merges_semantic_duplicates_but_keeps_raw_argmax_first():
    rows = [
        _row(0, 0.40, equivalence_key="strike"),
        _row(1, 0.35, equivalence_key="defend"),
        _row(2, 0.25, equivalence_key="defend"),
    ]

    selected = policy_top_k(rows, 2)

    assert [row["candidate"]["candidate_id"] for row in selected] == ["a0", "a1"]
    assert selected[1]["policy_probability"] == pytest.approx(0.60)
    assert selected[1]["candidate"]["equivalent_candidate_count"] == 2
    assert selected[1]["candidate"]["equivalent_candidate_ids"] == ["a1", "a2"]


def test_policy_top_k_reserves_a_required_category_without_replacing_argmax():
    rows = [
        _row(0, 0.60, equivalence_key="strike", category="card_attack"),
        _row(1, 0.25, equivalence_key="bash", category="card_attack"),
        _row(2, 0.10, equivalence_key="end", category="end_turn"),
        _row(3, 0.05, equivalence_key="defend", category="card_block"),
    ]

    selected = policy_top_k(rows, 3, required_categories=("card_block",))

    assert selected[0]["candidate"]["candidate_id"] == "a0"
    assert [row["candidate"]["candidate_id"] for row in selected] == ["a0", "a1", "a3"]


def test_visible_incoming_damage_requires_a_block_branch_only_when_uncovered():
    observation = {
        "global": {"max_hp": 80, "block": 3},
        "enemies": [{"hp": 20, "intent": [{"type": "Attack", "damage": 7}]}],
    }
    assert required_search_categories(observation) == ("card_block",)
    observation["global"]["block"] = 7
    assert required_search_categories(observation) == ()


def test_one_step_score_retains_a_configurable_policy_prior():
    assert regularized_one_step_score(
        value=-0.1, policy_probability=0.5, policy_log_weight=0.2
    ) == pytest.approx(-0.1 + 0.2 * math.log(0.5))
    with pytest.raises(ValueError):
        regularized_one_step_score(value=0.0, policy_probability=0.5, policy_log_weight=-1.0)


def test_one_step_selection_ignores_unsupported_candidates():
    chosen = choose_one_step_candidate([
        _row(0, 0.7, score=1.0, eligible=False),
        _row(1, 0.2, score=0.4),
        _row(2, 0.1, score=0.4),
    ])
    assert chosen["candidate"]["candidate_id"] == "a1"


def test_policy_advantage_gate_rejects_only_small_overrides():
    policy = _row(0, 0.7, score=0.30)
    weak = _row(1, 0.2, score=0.31)
    strong = _row(2, 0.1, score=0.36)
    chosen, fallback = apply_policy_advantage_gate(
        search_choice=weak, policy_choice=policy, minimum_advantage=0.02
    )
    assert chosen is policy
    assert fallback["reason"] == "insufficient_value_advantage"
    chosen, fallback = apply_policy_advantage_gate(
        search_choice=strong, policy_choice=policy, minimum_advantage=0.02
    )
    assert chosen is strong
    assert fallback is None


def test_policy_advantage_gate_requires_stronger_evidence_to_end_turn_early():
    policy = _row(0, 0.5, score=0.0)
    policy["candidate"]["action_type"] = "play_card"
    end_turn = _row(1, 0.4, score=0.12)
    end_turn["candidate"]["action_type"] = "end_turn"
    chosen, fallback = apply_policy_advantage_gate(
        search_choice=end_turn,
        policy_choice=policy,
        minimum_advantage=0.02,
        minimum_end_turn_advantage=0.15,
    )
    assert chosen is policy
    assert fallback["minimum_advantage"] == 0.15


def test_policy_advantage_gate_never_restores_ineligible_policy_candidate():
    policy = _row(0, 0.8, score=1.0, eligible=False)
    search = _row(1, 0.2, score=0.0)

    chosen, fallback = apply_policy_advantage_gate(
        search_choice=search,
        policy_choice=policy,
        minimum_advantage=0.5,
    )

    assert chosen is search
    assert fallback["reason"] == "policy_candidate_ineligible"


def test_one_step_takeover_gate_preserves_policy_candidate():
    evaluation = _row(0, 0.001)
    evaluation["candidate"]["action_type"] = "discard_potion"
    evaluation["worlds"] = [{"exact_transition": {"hp_loss": 10}}]

    assert one_step_takeover_ineligibility(
        evaluation,
        policy_candidate_id="a0",
        minimum_potion_policy_probability=0.02,
    ) == ()


def test_one_step_takeover_gate_rejects_unsupported_resource_actions():
    discard = _row(1, 0.01)
    discard["candidate"]["action_type"] = "discard_potion"
    discard["worlds"] = []
    potion = _row(2, 0.01)
    potion["candidate"]["action_type"] = "use_potion"
    potion["worlds"] = []

    assert one_step_takeover_ineligibility(
        discard,
        policy_candidate_id="a0",
        minimum_potion_policy_probability=0.02,
    ) == ("search_cannot_introduce_potion_discard",)
    assert one_step_takeover_ineligibility(
        potion,
        policy_candidate_id="a0",
        minimum_potion_policy_probability=0.02,
    ) == ("search_potion_below_policy_support_floor",)


def test_one_step_takeover_gate_rejects_wasted_transient_block_potion():
    potion = _row(2, 0.2)
    potion["candidate"].update({
        "action_type": "use_potion",
        "source_id": "POTION.BLOCK_POTION",
    })
    potion["root_visible_end_turn_hp_loss"] = 0.0
    potion["root_retains_block"] = False
    potion["worlds"] = [{"exact_transition": {"block_delta": 12.0}}]

    assert one_step_takeover_ineligibility(
        potion,
        policy_candidate_id="a0",
        minimum_potion_policy_probability=0.0,
    ) == ("transient_block_potion_without_visible_attack",)

    potion["root_retains_block"] = True
    assert one_step_takeover_ineligibility(
        potion,
        policy_candidate_id="a0",
        minimum_potion_policy_probability=0.0,
    ) == ()


def test_one_step_takeover_gate_rejects_damaging_early_end_turn():
    evaluation = _row(1, 0.1)
    evaluation["candidate"]["action_type"] = "end_turn"
    evaluation["worlds"] = [
        {"exact_transition": {"hp_loss": 0}},
        {"exact_transition": {"hp_loss": 11}},
    ]

    assert one_step_takeover_ineligibility(
        evaluation,
        policy_candidate_id="a0",
        minimum_potion_policy_probability=0.02,
    ) == ("one_step_end_turn_has_exact_hp_loss",)


def test_exact_terminal_death_is_vetoed_when_an_engine_survivor_exists():
    death = _row(0, 0.7, score=10.0)
    death["worlds"] = [
        {"outcome": {"terminal": True, "death_probability": 1.0}}
    ]
    survivor = _row(1, 0.3, score=-10.0)
    survivor["worlds"] = [
        {"outcome": {"terminal": False, "death_probability": 0.99}}
    ]

    assert apply_exact_terminal_death_veto([death, survivor]) == 1
    assert death["selection_eligible"] is False
    assert death["selection_ineligible_reasons"] == ["exact_terminal_death_veto"]
    assert survivor["selection_eligible"] is True


def test_exact_terminal_death_veto_does_not_invent_a_survivor():
    first = _row(0, 0.7)
    second = _row(1, 0.3)
    for row in (first, second):
        row["worlds"] = [
            {"outcome": {"terminal": True, "death_probability": 1.0}}
        ]

    assert apply_exact_terminal_death_veto([first, second]) == 0
    assert first["selection_eligible"] is True
    assert second["selection_eligible"] is True
