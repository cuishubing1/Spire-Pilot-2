from sts2_dataset.combat_counterfactual import (
    build_counterfactual_training_examples,
    counterfactual_gate_variants,
    build_pairwise_labels,
    on_policy_trigger_reasons,
    select_counterfactual_roots,
    summarize_counterfactual_action,
    summarize_counterfactual_root,
)


def test_on_policy_trigger_reasons_combines_support_and_risk():
    assert on_policy_trigger_reasons(
        hp_ratio=0.3,
        round_number=7,
        exact_encounter_round_p95=5,
        incoming_hp_loss=12,
        policy_entropy=0.6,
        policy_margin=0.1,
        chosen_action_train_count=2,
    ) == [
        "low_hp",
        "late_round_ood",
        "high_visible_incoming_loss",
        "high_policy_entropy",
        "low_policy_margin",
        "rare_chosen_action",
    ]


def test_counterfactual_gate_requires_combined_evidence():
    incoming = counterfactual_gate_variants(["high_visible_incoming_loss"])
    assert incoming["any_trigger"] is True
    assert incoming["strict"] is False
    combined = counterfactual_gate_variants(
        ["high_visible_incoming_loss", "high_policy_entropy", "low_policy_margin"]
    )
    assert combined["risk_and_uncertainty"] is True
    assert combined["strict"] is True


def test_select_counterfactual_roots_keeps_high_risk_first():
    roots = [
        {"step": 0, "hp_ratio": 0.9, "round": 1, "trigger_reasons": ["high_policy_entropy"]},
        {"step": 3, "hp_ratio": 0.2, "round": 4, "trigger_reasons": ["low_hp"]},
    ]
    selected = select_counterfactual_roots(roots, combat_failure=False, limit=1)
    assert selected[0]["step"] == 3
    assert selected[0]["priority_score"] > 4


def test_select_counterfactual_roots_adds_earliest_trigger_for_diversity():
    roots = [
        {"step": 0, "hp_ratio": 0.9, "round": 1, "trigger_reasons": []},
        {"step": 2, "hp_ratio": 0.8, "round": 1, "trigger_reasons": ["low_policy_margin"]},
        {"step": 8, "hp_ratio": 0.3, "round": 5, "trigger_reasons": ["low_hp"]},
        {"step": 9, "hp_ratio": 0.1, "round": 6, "trigger_reasons": ["low_hp"]},
    ]
    selected = select_counterfactual_roots(roots, combat_failure=True, limit=2)
    assert selected[0]["step"] == 9
    assert selected[1]["step"] == 2


def test_select_counterfactual_roots_supports_earliest_only_strategy():
    roots = [
        {"step": 2, "hp_ratio": 0.8, "round": 1, "trigger_reasons": ["low_policy_margin"]},
        {"step": 8, "hp_ratio": 0.2, "round": 6, "trigger_reasons": ["low_hp"]},
    ]
    selected = select_counterfactual_roots(
        roots, combat_failure=True, limit=1, strategy="earliest"
    )
    assert selected[0]["step"] == 2


def test_pairwise_labels_only_use_shared_terminal_worlds():
    first = summarize_counterfactual_action(
        {"candidate_id": "a"},
        [
            {
                "determinization_id": "w1",
                "utility": -0.1,
                "terminal": True,
                "death": False,
                "terminal_hp": 40,
                "hp_loss": 5,
                "potion_spent": 0,
            },
            {
                "determinization_id": "w2",
                "utility": -0.2,
                "terminal": True,
                "death": False,
                "terminal_hp": 35,
                "hp_loss": 10,
                "potion_spent": 0,
            },
        ],
        cvar_alpha=0.5,
    )
    second = summarize_counterfactual_action(
        {"candidate_id": "b"},
        [
            {
                "determinization_id": "w1",
                "utility": -0.4,
                "terminal": True,
                "death": False,
                "terminal_hp": 20,
                "hp_loss": 25,
                "potion_spent": 0,
            },
            {
                "determinization_id": "w2",
                "utility": -0.6,
                "terminal": True,
                "death": False,
                "terminal_hp": 10,
                "hp_loss": 35,
                "potion_spent": 0,
            },
        ],
        cvar_alpha=0.5,
    )
    labels = build_pairwise_labels([first, second])
    assert labels[0]["winner_candidate_id"] == "a"
    assert labels[0]["shared_determinization_count"] == 2


def test_counterfactual_root_summary_marks_policy_regret():
    actions = [
        {
            "candidate": {"candidate_id": "policy"},
            "teacher_eligible": True,
            "mean_utility": -0.4,
            "policy_probability": 0.7,
        },
        {
            "candidate": {"candidate_id": "better"},
            "teacher_eligible": True,
            "mean_utility": -0.1,
            "policy_probability": 0.2,
        },
        {
            "candidate": {"candidate_id": "worse"},
            "teacher_eligible": True,
            "mean_utility": -0.6,
            "policy_probability": 0.1,
        },
    ]
    summary = summarize_counterfactual_root(actions)
    assert summary["informative"] is True
    assert summary["policy_candidate_id"] == "policy"
    assert summary["best_candidate_id"] == "better"
    assert summary["policy_suboptimal"] is True
    assert abs(summary["policy_utility_regret"] - 0.3) < 1e-12


def test_counterfactual_training_examples_filter_and_index_pairs():
    root = {
        "teacher_eligible": True,
        "informative": True,
        "scenario_id": "train-fight",
        "root_fingerprint": "root-a",
        "step": 2,
        "trigger_reasons": ["low_policy_margin"],
        "determinization_count": 2,
        "continuation_policy": "frozen_p2_argmax",
        "policy_candidate_id": "a",
        "best_candidate_id": "b",
        "policy_suboptimal": True,
        "policy_utility_regret": 0.3,
        "utility_range": 0.5,
        "root_sample": {
            "candidates": [
                {"candidate_id": "a", "candidate_index": 0},
                {"candidate_id": "b", "candidate_index": 1},
            ]
        },
        "actions": [
            {
                "candidate": {"candidate_id": "a"},
                "policy_probability": 0.8,
                "mean_utility": -0.4,
                "lower_tail_cvar": -0.5,
                "death_probability": 0.0,
                "mean_hp_loss": 10.0,
                "mean_potion_spent": 0.0,
            },
            {
                "candidate": {"candidate_id": "b"},
                "policy_probability": 0.2,
                "mean_utility": -0.1,
                "lower_tail_cvar": -0.2,
                "death_probability": 0.0,
                "mean_hp_loss": 4.0,
                "mean_potion_spent": 0.0,
            },
        ],
        "pairwise_labels": [
            {
                "left_candidate_id": "a",
                "right_candidate_id": "b",
                "winner_candidate_id": "b",
                "tie": False,
                "mean_utility_delta_left_minus_right": -0.3,
                "shared_determinization_count": 2,
            }
        ],
    }
    report = {
        "dataset_split": "train",
        "combats": [
            {
                "act": 2,
                "ascension": 5,
                "encounter": "TEST",
                "teacher_roots": [root, {**root, "informative": False}],
            }
        ],
    }
    examples = build_counterfactual_training_examples(report)
    assert len(examples) == 1
    assert examples[0]["policy_suboptimal"] is True
    assert examples[0]["pairwise_labels"][0]["winner_candidate_index"] == 1
    assert examples[0]["example_id"].startswith("counterfactual-")


def test_counterfactual_training_examples_reject_validation_report():
    try:
        build_counterfactual_training_examples(
            {"dataset_split": "validation", "combats": []}
        )
    except ValueError as exc:
        assert "train-split" in str(exc)
    else:
        raise AssertionError("validation counterfactual report was accepted")
