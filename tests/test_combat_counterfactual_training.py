import torch

from sts2_dataset.combat_counterfactual_training import (
    pairwise_ranking_loss,
    pairwise_ranking_metrics,
    split_examples_by_scenario,
    teacher_best_action_loss,
)


def _example(scenario_id, *, winner=1, suboptimal=True):
    return {
        "scenario_id": scenario_id,
        "best_candidate_id": "b",
        "policy_suboptimal": suboptimal,
        "actions": [
            {"candidate_id": "a", "candidate_index": 0},
            {"candidate_id": "b", "candidate_index": 1},
        ],
        "pairwise_labels": [
            {
                "left_candidate_index": 0,
                "right_candidate_index": 1,
                "winner_candidate_index": winner,
                "mean_utility_delta_left_minus_right": -0.5,
            }
        ],
    }


def test_counterfactual_split_keeps_scenarios_disjoint():
    examples = [_example("one"), _example("one"), _example("two"), _example("three")]
    train, validation = split_examples_by_scenario(examples)
    assert {row["scenario_id"] for row in train}.isdisjoint(
        {row["scenario_id"] for row in validation}
    )
    assert len(train) + len(validation) == len(examples)


def test_pairwise_loss_and_metrics_reward_teacher_order():
    examples = [_example("one")]
    good = torch.tensor([[0.0, 2.0]], requires_grad=True)
    bad = torch.tensor([[2.0, 0.0]])
    assert pairwise_ranking_loss(good, examples) < pairwise_ranking_loss(bad, examples)
    pairwise_ranking_loss(good, examples).backward()
    assert good.grad is not None
    metrics = pairwise_ranking_metrics(good.detach(), examples)
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["policy_suboptimal_top1_accuracy"] == 1.0
    assert teacher_best_action_loss(good, examples) < teacher_best_action_loss(bad, examples)
