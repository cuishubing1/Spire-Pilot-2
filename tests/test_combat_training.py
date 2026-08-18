import pytest

from sts2_dataset.combat_training import _limit_by_combat


def test_smoke_limit_round_robins_across_combats():
    rows = [
        {"combat_id": "a", "value": index} for index in range(5)
    ] + [
        {"combat_id": "b", "value": index} for index in range(5)
    ]
    selected = _limit_by_combat(rows, 4, seed=7)
    assert len(selected) == 4
    assert {row["combat_id"] for row in selected} == {"a", "b"}


def test_training_module_keeps_torch_optional_for_data_pipeline():
    # Importing the module must not require torch; only the training entrypoint does.
    from sts2_dataset import combat_training

    assert combat_training.TRAINING_SCHEMA_VERSION == "combat-policy-training-0.1.0"
