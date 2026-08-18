import pytest


torch = pytest.importorskip("torch")

from sts2_dataset.combat_model import CombatObjective, CombatPolicyConfig, CombatPolicyTransformer


def test_encounter_vocabulary_is_opt_in_for_base_models():
    vocabulary = {
        "entity_types": ["<PAD>", "global"],
        "entity_identity": ["<PAD>", "<UNK>", "global"],
        "action_types": ["<PAD>", "end_turn"],
        "target_kinds": ["<PAD>", "none"],
        "encounter_identity": ["<PAD>", "<UNK>", "encounter:MONSTER.TEST"],
        "numeric_feature_dim": 8,
        "categorical_feature_dim": 8,
    }
    assert CombatPolicyConfig.from_vocabulary(vocabulary).encounter_identity_count == 0
    configured = CombatPolicyConfig.from_vocabulary(
        vocabulary, encounter_embedding_adapter=True
    )
    assert configured.encounter_identity_count == 3


def _batch():
    return {
        "entity_type": torch.tensor([[1, 2, 6], [1, 2, 0]]),
        "entity_identity": torch.tensor([[2, 3, 4], [2, 3, 0]]),
        "entity_numeric": torch.zeros(2, 3, 8),
        "entity_categorical": torch.zeros(2, 3, 8),
        "entity_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "action_type": torch.tensor([[1, 4], [1, 0]]),
        "action_source": torch.tensor([[1, 0], [1, 0]]),
        "action_target": torch.tensor([[2, 0], [0, 0]]),
        "action_target_kind": torch.tensor([[3, 1], [2, 0]]),
        "candidate_engine_numeric": torch.zeros(2, 2, 18),
        "action_mask": torch.tensor([[True, True], [True, False]]),
        "label": torch.tensor([1, 0]),
        "resource_target_mask": torch.tensor([True, True]),
        "target_hp_loss_fraction": torch.tensor([0.25, 0.0]),
        "target_immediate_hp_loss_fraction": torch.tensor([0.1, 0.0]),
        "target_death": torch.tensor([0.0, 1.0]),
        "target_terminal_hp_fraction": torch.tensor([0.4, 0.0]),
        "target_potion_spent": torch.tensor([1.0, 0.0]),
        "target_max_hp_delta": torch.tensor([0.0, 1.0]),
    }


def test_combat_policy_scores_only_legal_dynamic_candidates():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = CombatPolicyTransformer(config)
    logits = model(_batch())
    assert logits.shape == (2, 2)
    assert torch.isneginf(logits[1, 1])
    loss = model.behavior_cloning_loss(_batch())
    assert torch.isfinite(loss)
    loss.backward()


def test_label_smoothing_does_not_assign_mass_to_padded_actions():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = CombatPolicyTransformer(config)
    loss = model.behavior_cloning_loss(_batch(), label_smoothing=0.05)
    assert torch.isfinite(loss)


def test_candidate_engine_features_are_fused_without_changing_action_shape():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        candidate_engine_feature_dim=18,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = CombatPolicyTransformer(config)
    batch = _batch()
    model.candidate_engine_scale.data.fill_(1.0)
    first = model(batch)
    batch["candidate_engine_numeric"][0, 0, 3] = 0.5
    second = model(batch)
    assert first.shape == second.shape == (2, 2)
    assert not torch.allclose(first[0, 0], second[0, 0])


def test_encounter_adapter_can_change_policy_without_changing_action_shape():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        encounter_identity_count=3,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = CombatPolicyTransformer(config).eval()
    batch = _batch()
    batch["encounter_identity"] = torch.tensor([1, 1])
    unknown_logits = model(batch)
    with torch.no_grad():
        model.encounter_embedding.weight[2].fill_(0.5)
    batch["encounter_identity"] = torch.tensor([2, 2])
    known_logits = model(batch)
    assert unknown_logits.shape == known_logits.shape == (2, 2)
    assert not torch.allclose(unknown_logits[0], known_logits[0])


def test_encounter_residual_adapter_changes_policy_without_resource_predictions():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        encounter_identity_count=3,
        encounter_residual_adapter=True,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        resource_value_heads=True,
    )
    model = CombatPolicyTransformer(config).eval()
    batch = _batch()
    batch["encounter_identity"] = torch.tensor([1, 1])
    unknown_logits = model(batch)
    unknown_resources = model.resource_predictions(batch)
    with torch.no_grad():
        model.encounter_embedding.weight[2].fill_(0.5)
    batch["encounter_identity"] = torch.tensor([2, 2])
    known_logits = model(batch)
    known_resources = model.resource_predictions(batch)
    assert not torch.allclose(unknown_logits[0], known_logits[0])
    assert all(
        torch.equal(unknown_resources[key], known_resources[key])
        for key in unknown_resources
    )


def test_policy_resource_heads_train_selected_human_action_and_rerank_legal_candidates():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        resource_value_heads=True,
        decision_value_scale=1.0,
    )
    model = CombatPolicyTransformer(config)
    predictions = model.resource_predictions(_batch())
    assert predictions["hp_loss_fraction"].shape == (2, 2)
    assert predictions["immediate_hp_loss_fraction"].shape == (2, 2)
    assert torch.all((predictions["hp_loss_fraction"] >= 0) & (predictions["hp_loss_fraction"] <= 1))
    decision_logits = model.decision_logits(_batch())
    assert decision_logits.shape == (2, 2)
    assert torch.isneginf(decision_logits[1, 1])
    losses = model.policy_resource_loss(_batch())
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    defensive_objective = CombatObjective.from_config(config, immediate_hp_loss_weight=40.0)
    overridden_logits = model.decision_logits(_batch(), objective=defensive_objective)
    assert overridden_logits.shape == decision_logits.shape


def test_state_value_head_predicts_from_state_without_selecting_an_action():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        state_value_head=True,
    )
    model = CombatPolicyTransformer(config)
    predictions = model.state_value_predictions(_batch())
    assert predictions["hp_loss_fraction"].shape == (2,)
    assert predictions["death_logit"].shape == (2,)
    losses = model.state_value_loss(_batch())
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()


def test_shared_entity_encoding_matches_independent_policy_and_state_outputs():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        candidate_engine_feature_dim=18,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        resource_value_heads=True,
        state_value_head=True,
        decision_value_scale=0.5,
    )
    model = CombatPolicyTransformer(config).eval()
    objective = CombatObjective.from_config(config)
    expected_policy, expected_resources, expected_decision = (
        model.policy_resource_outputs(_batch(), objective=objective)
    )
    expected_state = model.state_value_predictions(_batch())
    policy, resources, decision, state = model.policy_resource_state_outputs(
        _batch(), objective=objective
    )
    assert torch.equal(policy, expected_policy)
    assert torch.equal(decision, expected_decision)
    assert all(torch.equal(resources[key], expected_resources[key]) for key in resources)
    assert all(torch.equal(state[key], expected_state[key]) for key in state)


def test_distributional_state_outcome_has_death_bin_and_survivor_hp_bins():
    config = CombatPolicyConfig(
        entity_type_count=11,
        entity_identity_count=8,
        action_type_count=5,
        target_kind_count=5,
        numeric_feature_dim=8,
        categorical_feature_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        state_outcome_bins=21,
    )
    model = CombatPolicyTransformer(config)
    targets = model.state_outcome_targets(_batch())
    assert targets.tolist() == [9, 0]
    predictions = model.state_outcome_predictions(_batch())
    assert predictions["probabilities"].shape == (2, 21)
    assert torch.allclose(predictions["probabilities"].sum(dim=1), torch.ones(2))
    loss = model.state_outcome_loss(_batch(), death_weight=4.0)
    assert torch.isfinite(loss)
    loss.backward()
