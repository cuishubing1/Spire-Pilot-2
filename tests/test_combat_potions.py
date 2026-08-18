from sts2_dataset.combat_potions import (
    GAME_VERSION,
    POTION_SPECS,
    POTION_SPECS_BY_ID,
    validate_potion_catalog,
)


def test_v01071_ironclad_potion_catalog_is_complete_and_unique():
    assert GAME_VERSION == "0.107.1"
    assert validate_potion_catalog() == {
        "total": 51,
        "shared": 45,
        "ironclad": 3,
        "shared_special": 3,
        "stochastic": 14,
    }
    assert len(POTION_SPECS_BY_ID) == len(POTION_SPECS)


def test_ironclad_exclusive_potions_have_distinct_evaluation_modes():
    assert POTION_SPECS_BY_ID["POTION.BLOOD_POTION"].evaluator == "upper_resource"
    assert POTION_SPECS_BY_ID["POTION.ASHWATER"].choice_mode == "hand_subset"
    assert POTION_SPECS_BY_ID["POTION.SOLDIERS_STEW"].horizon == "combat_end"


def test_complex_shared_potions_are_not_treated_as_immediate_scalar_effects():
    expected = {
        "POTION.ATTACK_POTION": ("nested_choice", "random_offer_one"),
        "POTION.GAMBLERS_BREW": ("nested_choice", "hand_subset"),
        "POTION.ENTROPIC_BREW": ("sampled_rollout", "random_generation"),
        "POTION.SNECKO_OIL": ("sampled_rollout", "random_generation"),
        "POTION.FAIRY_IN_A_BOTTLE": ("passive_reserve", "automatic"),
    }
    for potion_id, (evaluator, choice_mode) in expected.items():
        spec = POTION_SPECS_BY_ID[potion_id]
        assert (spec.evaluator, spec.choice_mode) == (evaluator, choice_mode)
