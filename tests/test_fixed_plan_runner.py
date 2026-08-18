from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_fixed_plan_policy as runner  # noqa: E402


def _ranked():
    return [
        {
            "candidate": {"candidate_id": "p1-best", "action_type": "end_turn"},
            "probability": 0.75,
            "policy_probability": 0.75,
        },
        {
            "candidate": {"candidate_id": "p1-second", "action_type": "play_card"},
            "probability": 0.25,
            "policy_probability": 0.25,
        },
    ]


def test_search_failure_falls_back_to_auditable_p1_choice(monkeypatch):
    monkeypatch.setattr(
        runner,
        "headless_state_to_model_sample",
        lambda *args, **kwargs: {"sample": True},
    )
    monkeypatch.setattr(
        runner,
        "_rank_actions",
        lambda *args, **kwargs: (_ranked(), 1.25),
    )

    report = runner._search_failure_policy_lookahead(
        mode="one_step",
        error=runner.EngineError("Not in combat"),
        state={"decision": "combat_play"},
        run_id="run-a",
        decision_index=17,
        combat_index=2,
        model=object(),
        tensorizer=object(),
        device="cuda",
        objective=object(),
    )

    assert report["status"] == "engine_restore_fallback"
    assert report["fallback_policy"] == "p1"
    assert report["chosen_candidate"]["candidate_id"] == "p1-best"
    assert report["policy_candidate"] == report["chosen_candidate"]
    assert report["successor_value_inference_ms"] == 0.0
    assert report["error"] == "Not in combat"


def test_turn_boundary_fallback_has_zero_search_expansion(monkeypatch):
    monkeypatch.setattr(
        runner,
        "headless_state_to_model_sample",
        lambda *args, **kwargs: {"sample": True},
    )
    monkeypatch.setattr(
        runner,
        "_rank_actions",
        lambda *args, **kwargs: (_ranked(), 2.5),
    )

    report = runner._search_failure_policy_lookahead(
        mode="turn_boundary",
        error=runner.EngineError("root mismatch"),
        state={"decision": "combat_play"},
        run_id="run-b",
        decision_index=3,
        combat_index=0,
        model=object(),
        tensorizer=object(),
        device="cuda",
        objective=object(),
    )

    assert report["value_inference_ms"] == 0.0
    assert report["expanded_paths"] == 0
    assert report["root_inference_ms"] == 2.5


def test_adaptive_turn_boundary_uses_visible_loss_and_current_hp(monkeypatch):
    monkeypatch.setattr(
        runner,
        "headless_state_to_model_sample",
        lambda *args, **kwargs: {"observation": {"global": {"hp": 20}}},
    )
    monkeypatch.setattr(
        runner,
        "visible_intent_end_turn_hp_loss",
        lambda observation: {"hp_loss": 10},
    )

    result = runner._adaptive_turn_boundary_trigger(
        {"decision": "combat_play"},
        minimum_hp_loss=8,
        minimum_hp_fraction=0.4,
    )

    assert result["triggered"] is True
    assert result["current_hp_fraction"] == 0.5


def test_adaptive_turn_boundary_requires_both_thresholds(monkeypatch):
    monkeypatch.setattr(
        runner,
        "headless_state_to_model_sample",
        lambda *args, **kwargs: {"observation": {"global": {"hp": 60}}},
    )
    monkeypatch.setattr(
        runner,
        "visible_intent_end_turn_hp_loss",
        lambda observation: {"hp_loss": 10},
    )

    result = runner._adaptive_turn_boundary_trigger(
        {"decision": "combat_play"},
        minimum_hp_loss=8,
        minimum_hp_fraction=0.4,
    )

    assert result["triggered"] is False
