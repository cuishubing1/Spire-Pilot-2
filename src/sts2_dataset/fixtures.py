from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import FIXTURE_ROOT
from .engine import Sts2Engine
from .legal_actions import enumerate_legal_actions
from .normalize import normalize_observation
from .policy import HeuristicPolicy
from .storage import RawRunWriter
from .util import sha256_file


FIXTURE_COMMANDS = [
    ("shop", {"cmd": "enter_room", "type": "shop"}),
    ("rest_site", {"cmd": "enter_room", "type": "rest_site"}),
    ("treasure", {"cmd": "enter_room", "type": "treasure"}),
    ("event_choice", {"cmd": "enter_room", "type": "event", "event": "MORPHIC_GROVE"}),
    ("combat_play", {"cmd": "enter_room", "type": "combat", "encounter": "SHRINKER_BEETLE_WEAK"}),
]


def collect_fixtures(config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for index, (phase, command) in enumerate(FIXTURE_COMMANDS, 1):
        run_id = f"fixture-{index:02d}-{phase}"
        target = FIXTURE_ROOT / f"{run_id}.jsonl.zst"
        if target.exists():
            results.append({"fixture_id": run_id, "status": "already_sealed", "sha256": sha256_file(target)})
            continue
        engine = Sts2Engine(config, run_id)
        policy = HeuristicPolicy(run_id)
        try:
            with RawRunWriter(target, run_id) as writer:
                raw = engine.reset(seed=run_id)
                raw = _advance_to_map(engine, raw, config, run_id, policy)
                if phase == "shop":
                    engine.raw_command({"cmd": "set_player", "gold": 999})
                response = engine.raw_command(command)
                payload = {"command": command, "response": response, "exchanges": engine.drain_exchanges()}
                writer.write("fixture", fixture_id=run_id, phase=phase, is_fixture=True, payload=payload)

                if phase == "shop" and response.get("decision") == "shop":
                    removal = next((a for a in enumerate_legal_actions(response) if a.action == "remove_card"), None)
                    if removal:
                        selected = engine.step(removal.to_dict())
                        writer.write(
                            "fixture",
                            fixture_id=run_id + "-card-select",
                            phase="card_select",
                            is_fixture=True,
                            payload={"source": "shop_card_removal", "response": selected, "exchanges": engine.drain_exchanges()},
                        )

                if phase == "combat_play":
                    state = response
                    for step in range(500):
                        if state.get("decision") != "combat_play":
                            break
                        obs = normalize_observation(state, config=config, run_id=run_id, step_id=step, audit_ref=None)
                        state = engine.step(policy.choose(obs.to_dict()))
                    resulting_phase = state.get("decision")
                    if resulting_phase in {"card_reward", "bundle_select", "card_select", "game_over"}:
                        writer.write(
                            "fixture",
                            fixture_id=run_id + "-post-combat",
                            phase=resulting_phase,
                            is_fixture=True,
                            payload={"source": "completed_weak_combat", "response": state, "exchanges": engine.drain_exchanges()},
                        )

                writer.seal()
            results.append({"fixture_id": run_id, "status": "sealed", "sha256": sha256_file(target)})
        finally:
            engine.close()
    results.append(_collect_bundle_fixture(config))
    results.append(_collect_boss_reward_fixture(config))
    return results


def _collect_bundle_fixture(config: dict[str, Any]) -> dict[str, Any]:
    run_id = "fixture-06-bundle_select"
    target = FIXTURE_ROOT / f"{run_id}.jsonl.zst"
    if target.exists():
        return {"fixture_id": run_id, "status": "already_sealed", "sha256": sha256_file(target)}
    engine = Sts2Engine(config, run_id)
    try:
        with RawRunWriter(target, run_id) as writer:
            # This locked seed is known to offer Scroll Boxes on v0.107.1.
            state = engine.reset(seed="smoke-v01071")
            option = next(
                (
                    o
                    for o in state.get("options", [])
                    if "SCROLL_BOXES" in str(o.get("text_key", ""))
                ),
                None,
            )
            if option is None:
                raise RuntimeError("Neow Scroll Boxes option not found")
            response = engine.step(
                {
                    "action_id": "fixture",
                    "action": "choose_option",
                    "args": {"option_index": option["index"]},
                }
            )
            if response.get("decision") != "bundle_select":
                raise RuntimeError(f"Expected bundle_select, got {response.get('decision')}")
            writer.write(
                "fixture",
                fixture_id=run_id,
                phase="bundle_select",
                is_fixture=True,
                payload={"source": "neow_scroll_boxes", "response": response, "exchanges": engine.drain_exchanges()},
            )
            writer.seal()
        return {"fixture_id": run_id, "status": "sealed", "sha256": sha256_file(target)}
    finally:
        engine.close()


def _collect_boss_reward_fixture(config: dict[str, Any]) -> dict[str, Any]:
    run_id = "fixture-07-boss_reward"
    target = FIXTURE_ROOT / f"{run_id}.jsonl.zst"
    if target.exists():
        return {"fixture_id": run_id, "status": "already_sealed", "sha256": sha256_file(target)}
    engine = Sts2Engine(config, run_id)
    policy = HeuristicPolicy(run_id)
    try:
        with RawRunWriter(target, run_id) as writer:
            state = engine.reset(seed=run_id)
            state = _advance_to_map(engine, state, config, run_id, policy)
            boss_id = engine.get_map()["boss"]["id"]
            engine.raw_command(
                {"cmd": "set_player", "hp": 999, "max_hp": 999, "deck": ["POMMEL_STRIKE"] * 10}
            )
            state = engine.raw_command({"cmd": "enter_room", "type": "combat", "encounter": boss_id})
            for step in range(2000):
                if state.get("decision") != "combat_play":
                    break
                obs = normalize_observation(state, config=config, run_id=run_id, step_id=step, audit_ref=None)
                state = engine.step(policy.choose(obs.to_dict()))
            else:
                raise RuntimeError("Boss fixture combat timed out")
            if state.get("decision") != "card_reward":
                raise RuntimeError(f"Expected boss card reward, got {state.get('decision')}")
            before = state
            obs = normalize_observation(state, config=config, run_id=run_id, step_id=2001, audit_ref=None)
            state = engine.step(policy.choose(obs.to_dict()))
            before_act = int((before.get("context") or {}).get("act") or 0)
            after_act = int((state.get("context") or {}).get("act") or 0)
            if after_act <= before_act:
                raise RuntimeError(f"Boss reward did not advance act: {before_act} -> {after_act}")
            writer.write(
                "fixture",
                fixture_id=run_id,
                phase="boss_reward",
                is_fixture=True,
                payload={
                    "source": "forced_real_boss_combat",
                    "boss_id": boss_id,
                    "before": before,
                    "after": state,
                    "exchanges": engine.drain_exchanges(),
                },
            )
            writer.seal()
        return {"fixture_id": run_id, "status": "sealed", "sha256": sha256_file(target)}
    finally:
        engine.close()


def _advance_to_map(engine, raw, config, run_id, policy):
    for step in range(50):
        if raw.get("decision") == "map_select":
            return raw
        obs = normalize_observation(raw, config=config, run_id=run_id, step_id=step, audit_ref=None)
        raw = engine.step(policy.choose(obs.to_dict()))
    raise RuntimeError(f"Fixture {run_id} could not reach map")
