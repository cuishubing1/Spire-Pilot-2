from __future__ import annotations

import uuid
from typing import Any

from .engine import Sts2Engine
from .normalize import normalize_observation
from .policy import HeuristicPolicy
from .util import utc_now


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    run_id = f"smoke-v01071-{uuid.uuid4().hex[:8]}"
    engine = Sts2Engine(config, run_id)
    policy = HeuristicPolicy("smoke-v01071")
    checks: list[str] = []
    try:
        raw = engine.reset(seed="smoke-v01071")
        checks.append("start_run")
        raw = _advance_to_map(engine, raw, config, run_id, policy)
        checks.append("map_select")
        raw = engine.get_state()
        visible_map = engine.get_map()
        initial = normalize_observation(raw, config=config, run_id=run_id, step_id=0, audit_ref=None, visible_map=visible_map)
        audit = engine.snapshot(0, initial.state_hash)
        checks.append("checkpoint_write")
        restored_raw = engine.restore(audit)
        restored_raw = engine.get_state()
        restored_map = engine.get_map()
        restored = normalize_observation(
            restored_raw, config=config, run_id=run_id, step_id=0, audit_ref=None, visible_map=restored_map
        )
        if restored.state_hash != initial.state_hash:
            raise RuntimeError(f"Map restore hash mismatch: {initial.state_hash} != {restored.state_hash}")
        checks.append("checkpoint_restore")

        raw = engine.raw_command({"cmd": "enter_room", "type": "combat", "encounter": "SHRINKER_BEETLE_WEAK"})
        if raw.get("decision") != "combat_play":
            raise RuntimeError(f"Expected combat_play, got {raw}")
        checks.append("enter_combat")
        combat = normalize_observation(raw, config=config, run_id=run_id, step_id=1, audit_ref=None)
        action = policy.choose(combat.to_dict())
        raw = engine.step(action)
        checks.append("legal_combat_action")

        for step in range(2, 502):
            if raw.get("decision") != "combat_play":
                break
            obs = normalize_observation(raw, config=config, run_id=run_id, step_id=step, audit_ref=None)
            raw = engine.step(policy.choose(obs.to_dict()))
        else:
            raise RuntimeError("Smoke combat did not terminate in 500 decisions")
        checks.append("combat_terminated")

        raw = engine.get_state()
        final = normalize_observation(raw, config=config, run_id=run_id, step_id=502, audit_ref=None)
        final_audit = engine.snapshot(502, final.state_hash)
        replayed_raw = engine.restore(final_audit)
        replayed_raw = engine.get_state()
        replayed_map = engine.get_map() if replayed_raw.get("decision") == "map_select" else None
        replayed = normalize_observation(
            replayed_raw, config=config, run_id=run_id, step_id=502, audit_ref=None, visible_map=replayed_map
        )
        if replayed.state_hash != final.state_hash:
            raise RuntimeError("Post-combat deterministic replay hash mismatch")
        checks.append("post_combat_restore")
        return {"status": "PASS", "checked_at": utc_now(), "checks": checks, "final_phase": final.phase}
    finally:
        engine.close()


def _advance_to_map(
    engine: Sts2Engine,
    raw: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
    policy: HeuristicPolicy,
) -> dict[str, Any]:
    for step in range(50):
        if raw.get("decision") == "map_select":
            return raw
        obs = normalize_observation(raw, config=config, run_id=run_id, step_id=step, audit_ref=None)
        if obs.terminal:
            raise RuntimeError("Run terminated before reaching the map")
        raw = engine.step(policy.choose(obs.to_dict()))
    raise RuntimeError("Could not advance initial event to map_select")
