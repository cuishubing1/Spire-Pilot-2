from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .constants import RAW_ROOT, SEEDS_PATH
from .engine import Sts2Engine
from .normalize import normalize_observation, outcome_delta
from .policy import HeuristicPolicy
from .storage import RawRunWriter
from .types import ObservationEnvelope
from .util import load_json, sha256_file, utc_now


def load_seeds(path: Path = SEEDS_PATH) -> list[str]:
    seeds = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed file contains duplicates")
    return seeds


def run_id_for(index: int, seed: str, fixture: bool = False) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"{'fixture' if fixture else 'natural'}-{index:03d}-{digest}"


class Collector:
    def __init__(self, config: dict[str, Any], engine_factory: Callable[[dict[str, Any], str], Sts2Engine] = Sts2Engine):
        self.config = config
        self.engine_factory = engine_factory

    def collect_many(self, count: int) -> list[dict[str, Any]]:
        seeds = load_seeds()
        if count > len(seeds):
            raise ValueError(f"Requested {count} runs but only {len(seeds)} seeds are locked")
        summaries = []
        for index, seed in enumerate(seeds[:count], 1):
            run_id = run_id_for(index, seed)
            target = RAW_ROOT / f"{run_id}.jsonl.zst"
            if target.exists():
                summaries.append({"run_id": run_id, "status": "already_sealed", "sha256": sha256_file(target)})
                continue
            summaries.append(self.collect_one(index=index, seed=seed))
        return summaries

    def collect_one(self, *, index: int, seed: str) -> dict[str, Any]:
        run_id = run_id_for(index, seed)
        target = RAW_ROOT / f"{run_id}.jsonl.zst"
        engine = self.engine_factory(self.config, run_id)
        policy = HeuristicPolicy(seed)
        max_act = max_floor = 0
        transition_count = 0
        started_at = utc_now()
        try:
            with RawRunWriter(target, run_id) as writer:
                writer.write(
                    "run_start",
                    dataset_version=self.config["dataset_version"],
                    seed=seed,
                    character=self.config["character"],
                    ascension=self.config["ascension"],
                    policy_id=policy.policy_id,
                    is_fixture=False,
                )
                raw = engine.reset(seed=seed)
                self._write_exchanges(writer, engine, step_id=0)
                current = self._materialize(engine, raw, run_id, 0)
                self._write_exchanges(writer, engine, step_id=0)

                while True:
                    envelope = current.to_dict()
                    context = current.context
                    max_act = max(max_act, int(context.get("act") or 0))
                    max_floor = max(max_floor, int(context.get("floor") or 0))
                    if current.terminal:
                        writer.write("decision", step_id=current.step_id, observation=envelope, action=None, transition=None)
                        victory = bool(current.agent_observation.get("screen", {}).get("victory"))
                        break
                    if transition_count >= int(self.config["max_steps"]):
                        raise RuntimeError(f"Run exceeded max_steps={self.config['max_steps']}")

                    action = policy.choose(envelope)
                    legal_ids = {item["action_id"] for item in current.legal_actions}
                    if action["action_id"] not in legal_ids:
                        raise RuntimeError(f"Policy emitted illegal action {action['action_id']}")

                    self._write_auto_transition_if_needed(writer, current, action)
                    next_raw = engine.step(action)
                    self._write_exchanges(writer, engine, step_id=current.step_id)
                    following = self._materialize(engine, next_raw, run_id, current.step_id + 1)
                    self._write_exchanges(writer, engine, step_id=following.step_id)
                    transition = {
                        "transition_id": f"{run_id}:{current.step_id:06d}",
                        "obs_t": envelope,
                        "legal_actions_t": current.legal_actions,
                        "action_t": action,
                        "outcome": outcome_delta(current, following),
                        "obs_t1": following.to_dict(),
                        "done": following.terminal,
                    }
                    writer.write("decision", step_id=current.step_id, observation=envelope, action=action, transition=transition)
                    transition_count += 1
                    current = following

                writer.write(
                    "run_end",
                    seed=seed,
                    character=self.config["character"],
                    ascension=self.config["ascension"],
                    policy_id=policy.policy_id,
                    is_fixture=False,
                    started_at=started_at,
                    ended_at=utc_now(),
                    transitions=transition_count,
                    terminal=True,
                    victory=victory,
                    max_act=max_act,
                    max_floor=max_floor,
                )
                sealed_path, digest = writer.seal()
            return {
                "run_id": run_id,
                "status": "sealed",
                "path": str(sealed_path),
                "sha256": digest,
                "transitions": transition_count,
                "victory": victory,
                "max_act": max_act,
                "max_floor": max_floor,
            }
        finally:
            engine.close()

    def _materialize(
        self, engine: Sts2Engine, raw: dict[str, Any], run_id: str, step_id: int
    ) -> ObservationEnvelope:
        # Action responses can be transient in v0.107.1 (notably Neow HP costs).
        # Re-export after pumping pending continuations before hashing or saving.
        raw = engine.get_state()
        visible_map = engine.get_map() if raw.get("decision") == "map_select" else None
        preliminary = normalize_observation(
            raw, config=self.config, run_id=run_id, step_id=step_id, audit_ref=None, visible_map=visible_map
        )
        audit = engine.snapshot(step_id, preliminary.state_hash)
        return normalize_observation(
            raw, config=self.config, run_id=run_id, step_id=step_id, audit_ref=audit, visible_map=visible_map
        )

    @staticmethod
    def _write_exchanges(writer: RawRunWriter, engine: Sts2Engine, *, step_id: int) -> None:
        for exchange in engine.drain_exchanges():
            writer.write("engine_exchange", step_id=step_id, **exchange)

    @staticmethod
    def _write_auto_transition_if_needed(
        writer: RawRunWriter, current: ObservationEnvelope, action: dict[str, Any]
    ) -> None:
        if current.phase != "map_select" or action["action"] != "select_map_node":
            return
        room_type = str((action.get("source") or {}).get("type", ""))
        if "Treasure" in room_type:
            writer.write(
                "auto_transition",
                step_id=current.step_id,
                phase="treasure",
                source_action_id=action["action_id"],
                note="Treasure is auto-resolved by the headless engine",
            )
