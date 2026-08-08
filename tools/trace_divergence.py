"""Replay one room from a map checkpoint and stop at the first state divergence."""

import sys

from sts2_dataset.constants import CONFIG_PATH, RAW_ROOT
from sts2_dataset.engine import Sts2Engine
from sts2_dataset.normalize import normalize_observation
from sts2_dataset.types import AuditRef
from sts2_dataset.util import iter_jsonl_zst, load_json

run_id = sys.argv[1]
start_step = int(sys.argv[2])
records = {
    r["step_id"]: r
    for r in iter_jsonl_zst(RAW_ROOT / f"{run_id}.jsonl.zst")
    if r.get("record_type") == "decision" and r.get("transition")
}
config = load_json(CONFIG_PATH)
engine = Sts2Engine(config, "trace-divergence")
try:
    expected = records[start_step]["observation"]
    raw = engine.restore(AuditRef(**expected["audit_ref"]))
    for step in range(start_step, max(records) + 1):
        record = records[step]
        transition = record["transition"]
        try:
            raw = engine.step(transition["action_t"])
            raw = engine.get_state()
        except Exception as exc:
            print("ERROR", step, transition["action_t"], exc)
            break
        next_expected = transition["obs_t1"]
        visible_map = engine.get_map() if raw.get("decision") == "map_select" else None
        actual = normalize_observation(
            raw,
            config=config,
            run_id=run_id,
            step_id=step + 1,
            audit_ref=None,
            visible_map=visible_map,
        )
        print(
            step,
            transition["action_t"]["action"],
            actual.phase,
            raw.get("event_id"),
            raw.get("context"),
            "expected", next_expected["phase"],
            actual.state_hash == next_expected["state_hash"],
        )
        if actual.state_hash != next_expected["state_hash"]:
            print("EXPECTED", next_expected["state_hash"], "ACTUAL", actual.state_hash)
            break
finally:
    engine.close()
