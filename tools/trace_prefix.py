"""Trace collector-equivalent prefix replay and print the first hash divergence."""

import sys
from pathlib import Path

from sts2_dataset.constants import AUDIT_ROOT, CONFIG_PATH, RAW_ROOT
from sts2_dataset.engine import Sts2Engine
from sts2_dataset.normalize import normalize_observation
from sts2_dataset.util import iter_jsonl_zst, load_json

run_id = sys.argv[1]
all_records = list(iter_jsonl_zst(RAW_ROOT / f"{run_id}.jsonl.zst"))
seed = next(r["seed"] for r in all_records if r.get("record_type") == "run_start")
records = [
    r
    for r in all_records
    if r.get("record_type") == "decision"
]
config = load_json(CONFIG_PATH)
engine = Sts2Engine(config, "trace-prefix")
save_path = AUDIT_ROOT / "trace-prefix" / ".boundary.native.json"
save_path.parent.mkdir(parents=True, exist_ok=True)


def materialize(raw, step):
    raw = engine.get_state()
    visible_map = engine.get_map() if raw.get("decision") == "map_select" else None
    obs = normalize_observation(
        raw, config=config, run_id=run_id, step_id=step,
        audit_ref=None, visible_map=visible_map,
    )
    engine.send({"cmd": "write_continue_save", "path": str(save_path)})
    return raw, obs


try:
    raw = engine.reset(seed=seed)
    raw, actual = materialize(raw, 0)
    for step, record in enumerate(records):
        expected = record["observation"]
        print(step, actual.phase, actual.context.get("room_type"), actual.state_hash == expected["state_hash"], actual.state_hash[:8], expected["state_hash"][:8])
        if actual.state_hash != expected["state_hash"] or record.get("action") is None:
            break
        raw = engine.step(record["action"])
        raw, actual = materialize(raw, step + 1)
finally:
    save_path.unlink(missing_ok=True)
    engine.close()
