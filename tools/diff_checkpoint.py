"""Diff a recorded transition observation against its audit restore."""

import sys
import time
from pathlib import Path

from sts2_dataset.constants import CONFIG_PATH, RAW_ROOT
from sts2_dataset.engine import Sts2Engine
from sts2_dataset.normalize import normalize_observation
from sts2_dataset.types import AuditRef
from sts2_dataset.util import iter_jsonl_zst, load_json


def walk(left, right, path="root"):
    if type(left) is not type(right):
        print(path, "TYPE", type(left).__name__, type(right).__name__)
    elif isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                print(f"{path}.{key}", "MISSING", left.get(key), right.get(key))
            else:
                walk(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            print(path, "LEN", len(left), len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            walk(a, b, f"{path}[{index}]")
    elif left != right:
        print(path, repr(left), repr(right))

transition_id = sys.argv[1]
quiet = "quiet" in sys.argv[2:]
run_id, step_text = transition_id.rsplit(":", 1)
step_id = int(step_text)
record = next(
    r
    for r in iter_jsonl_zst(RAW_ROOT / f"{run_id}.jsonl.zst")
    if r.get("record_type") == "decision" and r.get("step_id") == step_id
)
expected = record["observation"]
config = load_json(CONFIG_PATH)
engine = Sts2Engine(config, "diff-checkpoint")
try:
    raw = engine.restore(AuditRef(**expected["audit_ref"]))
    raw = engine.get_state()
    visible_map = engine.get_map() if raw.get("decision") == "map_select" else None
    actual = normalize_observation(
        raw, config=config, run_id=run_id, step_id=step_id, audit_ref=None, visible_map=visible_map
    )
    print(expected["state_hash"], actual.state_hash)
    if "poll" in sys.argv[2:]:
        for index in range(10):
            time.sleep(0.2)
            polled = engine.get_state()
            print("poll", index, polled.get("decision"), (polled.get("player") or {}).get("deck_size"))
    if not quiet:
        walk(expected["agent_observation"], actual.agent_observation)
except Exception:
    print("\n".join(engine.stderr_tail(300)))
    raise
finally:
    engine.close()
