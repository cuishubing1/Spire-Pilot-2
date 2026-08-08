"""Developer diagnostic: print visible-state differences after deterministic replay."""

import uuid

from sts2_dataset.constants import CONFIG_PATH
from sts2_dataset.engine import Sts2Engine
from sts2_dataset.normalize import normalize_observation
from sts2_dataset.policy import HeuristicPolicy
from sts2_dataset.smoke import _advance_to_map
from sts2_dataset.util import load_json


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


config = load_json(CONFIG_PATH)
run_id = "diff-" + uuid.uuid4().hex[:8]
engine = Sts2Engine(config, run_id)
policy = HeuristicPolicy("smoke-v01071")
try:
    raw = engine.reset(seed="smoke-v01071")
    raw = _advance_to_map(engine, raw, config, run_id, policy)
    before = normalize_observation(
        raw, config=config, run_id=run_id, step_id=0, audit_ref=None, visible_map=engine.get_map()
    )
    audit = engine.snapshot(0, before.state_hash)
    raw = engine.restore(audit)
    after = normalize_observation(
        raw, config=config, run_id=run_id, step_id=0, audit_ref=None, visible_map=engine.get_map()
    )
    print(before.state_hash, after.state_hash)
    starts = [i for i, x in enumerate(engine._exchanges) if x["request"].get("cmd") == "start_run"]
    for run_no, begin in enumerate(starts):
        end = starts[run_no + 1] if run_no + 1 < len(starts) else len(engine._exchanges)
        print("TRACE", run_no, [
            (x["request"].get("action", x["request"].get("cmd")), x["response"].get("decision"), x["response"].get("type"))
            for x in engine._exchanges[begin:end]
        ])
    walk(before.agent_observation, after.agent_observation)
finally:
    engine.close()
