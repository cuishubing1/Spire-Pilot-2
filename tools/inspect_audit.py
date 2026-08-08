"""Summarize replay base and action tail for one or more checkpoints."""

import sys
from pathlib import Path

from sts2_dataset.constants import AUDIT_ROOT
from sts2_dataset.util import read_zstd_json

run_id = sys.argv[1]
for value in sys.argv[2:]:
    step = int(value)
    bundle = read_zstd_json(AUDIT_ROOT / run_id / f"{step:06d}.checkpoint.zst")
    base = int(bundle.get("replay_base_action_count") or 0)
    print("STEP", step, "room", bundle.get("native_save_room_type"), "base", base, "actions", len(bundle["actions"]))
    print(bundle["actions"][base:])

