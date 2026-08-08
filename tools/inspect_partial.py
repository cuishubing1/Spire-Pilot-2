"""Print the tail of an interrupted raw run."""

import json
import sys
from pathlib import Path

from sts2_dataset.util import iter_jsonl_zst

records = []
try:
    records = list(iter_jsonl_zst(Path(sys.argv[1])))
except Exception as exc:
    print(type(exc).__name__, exc)
print("records", len(records))
for record in records[-40:]:
    if record.get("record_type") == "engine_exchange":
        request = record.get("request", {})
        response = record.get("response", {})
        print(record.get("sequence_no"), request.get("cmd"), request.get("action"), response.get("type"), response.get("decision"))
        if response.get("decision") == "shop":
            print("shop relics", json.dumps(response.get("relics"), ensure_ascii=False))
    else:
        print(record.get("sequence_no"), record.get("record_type"), record.get("step_id"))
