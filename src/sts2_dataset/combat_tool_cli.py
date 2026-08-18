from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator

from .combat_tool import load_combat_tool_checkpoint
from .constants import ROOT


REQUEST_SCHEMA_PATH = ROOT / "schemas" / "combat_tool_request_v0.schema.json"


def execute_request(tool: Any, request: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(REQUEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(request)
    return tool.decide(
        request["sample"],
        directive=request.get("directive"),
        mechanic_facts=request.get("mechanic_facts") or (),
        top_k=int(request.get("top_k", 3)),
    )


def _read_request(path: str) -> dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one fingerprinted Combat Tool V0 JSON request."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--request", default="-", help="JSON file, or - for stdin")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or torch device")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    tool = load_combat_tool_checkpoint(Path(args.checkpoint), device=args.device)
    response = execute_request(tool, _read_request(args.request))
    json.dump(
        response,
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        indent=None if args.compact else 2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
