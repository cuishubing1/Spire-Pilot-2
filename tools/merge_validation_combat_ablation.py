"""Merge deterministic validation-combat shards into one verified report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_validation_combat_ablation import _aggregate, _summary  # noqa: E402
from sts2_dataset.util import load_json, utc_now, write_json_atomic  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "validation_combat_ablation_p2_full.json"


def _encounter_pool(row: dict[str, Any]) -> str:
    room_type = str(row.get("room_type") or "").lower()
    if room_type == "elite":
        return "elite"
    if room_type == "boss":
        return "boss"
    if room_type == "monster":
        return "weak" if str(row.get("encounter") or "").endswith("_WEAK") else "strong"
    return "unknown"


def _act_encounter_pool_summary(
    rows: list[dict[str, Any]], methods: tuple[str, ...]
) -> dict[str, Any]:
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((int(row["act"]), _encounter_pool(row)), []).append(row)
    result: dict[str, Any] = {}
    for (act, pool), bucket_rows in sorted(buckets.items()):
        human = _aggregate(bucket_rows, "human")
        values = {"human": human}
        for method in methods:
            model = _aggregate(bucket_rows, method)
            human_loss = float(human["total_hp_loss"])
            model["hp_loss_vs_human_percent"] = (
                round(
                    (float(model["total_hp_loss"]) - human_loss)
                    / human_loss
                    * 100.0,
                    3,
                )
                if human_loss > 0
                else None
            )
            values[method] = model
        result[f"act{act}:{pool}"] = values
    return result


def _paired_diagnostics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [row for row in rows if method in row]
    deltas = [float(row[method]["hp_loss"]) - float(row["human"]["hp_loss"]) for row in rows]
    ordered = sorted(
        rows,
        key=lambda row: float(row[method]["hp_loss"]) - float(row["human"]["hp_loss"]),
        reverse=True,
    )
    return {
        "lower_hp_loss_than_human": sum(delta < 0 for delta in deltas),
        "equal_hp_loss_to_human": sum(delta == 0 for delta in deltas),
        "higher_hp_loss_than_human": sum(delta > 0 for delta in deltas),
        "mean_paired_hp_loss_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "highest_hp_loss_regret": [
            {
                "scenario_id": row["scenario_id"],
                "source_combat_id": row["source_combat_id"],
                "act": row["act"],
                "floor": row["floor"],
                "ascension": row["ascension"],
                "encounter": row["encounter"],
                "human_hp_loss": row["human"]["hp_loss"],
                "model_hp_loss": row[method]["hp_loss"],
                "model_status": row[method]["status"],
                "hp_loss_delta": round(
                    float(row[method]["hp_loss"]) - float(row["human"]["hp_loss"]), 3
                ),
            }
            for row in ordered[:10]
        ],
    }


def _method_delta(
    rows: list[dict[str, Any]], *, baseline: str, candidate: str
) -> dict[str, Any]:
    rows = [row for row in rows if baseline in row and candidate in row]
    deltas = [
        float(row[candidate]["hp_loss"]) - float(row[baseline]["hp_loss"])
        for row in rows
    ]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_lower_hp_loss": sum(delta < 0 for delta in deltas),
        "equal_hp_loss": sum(delta == 0 for delta in deltas),
        "candidate_higher_hp_loss": sum(delta > 0 for delta in deltas),
        "total_hp_loss_delta": round(sum(deltas), 3),
        "mean_hp_loss_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "deaths_avoided": sum(
            row[baseline].get("status") == "death"
            and row[candidate].get("status") != "death"
            for row in rows
        ),
        "deaths_introduced": sum(
            row[baseline].get("status") != "death"
            and row[candidate].get("status") == "death"
            for row in rows
        ),
        "policy_action_changes": sum(
            int(row[candidate].get("policy_action_change_count") or 0)
            for row in rows
        ),
        "lookahead_decisions": sum(
            int(row[candidate].get("lookahead_decision_count") or 0)
            for row in rows
        ),
    }


def merge(paths: list[Path], *, observed_wall_ms: float | None = None) -> dict[str, Any]:
    shards = [load_json(path.resolve()) for path in paths]
    if not shards:
        raise ValueError("at least one shard is required")
    if any(shard.get("status") != "pass" for shard in shards):
        bad = [(shard.get("shard_index"), shard.get("status")) for shard in shards]
        raise ValueError(f"all shards must pass before merge: {bad}")
    shard_count = int(shards[0]["shard_count"])
    indices = sorted(int(shard["shard_index"]) for shard in shards)
    if shard_count != len(shards) or indices != list(range(shard_count)):
        raise ValueError(f"incomplete shard set: count={shard_count}, indices={indices}")
    methods = tuple(shards[0]["methods"])
    checkpoints = shards[0]["checkpoints"]
    for shard in shards[1:]:
        if tuple(shard["methods"]) != methods:
            raise ValueError("method mismatch across shards")
        if shard["checkpoints"] != checkpoints:
            raise ValueError("checkpoint mismatch across shards")
    rows = [row for shard in shards for row in shard["combats"]]
    rows.sort(key=lambda row: int(row["global_index"]))
    ids = [str(row["scenario_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenarios across shards")
    planned = sum(int(shard["planned_combats"]) for shard in shards)
    if len(rows) != planned:
        raise ValueError(f"missing completed combats: rows={len(rows)}, planned={planned}")
    common_rows = [row for row in rows if all(method in row for method in methods)]
    return {
        "schema_version": "validation-combat-ablation-merged-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "device": "cuda",
        "shard_count": shard_count,
        "combats": len(rows),
        "fully_evaluated_combats": len(common_rows),
        "methods": list(methods),
        "checkpoints": checkpoints,
        "comparison_semantics": shards[0]["comparison_semantics"],
        "one_step": shards[0]["one_step"],
        "timing": {
            "observed_debug_run_wall_ms": observed_wall_ms,
            "last_resume_segment_parallel_wall_ms": max(
                float(shard["wall_ms"]) for shard in shards
            ),
            "summed_last_resume_segment_worker_wall_ms": round(
                sum(float(shard["wall_ms"]) for shard in shards), 3
            ),
            "note": (
                "The first full run included encounter-mapping fixes, CLI timeout isolation, "
                "and resumed shards. Per-shard wall_ms covers only the final resume segment."
            ),
        },
        "summary": _summary(rows, methods),
        "common_evaluable_summary": _summary(common_rows, methods),
        "by_act_encounter_pool": _act_encounter_pool_summary(common_rows, methods),
        "engine_unsupported_combats": [
            {
                "scenario_id": row["scenario_id"],
                "source_combat_id": row["source_combat_id"],
                "encounter": row["encounter"],
                "error": row["evaluation_error"],
            }
            for row in rows if "evaluation_error" in row
        ],
        "paired_diagnostics": {
            method: _paired_diagnostics(rows, method) for method in methods
        },
        "controlled_method_deltas": {
            "p2_one_step_vs_policy": _method_delta(
                rows, baseline="p2_policy", candidate="p2_one_step"
            ),
            "residual_one_step_vs_policy": _method_delta(
                rows, baseline="residual_policy", candidate="residual_one_step"
            ),
            "residual_policy_vs_p2": _method_delta(
                rows, baseline="p2_policy", candidate="residual_policy"
            ),
            "residual_one_step_vs_p2_one_step": _method_delta(
                rows, baseline="p2_one_step", candidate="residual_one_step"
            ),
        },
        "source_shards": [str(path.resolve()) for path in paths],
        "combat_rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--observed-wall-ms", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = merge(args.shards, observed_wall_ms=args.observed_wall_ms)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "combats": report["combats"],
        "timing": report["timing"],
        "summary": report["summary"]["overall"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
