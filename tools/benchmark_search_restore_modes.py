"""Paired high-budget search benchmark for combat restore modes.

The benchmark advances one real-engine combat root per Act to a non-trivial
action prefix, then runs the same CUDA model, search seed, budget, and root
through legacy cached restore and the automatic compact restore path.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_policy_guided_mcts import (  # noqa: E402
    DEFAULT_CONFIG,
    search_current_root,
)
from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    _game_data_dir,
)
from profile_sts2_cli_restore import _prefix_states  # noqa: E402
from run_combat_mcts_act_sweep import (  # noqa: E402
    DEFAULT_TRANSITIONS,
    _create_base_save,
    _engine,
    _enter_command,
    _first_a0_ironclad_snapshots,
    _prepare_scenario_save,
    _resolve_optional_precombat_selects,
    _scenario_specs,
)
from run_combat_policy_online import _load_policy  # noqa: E402
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "combat_policy_p1_v13_cuda"
    / "20260817T151201Z"
    / "model.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_search_restore_modes_v0.json"
DEFAULT_BASELINE_MODE = "cached_batch_auto"
DEFAULT_OPTIMIZED_MODE = "cached_batch_auto_prepared"


def _search_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_budget": int(result["requested_budget"]),
        "effective_budget": int(result["effective_budget"]),
        "root_prefix_length": int(result["root_prefix_length"]),
        "tree_node_count": int(result["tree_node_count"]),
        "engine_action_count": int(result["engine_action_count"]),
        "engine_action_ipc_count": int(
            result.get("engine_action_ipc_count", result["engine_action_count"])
        ),
        "max_depth_reached": int(result["max_depth_reached"]),
        "search_wall_ms": float(result["search_wall_ms"]),
        "simulations_per_second": float(result["simulations_per_second"]),
        "root_restore_latency_ms": result["root_restore_latency_ms"],
        "simulation_latency_ms": result["simulation_latency_ms"],
        "model_inference_total_ms": float(result["model_inference_total_ms"]),
        "engine_action_latency_ms": result.get("engine_action_latency_ms"),
        "engine_action_latency_by_type_ms": result.get(
            "engine_action_latency_by_type_ms"
        ),
        "engine_transition_cache": result.get("engine_transition_cache"),
        "chosen_candidate_id": str(result["chosen_candidate"]["candidate_id"]),
        "search_selected_candidate_id": str(
            result["search_selected_candidate"]["candidate_id"]
        ),
        "policy_candidate_id": str(result["policy_candidate"]["candidate_id"]),
        "policy_fallback": result["policy_fallback"],
    }


def _paired_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_mode: str,
    optimized_mode: str,
) -> dict[str, Any]:
    paired = []
    for repeat in sorted({int(row["repeat"]) for row in rows}):
        by_mode = {
            str(row["restore_mode"]): row
            for row in rows
            if int(row["repeat"]) == repeat
        }
        baseline = by_mode[baseline_mode]["search"]
        optimized = by_mode[optimized_mode]["search"]
        paired.append({
            "repeat": repeat,
            "search_speedup": (
                baseline["search_wall_ms"] / optimized["search_wall_ms"]
            ),
            "restore_speedup": (
                baseline["root_restore_latency_ms"]["mean"]
                / optimized["root_restore_latency_ms"]["mean"]
            ),
            "chosen_candidate_equal": (
                baseline["chosen_candidate_id"] == optimized["chosen_candidate_id"]
            ),
            "tree_node_count_equal": (
                baseline["tree_node_count"] == optimized["tree_node_count"]
            ),
            "engine_action_count_equal": (
                baseline["engine_action_count"] == optimized["engine_action_count"]
            ),
        })
    return {
        "repeat_count": len(paired),
        "search_speedup_mean": round(
            statistics.fmean(row["search_speedup"] for row in paired), 3
        ),
        "restore_speedup_mean": round(
            statistics.fmean(row["restore_speedup"] for row in paired), 3
        ),
        "chosen_candidate_equal_all": all(
            row["chosen_candidate_equal"] for row in paired
        ),
        "tree_shape_equal_all": all(
            row["tree_node_count_equal"] and row["engine_action_count_equal"]
            for row in paired
        ),
        "pairs": paired,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if device != "cuda":
        raise EngineError(f"CUDA is required; resolved device={device!r}")
    objective = CombatObjective.from_config(model.config)
    base_config = load_json(args.config.resolve())
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, include_controls=False)
    game_data_dir = _game_data_dir(args.game_dir)
    reports: list[dict[str, Any]] = []
    restore_modes = (args.baseline_mode, args.optimized_mode)

    with tempfile.TemporaryDirectory(prefix="sts2_search_restore_modes_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        for scenario_index, scenario in enumerate(scenarios):
            entrance_save = temp / f"scenario-{scenario_index}.save"
            prepared = _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base_save,
                scenario=scenario,
                path=entrance_save,
            )
            with _engine(args, game_data_dir) as source:
                state, _ = source.send({
                    "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                })
                state, _ = source.send(_enter_command(scenario))
                state, initial_prefix = _resolve_optional_precombat_selects(source, state)
                states = _prefix_states(
                    source,
                    state,
                    initial_prefix,
                    max_actions=args.prefix_actions,
                )
            eligible = [
                (length, prefix, root_state)
                for length, (prefix, root_state) in states.items()
                if length <= args.prefix_actions
                and root_state.get("decision") == "combat_play"
            ]
            if not eligible:
                raise EngineError(
                    f"{scenario['scenario_id']} has no combat root at or before prefix "
                    f"{args.prefix_actions}"
                )
            prefix_length, root_prefix, root_state = max(eligible, key=lambda row: row[0])

            with contextlib.ExitStack() as stack:
                workers = {
                    mode: stack.enter_context(_engine(args, game_data_dir))
                    for mode in restore_modes
                }
                for worker in workers.values():
                    warm_state, _ = worker.send({
                        "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                    })
                    if warm_state.get("decision") != "map_select":
                        raise EngineError(f"search worker warmup failed: {warm_state!r}")

                # Warm CUDA kernels and both protocol paths outside measurements.
                for mode in restore_modes:
                    config = copy.deepcopy(base_config)
                    config.setdefault("engine_restore", {})["mode"] = mode
                    search_current_root(
                        worker=workers[mode],
                        entrance_save=entrance_save,
                        enter_command=_enter_command(scenario),
                        root_prefix=root_prefix,
                        root_state=root_state,
                        model=model,
                        tensorizer=tensorizer,
                        device=device,
                        objective=objective,
                        config=config,
                        budget=args.warmup_budget,
                        max_depth=args.max_depth,
                        search_seed=args.search_seed - 1,
                    )

                rows: list[dict[str, Any]] = []
                for budget in args.budgets:
                    for repeat in range(args.repeats):
                        order = (
                            restore_modes
                            if repeat % 2 == 0
                            else tuple(reversed(restore_modes))
                        )
                        for mode in order:
                            config = copy.deepcopy(base_config)
                            config.setdefault("engine_restore", {})["mode"] = mode
                            result = search_current_root(
                                worker=workers[mode],
                                entrance_save=entrance_save,
                                enter_command=_enter_command(scenario),
                                root_prefix=root_prefix,
                                root_state=root_state,
                                model=model,
                                tensorizer=tensorizer,
                                device=device,
                                objective=objective,
                                config=config,
                                budget=budget,
                                max_depth=args.max_depth,
                                search_seed=args.search_seed + repeat * 1_000_003,
                            )
                            rows.append({
                                "budget": budget,
                                "repeat": repeat,
                                "restore_mode": mode,
                                "search": _search_summary(result),
                            })
                            print(json.dumps({
                                "scenario": scenario["scenario_id"],
                                "prefix": prefix_length,
                                "budget": budget,
                                "repeat": repeat,
                                "mode": mode,
                                "wall_ms": result["search_wall_ms"],
                                "restore_mean_ms": result["root_restore_latency_ms"]["mean"],
                                "simulations_per_second": result["simulations_per_second"],
                            }, ensure_ascii=False), flush=True)

            budget_summaries = {
                str(budget): _paired_summary(
                    [row for row in rows if int(row["budget"]) == budget],
                    baseline_mode=args.baseline_mode,
                    optimized_mode=args.optimized_mode,
                )
                for budget in args.budgets
            }
            reports.append({
                "scenario_id": scenario["scenario_id"],
                "act": int(scenario["act"]),
                "encounter": scenario["encounter"],
                "prepared_root": prepared,
                "root_prefix_length": prefix_length,
                "budget_summaries": budget_summaries,
                "runs": rows,
            })

    all_summaries = [
        summary
        for scenario in reports
        for summary in scenario["budget_summaries"].values()
    ]
    return {
        "schema_version": "combat-search-restore-modes-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "budgets": args.budgets,
        "repeats": args.repeats,
        "requested_prefix_actions": args.prefix_actions,
        "max_depth": args.max_depth,
        "baseline_mode": args.baseline_mode,
        "optimized_mode": args.optimized_mode,
        "mean_search_speedup": round(statistics.fmean(
            summary["search_speedup_mean"] for summary in all_summaries
        ), 3),
        "mean_restore_speedup": round(statistics.fmean(
            summary["restore_speedup_mean"] for summary in all_summaries
        ), 3),
        "chosen_candidate_equal_all": all(
            summary["chosen_candidate_equal_all"] for summary in all_summaries
        ),
        "tree_shape_equal_all": all(
            summary["tree_shape_equal_all"] for summary in all_summaries
        ),
        "scenarios": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", default="search-restore-modes-v0")
    parser.add_argument("--search-seed", type=int, default=20260818)
    parser.add_argument("--prefix-actions", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-budget", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument(
        "--baseline-mode",
        default=DEFAULT_BASELINE_MODE,
    )
    parser.add_argument(
        "--optimized-mode",
        default=DEFAULT_OPTIMIZED_MODE,
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "mean_search_speedup": report["mean_search_speedup"],
        "mean_restore_speedup": report["mean_restore_speedup"],
        "chosen_candidate_equal_all": report["chosen_candidate_equal_all"],
        "tree_shape_equal_all": report["tree_shape_equal_all"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
