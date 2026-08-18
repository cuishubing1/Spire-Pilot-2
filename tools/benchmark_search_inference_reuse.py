"""Pair repeated and shared entity-encoder inference in identical CUDA searches."""

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

from benchmark_policy_guided_mcts import DEFAULT_CONFIG, search_current_root  # noqa: E402
from benchmark_search_restore_modes import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _search_summary,
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


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_search_inference_reuse_v0.json"


def _variants(dimension: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if dimension == "entity_encoding":
        return (
            {
                "label": "repeated_encoder",
                "reuse_entity_encoding": False,
                "reuse_precomputed_sample": True,
                "engine_transition_cache_enabled": False,
            },
            {
                "label": "shared_encoder",
                "reuse_entity_encoding": True,
                "reuse_precomputed_sample": True,
                "engine_transition_cache_enabled": False,
            },
        )
    if dimension == "precomputed_sample":
        return (
            {
                "label": "repeated_sample",
                "reuse_entity_encoding": True,
                "reuse_precomputed_sample": False,
                "engine_transition_cache_enabled": False,
            },
            {
                "label": "shared_sample",
                "reuse_entity_encoding": True,
                "reuse_precomputed_sample": True,
                "engine_transition_cache_enabled": False,
            },
        )
    if dimension == "transition_cache":
        return (
            {
                "label": "uncached_transitions",
                "reuse_entity_encoding": True,
                "reuse_precomputed_sample": True,
                "engine_transition_cache_enabled": False,
            },
            {
                "label": "cached_transitions",
                "reuse_entity_encoding": True,
                "reuse_precomputed_sample": True,
                "engine_transition_cache_enabled": True,
            },
        )
    if dimension == "full_stack":
        return (
            {
                "label": "original_stack",
                "engine_restore_mode": "cached_batch",
                "reuse_entity_encoding": False,
                "reuse_precomputed_sample": False,
                "engine_transition_cache_enabled": False,
            },
            {
                "label": "optimized_stack",
                "engine_restore_mode": "cached_batch_auto_prepared",
                "reuse_entity_encoding": True,
                "reuse_precomputed_sample": True,
                "engine_transition_cache_enabled": True,
            },
        )
    raise ValueError(f"unsupported comparison dimension: {dimension}")


def _paired_summary(
    rows: list[dict[str, Any]], *, baseline_label: str, optimized_label: str
) -> dict[str, Any]:
    pairs = []
    for repeat in sorted({int(row["repeat"]) for row in rows}):
        by_variant = {
            str(row["variant"]): row["search"]
            for row in rows
            if int(row["repeat"]) == repeat
        }
        baseline = by_variant[baseline_label]
        optimized = by_variant[optimized_label]
        pairs.append({
            "repeat": repeat,
            "search_speedup": baseline["search_wall_ms"] / optimized["search_wall_ms"],
            "inference_speedup": (
                baseline["model_inference_total_ms"]
                / optimized["model_inference_total_ms"]
            ),
            "chosen_candidate_equal": (
                baseline["chosen_candidate_id"] == optimized["chosen_candidate_id"]
            ),
            "search_candidate_equal": (
                baseline["search_selected_candidate_id"]
                == optimized["search_selected_candidate_id"]
            ),
            "policy_candidate_equal": (
                baseline["policy_candidate_id"] == optimized["policy_candidate_id"]
            ),
            "tree_node_count_equal": (
                baseline["tree_node_count"] == optimized["tree_node_count"]
            ),
            "engine_action_count_equal": (
                baseline["engine_action_count"] == optimized["engine_action_count"]
            ),
        })
    exact = all(
        row["chosen_candidate_equal"]
        and row["search_candidate_equal"]
        and row["policy_candidate_equal"]
        and row["tree_node_count_equal"]
        and row["engine_action_count_equal"]
        for row in pairs
    )
    return {
        "repeat_count": len(pairs),
        "search_speedup_mean": round(
            statistics.fmean(row["search_speedup"] for row in pairs), 3
        ),
        "inference_speedup_mean": round(
            statistics.fmean(row["inference_speedup"] for row in pairs), 3
        ),
        "exact_search_equal_all": exact,
        "pairs": pairs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if device != "cuda":
        raise EngineError(f"CUDA is required; resolved device={device!r}")
    objective = CombatObjective.from_config(model.config)
    base_config = load_json(args.config.resolve())
    restore_mode = str(base_config["engine_restore"]["mode"])
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, include_controls=False)
    game_data_dir = _game_data_dir(args.game_dir)
    reports: list[dict[str, Any]] = []
    variants = _variants(args.dimension)
    baseline_label = str(variants[0]["label"])
    optimized_label = str(variants[1]["label"])

    with tempfile.TemporaryDirectory(prefix="sts2_search_inference_reuse_") as temp_dir:
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
                    source, state, initial_prefix, max_actions=args.prefix_actions
                )
            eligible = [
                (length, prefix, root_state)
                for length, (prefix, root_state) in states.items()
                if length <= args.prefix_actions
                and root_state.get("decision") == "combat_play"
            ]
            if not eligible:
                raise EngineError(f"{scenario['scenario_id']} has no eligible combat root")
            prefix_length, root_prefix, root_state = max(eligible, key=lambda row: row[0])

            with contextlib.ExitStack() as stack:
                workers = {
                    str(variant["label"]): stack.enter_context(
                        _engine(args, game_data_dir)
                    )
                    for variant in variants
                }
                for worker in workers.values():
                    warm_state, _ = worker.send({
                        "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                    })
                    if warm_state.get("decision") != "map_select":
                        raise EngineError(f"search worker warmup failed: {warm_state!r}")

                for variant in variants:
                    label = str(variant["label"])
                    config = copy.deepcopy(base_config)
                    config.setdefault("engine_restore", {})["mode"] = str(
                        variant.get("engine_restore_mode", restore_mode)
                    )
                    config.setdefault("model_inference", {})[
                        "reuse_entity_encoding"
                    ] = bool(variant["reuse_entity_encoding"])
                    config["model_inference"]["reuse_precomputed_sample"] = bool(
                        variant["reuse_precomputed_sample"]
                    )
                    config.setdefault("engine_transition_cache", {})["enabled"] = bool(
                        variant["engine_transition_cache_enabled"]
                    )
                    search_current_root(
                        worker=workers[label],
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
                        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
                        for variant in order:
                            label = str(variant["label"])
                            config = copy.deepcopy(base_config)
                            config.setdefault("engine_restore", {})["mode"] = str(
                                variant.get("engine_restore_mode", restore_mode)
                            )
                            config.setdefault("model_inference", {})[
                                "reuse_entity_encoding"
                            ] = bool(variant["reuse_entity_encoding"])
                            config["model_inference"]["reuse_precomputed_sample"] = bool(
                                variant["reuse_precomputed_sample"]
                            )
                            config.setdefault("engine_transition_cache", {})[
                                "enabled"
                            ] = bool(variant["engine_transition_cache_enabled"])
                            result = search_current_root(
                                worker=workers[label],
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
                                "variant": label,
                                "reuse_entity_encoding": bool(
                                    variant["reuse_entity_encoding"]
                                ),
                                "reuse_precomputed_sample": bool(
                                    variant["reuse_precomputed_sample"]
                                ),
                                "engine_transition_cache_enabled": bool(
                                    variant["engine_transition_cache_enabled"]
                                ),
                                "engine_restore_mode": str(
                                    variant.get("engine_restore_mode", restore_mode)
                                ),
                                "search": _search_summary(result),
                            })
                            print(json.dumps({
                                "scenario": scenario["scenario_id"],
                                "budget": budget,
                                "repeat": repeat,
                                "variant": label,
                                "wall_ms": result["search_wall_ms"],
                                "inference_ms": result["model_inference_total_ms"],
                                "simulations_per_second": result["simulations_per_second"],
                            }, ensure_ascii=False), flush=True)

            budget_summaries = {
                str(budget): _paired_summary(
                    [row for row in rows if int(row["budget"]) == budget],
                    baseline_label=baseline_label,
                    optimized_label=optimized_label,
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

    summaries = [
        summary
        for scenario in reports
        for summary in scenario["budget_summaries"].values()
    ]
    exact = all(summary["exact_search_equal_all"] for summary in summaries)
    return {
        "schema_version": "combat-search-inference-reuse-0.1.0",
        "generated_at": utc_now(),
        "status": "pass" if exact else "fail",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "engine_restore_mode": restore_mode,
        "comparison_dimension": args.dimension,
        "baseline_variant": variants[0],
        "optimized_variant": variants[1],
        "budgets": args.budgets,
        "repeats": args.repeats,
        "requested_prefix_actions": args.prefix_actions,
        "max_depth": args.max_depth,
        "mean_search_speedup": round(statistics.fmean(
            summary["search_speedup_mean"] for summary in summaries
        ), 3),
        "mean_inference_speedup": round(statistics.fmean(
            summary["inference_speedup_mean"] for summary in summaries
        ), 3),
        "exact_search_equal_all": exact,
        "scenarios": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", default="search-inference-reuse-v0")
    parser.add_argument("--search-seed", type=int, default=20260818)
    parser.add_argument("--prefix-actions", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-budget", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument(
        "--dimension",
        choices=(
            "entity_encoding",
            "precomputed_sample",
            "transition_cache",
            "full_stack",
        ),
        default="entity_encoding",
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
        "mean_inference_speedup": report["mean_inference_speedup"],
        "exact_search_equal_all": report["exact_search_equal_all"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
