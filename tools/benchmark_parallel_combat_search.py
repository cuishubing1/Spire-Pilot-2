"""Measure root-parallel real-engine search throughput on fixed Act 1-3 roots.

The parallel searches intentionally keep independent trees.  This tool answers
only whether multiple persistent sts2-cli workers remove the engine throughput
bottleneck; it does not replace the production root-selection algorithm.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from benchmark_policy_guided_mcts import (  # noqa: E402
    DEFAULT_CONFIG,
    _resolve_checkpoint,
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


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_search_parallel_throughput_v0.json"


def _run_search(
    *,
    worker: Any,
    entrance_save: Path,
    scenario: dict[str, Any],
    root_state: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    config: dict[str, Any],
    budget: int,
    search_seed: int,
) -> dict[str, Any]:
    return search_current_root(
        worker=worker,
        entrance_save=entrance_save,
        enter_command=_enter_command(scenario),
        root_prefix=[],
        root_state=root_state,
        model=model,
        tensorizer=tensorizer,
        device=device,
        objective=objective,
        config=config,
        budget=budget,
        max_depth=int(config["puct"]["maximum_player_decision_depth"]),
        search_seed=search_seed,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if device != "cuda" and args.device == "cuda":
        raise EngineError(f"CUDA was requested but policy loaded on {device}")
    objective = CombatObjective.from_config(model.config)
    config = copy.deepcopy(load_json(args.config.resolve()))
    config.setdefault("engine_restore", {})["mode"] = "cached_batch"
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, include_controls=False)
    game_data_dir = _game_data_dir(args.game_dir)
    reports = []

    with tempfile.TemporaryDirectory(prefix="sts2_parallel_search_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        for index, scenario in enumerate(scenarios):
            entrance_save = temp / f"scenario-{index}.save"
            _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base_save,
                scenario=scenario,
                path=entrance_save,
            )
            with _engine(args, game_data_dir) as source:
                root_state, _ = source.send({
                    "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                })
                root_state, _ = source.send(_enter_command(scenario))
                root_state, precombat_prefix = _resolve_optional_precombat_selects(
                    source, root_state
                )
                if precombat_prefix:
                    raise EngineError(
                        "parallel throughput probe currently requires an empty precombat prefix"
                    )

            with _engine(args, game_data_dir) as baseline_worker:
                baseline_worker.send({
                    "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                })
                baseline_started = time.perf_counter()
                baseline = _run_search(
                    worker=baseline_worker,
                    entrance_save=entrance_save,
                    scenario=scenario,
                    root_state=root_state,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    objective=objective,
                    config=config,
                    budget=args.total_budget,
                    search_seed=args.search_seed,
                )
                baseline_wall_ms = (time.perf_counter() - baseline_started) * 1000.0

            if args.total_budget % args.workers != 0:
                raise ValueError("total budget must be divisible by worker count")
            per_worker_budget = args.total_budget // args.workers
            with contextlib.ExitStack() as stack:
                workers = [
                    stack.enter_context(_engine(args, game_data_dir))
                    for _ in range(args.workers)
                ]
                for worker in workers:
                    warm, _ = worker.send({
                        "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                    })
                    if warm.get("decision") != "map_select":
                        raise EngineError(f"parallel search worker warmup failed: {warm!r}")

                def exercise(item: tuple[int, Any]) -> dict[str, Any]:
                    worker_index, worker = item
                    return _run_search(
                        worker=worker,
                        entrance_save=entrance_save,
                        scenario=scenario,
                        root_state=root_state,
                        model=model,
                        tensorizer=tensorizer,
                        device=device,
                        objective=objective,
                        config=config,
                        budget=per_worker_budget,
                        search_seed=args.search_seed + worker_index * 1_000_003,
                    )

                parallel_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    independent = list(pool.map(exercise, enumerate(workers)))
                parallel_wall_ms = (time.perf_counter() - parallel_started) * 1000.0

            baseline_simulations = int(baseline["effective_budget"])
            parallel_simulations = sum(int(row["effective_budget"]) for row in independent)
            baseline_rate = 1000.0 * baseline_simulations / baseline_wall_ms
            parallel_rate = 1000.0 * parallel_simulations / parallel_wall_ms
            reports.append({
                "scenario_id": scenario["scenario_id"],
                "act": scenario["act"],
                "encounter": scenario["encounter"],
                "baseline": {
                    "worker_count": 1,
                    "requested_budget": args.total_budget,
                    "effective_simulations": baseline_simulations,
                    "wall_ms": round(baseline_wall_ms, 3),
                    "simulations_per_second": round(baseline_rate, 3),
                    "chosen_candidate_id": baseline["chosen_candidate"]["candidate_id"],
                },
                "root_parallel_probe": {
                    "worker_count": args.workers,
                    "requested_budget_per_worker": per_worker_budget,
                    "effective_simulations": parallel_simulations,
                    "wall_ms": round(parallel_wall_ms, 3),
                    "simulations_per_second": round(parallel_rate, 3),
                    "speedup_vs_one_worker": round(parallel_rate / baseline_rate, 3),
                    "independent_chosen_candidate_ids": [
                        row["chosen_candidate"]["candidate_id"] for row in independent
                    ],
                    "independent_search_wall_ms": [
                        float(row["search_wall_ms"]) for row in independent
                    ],
                    "shared_tree": False,
                    "production_root_selection": False,
                },
            })
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "baseline_sim_s": round(baseline_rate, 3),
                "parallel_sim_s": round(parallel_rate, 3),
                "speedup": round(parallel_rate / baseline_rate, 3),
            }, ensure_ascii=False), flush=True)

    return {
        "schema_version": "combat-search-parallel-throughput-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "total_budget": args.total_budget,
        "parallel_workers": args.workers,
        "mean_speedup": round(statistics.fmean(
            row["root_parallel_probe"]["speedup_vs_one_worker"] for row in reports
        ), 3),
        "meets_two_x_gate_all_scenarios": all(
            row["root_parallel_probe"]["speedup_vs_one_worker"] >= 2.0
            for row in reports
        ),
        "scope": "throughput feasibility only; independent trees are not merged",
        "scenarios": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", default="parallel-search-v0")
    parser.add_argument("--search-seed", type=int, default=20260817)
    parser.add_argument("--total-budget", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
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
        "device": report["device"],
        "mean_speedup": report["mean_speedup"],
        "meets_two_x_gate_all_scenarios": report["meets_two_x_gate_all_scenarios"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
