"""Profile cached sts2-cli combat restore stages on fixed Act 1-3 roots.

This is an engineering profiler, not a gameplay benchmark.  It verifies every
profiled restore against a state produced by the same real-engine action prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
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
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    first_card_select_candidate,
    headless_state_to_model_sample,
)
from sts2_dataset.util import utc_now, write_json_atomic  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "sts2_cli_restore_stage_profile_v0.json"
PROFILE_KEYS = (
    "cleanup",
    "load_save",
    "enter_combat",
    "prefix_replay",
    "set_draw_order",
    "suffix_replay",
    "final_projection",
    "server_pre_serialize_total",
)


def _next_command(state: dict[str, Any], step: int) -> dict[str, Any]:
    if state.get("decision") == "card_select":
        return candidate_to_headless_command(first_card_select_candidate(state))
    if state.get("decision") != "combat_play":
        raise EngineError(f"cannot advance unsupported decision: {state.get('decision')}")
    sample = headless_state_to_model_sample(
        state,
        transition_id=f"restore-profile:{step}",
        combat_id="restore-profile",
    )
    candidates = sorted(sample["candidates"], key=lambda row: int(row["candidate_index"]))
    non_terminal = [
        row for row in candidates if row.get("action", {}).get("type") != "end_turn"
    ]
    chosen = non_terminal[0] if non_terminal else candidates[0]
    return candidate_to_headless_command(chosen)


def _prefix_states(
    engine: Any,
    state: dict[str, Any],
    initial_prefix: list[dict[str, Any]],
    *,
    max_actions: int,
) -> dict[int, tuple[list[dict[str, Any]], dict[str, Any]]]:
    prefix = list(initial_prefix)
    states: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {
        len(prefix): (list(prefix), state)
    }
    for step in range(max_actions):
        if state.get("decision") not in {"combat_play", "card_select"}:
            break
        command = _next_command(state, step)
        state, _ = engine.send(command)
        prefix.append(command)
        states[len(prefix)] = (list(prefix), state)
    return states


def _selected_lengths(available: list[int], requested: list[int]) -> list[int]:
    result = []
    for target in requested:
        eligible = [value for value in available if value <= target]
        if eligible:
            result.append(max(eligible))
    if available:
        result.extend((min(available), max(available)))
    return sorted(set(result))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _parallel_restore_throughput(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    prefix: list[dict[str, Any]],
    expected: dict[str, Any],
    projection_mode: str,
    save_reuse_mode: str,
) -> list[dict[str, Any]]:
    rows = []
    baseline_rate = None
    for worker_count in args.worker_counts:
        with contextlib.ExitStack() as stack:
            workers = [
                stack.enter_context(_engine(args, game_data_dir))
                for _ in range(worker_count)
            ]
            requests = []
            for index, worker in enumerate(workers):
                warm, _ = worker.send({
                    "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                })
                if warm.get("decision") != "map_select":
                    raise EngineError(f"parallel worker warmup failed: {warm!r}")
                cache_name = f"parallel-restore-{scenario['scenario_id']}-{index}"
                cached, _ = worker.send({
                    "cmd": "cache_save", "name": cache_name, "path": str(entrance_save)
                })
                if cached.get("type") != "ok":
                    raise EngineError(f"parallel cache_save failed: {cached!r}")
                request = {
                    "cmd": "restore_combat",
                    "cache": cache_name,
                    "entry": _enter_command(scenario),
                    "prefix": prefix,
                }
                if projection_mode == "compact":
                    request["prefix_projection"] = "compact"
                if save_reuse_mode == "prepared":
                    request["reuse_prepared_save"] = True
                requests.append(request)

            def exercise(item: tuple[Any, dict[str, Any]]) -> list[float]:
                worker, request = item
                latencies = []
                for _ in range(args.repeats):
                    restored, elapsed_ms = worker.send(request)
                    if restored != expected:
                        raise EngineError(
                            f"parallel restore mismatch in {scenario['scenario_id']}"
                        )
                    latencies.append(float(elapsed_ms))
                return latencies

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                latency_groups = list(pool.map(exercise, zip(workers, requests)))
            wall_ms = (time.perf_counter() - started) * 1000.0
            all_latencies = [value for group in latency_groups for value in group]
            total_restores = worker_count * args.repeats
            rate = 1000.0 * total_restores / wall_ms
            if baseline_rate is None:
                baseline_rate = rate
            rows.append({
                "worker_count": worker_count,
                "projection_mode": projection_mode,
                "save_reuse_mode": save_reuse_mode,
                "total_restores": total_restores,
                "wall_ms": round(wall_ms, 3),
                "restores_per_second": round(rate, 3),
                "speedup_vs_one_worker": round(rate / baseline_rate, 3),
                "per_restore_round_trip_ms": _summary(all_latencies),
                "exact_state_match": True,
            })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, include_controls=False)
    game_data_dir = _game_data_dir(args.game_dir)
    scenario_reports = []

    with tempfile.TemporaryDirectory(prefix="sts2_restore_profile_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        for index, scenario in enumerate(scenarios):
            entrance_save = temp / f"scenario-{index}.save"
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
                prefix_states = _prefix_states(
                    source, state, initial_prefix, max_actions=args.max_actions
                )

            lengths = _selected_lengths(sorted(prefix_states), args.prefix_lengths)
            samples = []
            with _engine(args, game_data_dir) as worker:
                warm, _ = worker.send({
                    "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                })
                if warm.get("decision") != "map_select":
                    raise EngineError(f"worker warmup failed: {warm!r}")
                cache_name = f"restore-profile-{index}"
                cached, _ = worker.send({
                    "cmd": "cache_save", "name": cache_name, "path": str(entrance_save)
                })
                if cached.get("type") != "ok":
                    raise EngineError(f"cache_save failed: {cached!r}")

                for prefix_length in lengths:
                    prefix, expected = prefix_states[prefix_length]
                    for projection_mode, save_reuse_mode in itertools.product(
                        args.projection_modes,
                        args.save_reuse_modes,
                    ):
                        stage_values = {key: [] for key in PROFILE_KEYS}
                        round_trip_values = []
                        residual_values = []
                        for _ in range(args.repeats):
                            request: dict[str, Any] = {
                                "cmd": "restore_combat",
                                "cache": cache_name,
                                "entry": _enter_command(scenario),
                                "prefix": prefix,
                                "profile": True,
                            }
                            if projection_mode == "compact":
                                request["prefix_projection"] = "compact"
                            if save_reuse_mode == "prepared":
                                request["reuse_prepared_save"] = True
                            restored, round_trip_ms = worker.send(request)
                            profile = restored.pop("_profile_ms", None)
                            if profile is None:
                                raise EngineError("restore_combat did not return _profile_ms")
                            if restored != expected:
                                raise EngineError(
                                    f"{projection_mode}/{save_reuse_mode} restore mismatch in "
                                    f"{scenario['scenario_id']} at prefix {prefix_length}"
                                )
                            for key in PROFILE_KEYS:
                                stage_values[key].append(float(profile[key]))
                            round_trip_values.append(float(round_trip_ms))
                            residual_values.append(
                                max(
                                    0.0,
                                    float(round_trip_ms)
                                    - float(profile["server_pre_serialize_total"]),
                                )
                            )
                        samples.append({
                            "prefix_length": prefix_length,
                            "projection_mode": projection_mode,
                            "save_reuse_mode": save_reuse_mode,
                            "repeats": args.repeats,
                            "round_trip_ms": _summary(round_trip_values),
                            "client_protocol_and_serialize_residual_ms": _summary(residual_values),
                            "stages_ms": {
                                key: _summary(values) for key, values in stage_values.items()
                            },
                            "exact_state_match": True,
                        })

            longest_prefix_length = max(lengths)
            longest_prefix, longest_expected = prefix_states[longest_prefix_length]
            parallel_throughput = []
            for projection_mode in args.projection_modes:
                for save_reuse_mode in args.save_reuse_modes:
                    parallel_throughput.extend(_parallel_restore_throughput(
                        args,
                        game_data_dir=game_data_dir,
                        entrance_save=entrance_save,
                        scenario=scenario,
                        prefix=longest_prefix,
                        expected=longest_expected,
                        projection_mode=projection_mode,
                        save_reuse_mode=save_reuse_mode,
                    ))

            scenario_reports.append({
                "scenario_id": scenario["scenario_id"],
                "act": scenario["act"],
                "encounter": scenario["encounter"],
                "root": prepared,
                "profile_samples": samples,
                "parallel_restore_prefix_length": longest_prefix_length,
                "parallel_restore_throughput": parallel_throughput,
            })
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "samples": [
                    {
                        "prefix": row["prefix_length"],
                        "projection": row["projection_mode"],
                        "save_reuse": row["save_reuse_mode"],
                        "round_trip_p50_ms": row["round_trip_ms"]["p50"],
                        "load_save_p50_ms": row["stages_ms"]["load_save"]["p50"],
                    }
                    for row in samples
                ],
                "parallel": [
                    {
                        "workers": row["worker_count"],
                        "projection": row["projection_mode"],
                        "save_reuse": row["save_reuse_mode"],
                        "restores_per_second": row["restores_per_second"],
                        "speedup": row["speedup_vs_one_worker"],
                    }
                    for row in parallel_throughput
                ],
            }, ensure_ascii=False), flush=True)

    return {
        "schema_version": "sts2-cli-restore-profile-0.2.0",
        "generated_at": utc_now(),
        "status": "pass",
        "repeats": args.repeats,
        "information_boundary": {
            "hidden_rng_as_model_input": False,
            "profile_only_engine_internals": True,
        },
        "scenarios": scenario_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", default="restore-profile-v0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--projection-modes",
        choices=("legacy", "compact"),
        nargs="+",
        default=["legacy", "compact"],
    )
    parser.add_argument(
        "--save-reuse-modes",
        choices=("json", "prepared"),
        nargs="+",
        default=["json"],
    )
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "scenario_count": len(report["scenarios"]),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
