"""Verify exact combat restoration by replaying from the room entrance.

This is the first battle-search correctness gate.  It deliberately avoids the
native in-combat save semantics: a map checkpoint is loaded, the same Monster
node is entered, and the recorded combat action prefix is replayed.  Every
decision response is compared byte-semantically through canonical JSON hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark_sts2_cli import (
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    EngineProcess,
    REPO_ROOT,
    _game_data_dir,
    _latency_summary,
    _safe_combat_action,
    _skip_neow,
)


def _state_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {
            "path": path,
            "expected": expected,
            "actual": actual,
            "reason": "type_mismatch",
        }
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return {
                "path": path,
                "expected_only": sorted(expected_keys - actual_keys),
                "actual_only": sorted(actual_keys - expected_keys),
                "reason": "key_mismatch",
            }
        for key in sorted(expected_keys):
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {
                "path": path,
                "expected": len(expected),
                "actual": len(actual),
                "reason": "length_mismatch",
            }
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return {
            "path": path,
            "expected": expected,
            "actual": actual,
            "reason": "value_mismatch",
        }
    return None


def _new_engine(args: argparse.Namespace, game_data_dir: Path) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet,
        engine_dll=args.engine_dll,
        game_data_dir=game_data_dir,
        sts2_lib=args.sts2_lib,
        timeout_s=args.timeout,
    )


def _select_monster(state: dict[str, Any]) -> dict[str, Any]:
    choices = state.get("choices", [])
    try:
        return next(choice for choice in choices if choice.get("type") == "Monster")
    except StopIteration as exc:
        raise EngineError(f"map exposes no Monster choice: {choices!r}") from exc


def _representative_combat_action(state: dict[str, Any]) -> dict[str, Any]:
    """Exercise non-lethal attacks before falling back to the safe policy."""
    energy = state.get("energy", 0)
    enemies = state.get("enemies", [])
    if enemies:
        target = enemies[0]
        for card in state.get("hand", []):
            if not (
                card.get("can_play")
                and card.get("cost", 99) <= energy
                and card.get("type") == "Attack"
            ):
                continue
            previews = card.get("damage_by_target") or []
            preview = next(
                (
                    row
                    for row in previews
                    if row.get("target_index") == target.get("index")
                ),
                {},
            )
            damage = preview.get("total_damage", preview.get("damage"))
            if damage is None:
                damage = (card.get("stats") or {}).get("damage", 0)
            # Leave a generous reserve so the correctness probe never enters the
            # currently unstable combat-reward settlement path.
            if int(target.get("hp", 0)) - int(damage or 0) > 15:
                args: dict[str, Any] = {"card_index": card["index"]}
                if card.get("target_type") == "AnyEnemy":
                    args["target_index"] = target["index"]
                return {"cmd": "action", "action": "play_card", "args": args}
    return _safe_combat_action(state)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    game_data_dir = _game_data_dir(args.game_dir)
    for required in (args.dotnet, args.engine_dll):
        if not required.is_file():
            raise FileNotFoundError(f"required artifact is missing: {required}")
    if not args.sts2_lib.is_dir():
        raise FileNotFoundError(f"sts2-cli lib directory is missing: {args.sts2_lib}")

    result: dict[str, Any] = {
        "status": "running",
        "configuration": {
            "game_data_dir": str(game_data_dir),
            "engine_dll": str(args.engine_dll.resolve()),
            "character": args.character,
            "ascension": args.ascension,
            "seed": args.seed,
            "prefix_steps": args.prefix_steps,
            "repeats": args.repeats,
            "persistent_repeats": args.persistent_repeats,
            "command_timeout_s": args.timeout,
        },
    }

    with tempfile.TemporaryDirectory(prefix="sts2_combat_replay_") as temp_dir:
        root_save = Path(temp_dir) / "combat_entrance.save"
        commands: list[dict[str, Any]] = []
        action_labels: list[str] = []
        expected_states: list[dict[str, Any]] = []

        with _new_engine(args, game_data_dir) as source:
            state, start_run_ms = source.send(
                {
                    "cmd": "start_run",
                    "character": args.character,
                    "ascension": args.ascension,
                    "seed": args.seed,
                    "lang": "en",
                }
            )
            state, neow_latencies = _skip_neow(source, state)
            _, set_player_ms = source.send(
                {"cmd": "set_player", "hp": args.player_hp, "max_hp": args.player_hp}
            )
            state, _ = source.send({"cmd": "get_state"})
            map_choice = _select_monster(state)
            save_response, root_save_ms = source.send(
                {"cmd": "write_continue_save", "path": str(root_save)}
            )
            if not save_response.get("success") or not root_save.is_file():
                raise EngineError(f"failed to write combat entrance save: {save_response!r}")
            state, enter_combat_ms = source.send(
                {
                    "cmd": "action",
                    "action": "select_map_node",
                    "args": {"col": map_choice["col"], "row": map_choice["row"]},
                }
            )
            if state.get("decision") != "combat_play":
                raise EngineError(f"Monster node did not enter combat: {state!r}")
            combat_root_state = state

            source_action_latencies: list[float] = []
            for step in range(args.prefix_steps):
                command = _representative_combat_action(state)
                commands.append(command)
                label = str(command.get("action"))
                if command.get("action") == "play_card":
                    card_index = (command.get("args") or {}).get("card_index")
                    card = next(
                        (
                            item
                            for item in state.get("hand", [])
                            if item.get("index") == card_index
                        ),
                        {},
                    )
                    label = f"play_card:{card.get('type', 'Unknown')}"
                action_labels.append(label)
                state, elapsed_ms = source.send(command)
                source_action_latencies.append(elapsed_ms)
                if state.get("decision") != "combat_play":
                    raise EngineError(
                        f"source combat ended at prefix step {step}; use a shorter prefix"
                    )
                expected_states.append(state)

            target_state = state
            probe_action = _representative_combat_action(target_state)
            expected_successor, source_probe_ms = source.send(probe_action)
            if expected_successor.get("decision") != "combat_play":
                raise EngineError(
                    "probe action ended combat; use a shorter prefix or different seed"
                )

        result["source"] = {
            "startup_ms": round(source.startup_ms, 3),
            "start_run_ms": round(start_run_ms, 3),
            "neow_action_latency": _latency_summary(neow_latencies),
            "set_player_ms": round(set_player_ms, 3),
            "write_entrance_checkpoint_ms": round(root_save_ms, 3),
            "enter_combat_ms": round(enter_combat_ms, 3),
            "prefix_action_latency": _latency_summary(source_action_latencies),
            "combat_root_hash": _state_hash(combat_root_state),
            "target_hash": _state_hash(target_state),
            "probe_action": probe_action,
            "probe_action_ms": round(source_probe_ms, 3),
            "probe_successor_hash": _state_hash(expected_successor),
            "selected_node": map_choice,
            "action_mix": {
                label: action_labels.count(label) for label in sorted(set(action_labels))
            },
            "target_enemy_hp": [enemy.get("hp") for enemy in target_state.get("enemies", [])],
        }

        samples: list[dict[str, Any]] = []
        all_match = True
        for repeat in range(args.repeats):
            sample_started = time.perf_counter()
            with _new_engine(args, game_data_dir) as replay:
                restored_map, load_ms = replay.send(
                    {"cmd": "load_save", "path": str(root_save), "lang": "en"}
                )
                if restored_map.get("decision") != "map_select":
                    raise EngineError(
                        f"entrance save did not restore map_select: {restored_map!r}"
                    )
                state, enter_ms = replay.send(
                    {
                        "cmd": "action",
                        "action": "select_map_node",
                        "args": {"col": map_choice["col"], "row": map_choice["row"]},
                    }
                )

                root_match = state == combat_root_state
                first_difference = (
                    None
                    if root_match
                    else _first_difference(combat_root_state, state)
                )
                prefix_matches = 0
                replay_action_latencies: list[float] = []
                for step, (command, expected_state) in enumerate(
                    zip(commands, expected_states)
                ):
                    state, elapsed_ms = replay.send(command)
                    replay_action_latencies.append(elapsed_ms)
                    if state == expected_state:
                        prefix_matches += 1
                    elif first_difference is None:
                        difference = _first_difference(expected_state, state)
                        first_difference = {"step": step, **(difference or {})}

                target_match = state == target_state
                successor, probe_ms = replay.send(probe_action)
                successor_match = successor == expected_successor
                if not successor_match and first_difference is None:
                    difference = _first_difference(expected_successor, successor)
                    first_difference = {
                        "step": "probe_successor",
                        **(difference or {}),
                    }

                sample_match = (
                    root_match
                    and prefix_matches == len(commands)
                    and target_match
                    and successor_match
                )
                all_match = all_match and sample_match
                samples.append(
                    {
                        "repeat": repeat,
                        "startup_ms": round(replay.startup_ms, 3),
                        "load_entrance_ms": round(load_ms, 3),
                        "enter_combat_ms": round(enter_ms, 3),
                        "replay_action_latency": _latency_summary(
                            replay_action_latencies
                        ),
                        "probe_action_ms": round(probe_ms, 3),
                        "total_branch_ms": round(
                            (time.perf_counter() - sample_started) * 1000.0, 3
                        ),
                        "root_match": root_match,
                        "prefix_state_matches": prefix_matches,
                        "prefix_state_count": len(commands),
                        "target_match": target_match,
                        "successor_match": successor_match,
                        "first_difference": first_difference,
                    }
                )

        branch_latencies = [float(sample["total_branch_ms"]) for sample in samples]
        replay_latencies = [
            float(sample["replay_action_latency"]["total_ms"]) for sample in samples
        ]
        result["replay"] = {
            "all_exact": all_match,
            "exact_repeats": sum(
                bool(
                    sample["root_match"]
                    and sample["target_match"]
                    and sample["successor_match"]
                    and sample["prefix_state_matches"] == sample["prefix_state_count"]
                )
                for sample in samples
            ),
            "samples": samples,
            "action_prefix_latency": _latency_summary(replay_latencies),
            "fresh_process_branch_latency": _latency_summary(branch_latencies),
            "fresh_process_branches_per_second_mean": round(
                1000.0 / statistics.fmean(branch_latencies), 3
            )
            if branch_latencies
            else None,
        }

        persistent_samples: list[dict[str, Any]] = []
        persistent_all_match = True
        if args.persistent_repeats > 0:
            with _new_engine(args, game_data_dir) as replay:
                for repeat in range(args.persistent_repeats):
                    sample_started = time.perf_counter()
                    load_command = "load_save" if repeat == 0 else "reload_save"
                    restored_map, load_ms = replay.send(
                        {"cmd": load_command, "path": str(root_save), "lang": "en"}
                    )
                    if restored_map.get("decision") != "map_select":
                        raise EngineError(
                            f"persistent entrance restore did not produce map_select: {restored_map!r}"
                        )
                    state, enter_ms = replay.send(
                        {
                            "cmd": "action",
                            "action": "select_map_node",
                            "args": {"col": map_choice["col"], "row": map_choice["row"]},
                        }
                    )
                    root_match = state == combat_root_state
                    prefix_matches = 0
                    first_difference = None if root_match else _first_difference(combat_root_state, state)
                    replay_action_latencies: list[float] = []
                    for step, (command, expected_state) in enumerate(zip(commands, expected_states)):
                        state, elapsed_ms = replay.send(command)
                        replay_action_latencies.append(elapsed_ms)
                        if state == expected_state:
                            prefix_matches += 1
                        elif first_difference is None:
                            difference = _first_difference(expected_state, state)
                            first_difference = {"step": step, **(difference or {})}
                    target_match = state == target_state
                    successor, probe_ms = replay.send(probe_action)
                    successor_match = successor == expected_successor
                    sample_match = (
                        root_match
                        and prefix_matches == len(commands)
                        and target_match
                        and successor_match
                    )
                    persistent_all_match = persistent_all_match and sample_match
                    persistent_samples.append({
                        "repeat": repeat,
                        "restore_command": load_command,
                        "load_entrance_ms": round(load_ms, 3),
                        "enter_combat_ms": round(enter_ms, 3),
                        "replay_action_latency": _latency_summary(replay_action_latencies),
                        "probe_action_ms": round(probe_ms, 3),
                        "total_branch_ms": round(
                            (time.perf_counter() - sample_started) * 1000.0, 3
                        ),
                        "root_match": root_match,
                        "prefix_state_matches": prefix_matches,
                        "prefix_state_count": len(commands),
                        "target_match": target_match,
                        "successor_match": successor_match,
                        "first_difference": first_difference,
                    })
        persistent_branch_latencies = [
            float(sample["total_branch_ms"]) for sample in persistent_samples
        ]
        warm_persistent_latencies = [
            float(sample["total_branch_ms"])
            for sample in persistent_samples
            if sample["restore_command"] == "reload_save"
        ]
        result["persistent_replay"] = {
            "all_exact": persistent_all_match,
            "exact_repeats": sum(
                bool(
                    sample["root_match"]
                    and sample["target_match"]
                    and sample["successor_match"]
                    and sample["prefix_state_matches"] == sample["prefix_state_count"]
                )
                for sample in persistent_samples
            ),
            "samples": persistent_samples,
            "branch_latency": _latency_summary(persistent_branch_latencies),
            "warm_branch_latency": _latency_summary(warm_persistent_latencies),
            "branches_per_second_mean": round(
                1000.0 / statistics.fmean(persistent_branch_latencies), 3
            ) if persistent_branch_latencies else None,
            "warm_branches_per_second_mean": round(
                1000.0 / statistics.fmean(warm_persistent_latencies), 3
            ) if warm_persistent_latencies else None,
        }

    result["status"] = "pass" if (
        result["replay"]["all_exact"] and result["persistent_replay"]["all_exact"]
    ) else "fail"
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", default="spire-pilot-2-benchmark-v1")
    parser.add_argument("--player-hp", type=int, default=999)
    parser.add_argument("--prefix-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--persistent-repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.prefix_steps < 0:
        parser.error("--prefix-steps cannot be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.persistent_repeats < 0:
        parser.error("--persistent-repeats cannot be negative")
    if args.player_hp < 1:
        parser.error("--player-hp must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_benchmark(args)
        exit_code = 0 if result.get("status") == "pass" else 1
    except Exception as exc:
        result = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
