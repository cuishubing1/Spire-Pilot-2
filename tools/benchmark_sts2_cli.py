"""Measure sts2-cli startup, combat action, and checkpoint branch latency.

The benchmark talks to the real v0.107.1 game assembly through the sts2-cli
JSON-lines protocol.  Every request has a hard timeout because an engine branch
must never be allowed to stall an entire search worker.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME_DIR = Path(r"D:\steam\steamapps\common\Slay the Spire 2")
DEFAULT_DOTNET = REPO_ROOT / ".dotnet" / "dotnet.exe"
DEFAULT_ENGINE_DLL = (
    REPO_ROOT
    / "third_party"
    / "sts2-cli"
    / "src"
    / "Sts2Headless"
    / "bin"
    / "Debug"
    / "net9.0"
    / "Sts2Headless.dll"
)
DEFAULT_STS2_LIB = REPO_ROOT / "third_party" / "sts2-cli" / "lib"


class EngineError(RuntimeError):
    """Base error raised by the benchmark protocol wrapper."""


class EngineTimeout(EngineError):
    """Raised when a command does not produce a JSON response in time."""


def _reader(stream: Any, output: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line.rstrip("\r\n"))
    finally:
        output.put(None)


def _stderr_reader(stream: Any, output: deque[str]) -> None:
    for line in iter(stream.readline, ""):
        output.append(line.rstrip("\r\n"))


class EngineProcess:
    def __init__(
        self,
        *,
        dotnet: Path,
        engine_dll: Path,
        game_data_dir: Path,
        sts2_lib: Path,
        timeout_s: float,
    ) -> None:
        self.timeout_s = timeout_s
        self._stdout: queue.Queue[str | None] = queue.Queue()
        # Boss/event failures often emit a long async stack before the recovery
        # path reports its secondary error.  Keep enough context to retain the
        # original exception rather than only the final fallback messages.
        self._stderr: deque[str] = deque(maxlen=500)
        env = os.environ.copy()
        env["DOTNET_ROOT"] = str(dotnet.parent)
        env["STS2_GAME_DIR"] = str(game_data_dir)
        env["STS2_LIB"] = str(sts2_lib)

        started = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(dotnet), str(engine_dll)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        threading.Thread(
            target=_reader, args=(self.proc.stdout, self._stdout), daemon=True
        ).start()
        threading.Thread(
            target=_stderr_reader, args=(self.proc.stderr, self._stderr), daemon=True
        ).start()
        ready = self._read_json(timeout_s, context="engine startup")
        self.startup_ms = (time.perf_counter() - started) * 1000.0
        if ready.get("type") != "ready":
            self.kill()
            raise EngineError(f"expected ready response, got {ready!r}")

    def _read_json(self, timeout_s: float, *, context: str) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.kill()
                tail = "\n".join(self._stderr)
                raise EngineTimeout(
                    f"timeout after {timeout_s:.1f}s during {context}; stderr tail:\n{tail}"
                )
            try:
                line = self._stdout.get(timeout=remaining)
            except queue.Empty as exc:
                self.kill()
                tail = "\n".join(self._stderr)
                raise EngineTimeout(
                    f"timeout after {timeout_s:.1f}s during {context}; stderr tail:\n{tail}"
                ) from exc
            if line is None:
                code = self.proc.poll()
                tail = "\n".join(self._stderr)
                raise EngineError(
                    f"engine exited with code {code} during {context}; stderr tail:\n{tail}"
                )
            if not line.startswith("{"):
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(response, dict):
                return response

    def send(
        self, command: dict[str, Any], *, timeout_s: float | None = None
    ) -> tuple[dict[str, Any], float]:
        if self.proc.poll() is not None:
            raise EngineError(f"engine already exited with code {self.proc.returncode}")
        assert self.proc.stdin is not None
        started = time.perf_counter()
        try:
            self.proc.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError("failed to write command to engine") from exc
        response = self._read_json(timeout_s or self.timeout_s, context=repr(command))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.get("type") == "error":
            raise EngineError(f"engine rejected {command!r}: {response.get('message')}")
        return response, elapsed_ms

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send({"cmd": "quit"}, timeout_s=min(self.timeout_s, 2.0))
            except EngineError:
                self.kill()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "EngineProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _game_data_dir(path: Path) -> Path:
    candidate = path.resolve()
    if (candidate / "sts2.dll").is_file():
        return candidate
    nested = candidate / "data_sts2_windows_x86_64"
    if (nested / "sts2.dll").is_file():
        return nested
    raise FileNotFoundError(
        f"could not find sts2.dll under {candidate} or {nested}"
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    total_ms = sum(values)
    return {
        "count": len(values),
        "total_ms": round(total_ms, 3),
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
        "p50_ms": round(statistics.median(values), 3) if values else None,
        "p95_ms": round(_percentile(values, 0.95), 3) if values else None,
        "max_ms": round(max(values), 3) if values else None,
        "actions_per_second": round(1000.0 * len(values) / total_ms, 3)
        if total_ms > 0
        else None,
    }


def _skip_neow(engine: EngineProcess, state: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    latencies: list[float] = []
    for _ in range(20):
        decision = state.get("decision")
        if decision == "map_select":
            return state, latencies
        if decision == "event_choice":
            options = [option for option in state.get("options", []) if not option.get("is_locked")]
            if not options:
                raise EngineError("Neow event exposed no unlocked option")
            command = {
                "cmd": "action",
                "action": "choose_option",
                "args": {"option_index": options[0]["index"]},
            }
        elif decision == "card_reward":
            command = {"cmd": "action", "action": "skip_card_reward"}
        elif decision == "bundle_select":
            command = {
                "cmd": "action",
                "action": "select_bundle",
                "args": {"bundle_index": 0},
            }
        elif decision == "card_select":
            if state.get("min_select", 0) == 0:
                command = {"cmd": "action", "action": "skip_select"}
            else:
                command = {
                    "cmd": "action",
                    "action": "select_cards",
                    "args": {"indices": "0"},
                }
        else:
            command = {"cmd": "action", "action": "proceed"}
        state, elapsed = engine.send(command)
        latencies.append(elapsed)
    raise EngineError(f"failed to leave Neow flow; final decision={state.get('decision')!r}")


def _safe_combat_action(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("decision") != "combat_play":
        raise EngineError(f"expected combat_play, got {state.get('decision')!r}")
    energy = state.get("energy", 0)
    playable = [
        card
        for card in state.get("hand", [])
        if card.get("can_play") and card.get("cost", 99) <= energy
    ]
    # Avoid ending the benchmark by killing the enemy: play skills first and
    # otherwise end the turn. This measures both card actions and turn settling.
    skills = [card for card in playable if str(card.get("type", "")).lower() != "attack"]
    if not skills:
        return {"cmd": "action", "action": "end_turn"}
    card = skills[0]
    args: dict[str, Any] = {"card_index": card["index"]}
    if card.get("target_type") == "AnyEnemy" and state.get("enemies"):
        args["target_index"] = state["enemies"][0]["index"]
    return {"cmd": "action", "action": "play_card", "args": args}


def _state_signature(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "decision": state.get("decision"),
        "round": state.get("round"),
        "energy": state.get("energy"),
        "player_hp": player.get("hp"),
        "map_choices": [
            {
                "col": choice.get("col"),
                "row": choice.get("row"),
                "type": choice.get("type"),
            }
            for choice in state.get("choices", [])
        ],
        "hand": [card.get("name") for card in state.get("hand", [])],
        "enemies": [
            {"name": enemy.get("name"), "hp": enemy.get("hp")}
            for enemy in state.get("enemies", [])
        ],
    }


def _engine_factory(args: argparse.Namespace, game_data_dir: Path) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet,
        engine_dll=args.engine_dll,
        game_data_dir=game_data_dir,
        sts2_lib=args.sts2_lib,
        timeout_s=args.timeout,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    game_data_dir = _game_data_dir(args.game_dir)
    for required in (args.dotnet, args.engine_dll):
        if not required.is_file():
            raise FileNotFoundError(f"required executable/artifact is missing: {required}")
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
            "encounter": args.encounter,
            "steps": args.steps,
            "branch_probes": args.branch_probes,
            "command_timeout_s": args.timeout,
        },
    }

    with tempfile.TemporaryDirectory(prefix="sts2_cli_benchmark_") as temp_dir:
        map_save_path = Path(temp_dir) / "map_checkpoint.json"
        combat_save_path = Path(temp_dir) / "combat_checkpoint.json"
        with _engine_factory(args, game_data_dir) as engine:
            result["startup_ms"] = round(engine.startup_ms, 3)
            state, start_ms = engine.send(
                {
                    "cmd": "start_run",
                    "character": args.character,
                    "ascension": args.ascension,
                    "seed": args.seed,
                    "lang": "en",
                }
            )
            state, neow_latencies = _skip_neow(engine, state)
            _, set_player_ms = engine.send(
                {"cmd": "set_player", "hp": 999, "max_hp": 999}
            )
            state, post_set_state_ms = engine.send({"cmd": "get_state"})
            if state.get("decision") != "map_select":
                raise EngineError(f"expected map_select before checkpoint, got {state!r}")
            map_signature = _state_signature(state)
            choices = state.get("choices", [])
            if not choices:
                raise EngineError("map checkpoint exposed no selectable node")
            map_choice = next(
                (choice for choice in choices if choice.get("type") == "Monster"),
                choices[0],
            )
            map_save_response, map_save_ms = engine.send(
                {"cmd": "write_continue_save", "path": str(map_save_path)}
            )
            if not map_save_response.get("success") or not map_save_path.is_file():
                raise EngineError(f"map checkpoint write failed: {map_save_response!r}")

            state, enter_room_ms = engine.send(
                {"cmd": "enter_room", "type": "combat", "encounter": args.encounter}
            )
            if state.get("decision") != "combat_play":
                raise EngineError(f"enter_room did not produce combat_play: {state!r}")
            state, get_state_ms = engine.send({"cmd": "get_state"})

            action_latencies: list[float] = []
            action_kinds: list[str] = []
            for _ in range(args.steps):
                command = _safe_combat_action(state)
                state, elapsed = engine.send(command)
                action_latencies.append(elapsed)
                action_kinds.append(command["action"])
                if state.get("decision") != "combat_play":
                    break

            result["single_process"] = {
                "start_run_ms": round(start_ms, 3),
                "neow_action_latency": _latency_summary(neow_latencies),
                "set_player_ms": round(set_player_ms, 3),
                "post_set_get_state_ms": round(post_set_state_ms, 3),
                "write_map_checkpoint_ms": round(map_save_ms, 3),
                "enter_combat_ms": round(enter_room_ms, 3),
                "get_state_ms": round(get_state_ms, 3),
                "combat_action_latency": _latency_summary(action_latencies),
                "action_mix": {
                    kind: action_kinds.count(kind) for kind in sorted(set(action_kinds))
                },
                "final_decision": state.get("decision"),
            }

        branch_samples: list[dict[str, Any]] = []
        for probe_index in range(args.branch_probes):
            branch_started = time.perf_counter()
            with _engine_factory(args, game_data_dir) as branch:
                restored, load_ms = branch.send(
                    {"cmd": "load_save", "path": str(map_save_path), "lang": "en"}
                )
                restored_signature = _state_signature(restored)
                sample: dict[str, Any] = {
                    "probe": probe_index,
                    "startup_ms": round(branch.startup_ms, 3),
                    "load_ms": round(load_ms, 3),
                    "restored_decision": restored.get("decision"),
                    "exact_visible_signature_match": restored_signature
                    == map_signature,
                    "restored_signature": restored_signature,
                }
                if restored.get("decision") == "map_select":
                    command = {
                        "cmd": "action",
                        "action": "select_map_node",
                        "args": {
                            "col": map_choice["col"],
                            "row": map_choice["row"],
                        },
                    }
                    after_action, action_ms = branch.send(command)
                    sample.update(
                        {
                            "action": command["action"],
                            "action_ms": round(action_ms, 3),
                            "after_action_decision": after_action.get("decision"),
                        }
                    )
                sample["total_ms"] = round(
                    (time.perf_counter() - branch_started) * 1000.0, 3
                )
                branch_samples.append(sample)

        branch_totals = [float(sample["total_ms"]) for sample in branch_samples]
        result["fresh_process_room_boundary_branches"] = {
            "checkpoint_signature": map_signature,
            "selected_node": map_choice,
            "samples": branch_samples,
            "latency": _latency_summary(branch_totals),
            "exact_visible_signature_matches": sum(
                bool(sample["exact_visible_signature_match"])
                for sample in branch_samples
            ),
        }

        # Create the combat checkpoint through normal map navigation rather than
        # the debug enter_room hook, so the restore result reflects native room
        # checkpoint semantics instead of missing map progress.
        with _engine_factory(args, game_data_dir) as combat_source:
            source_map, source_load_ms = combat_source.send(
                {"cmd": "load_save", "path": str(map_save_path), "lang": "en"}
            )
            if source_map.get("decision") != "map_select":
                raise EngineError(
                    f"normal combat source did not restore map_select: {source_map!r}"
                )
            combat_state, source_enter_ms = combat_source.send(
                {
                    "cmd": "action",
                    "action": "select_map_node",
                    "args": {"col": map_choice["col"], "row": map_choice["row"]},
                }
            )
            if combat_state.get("decision") != "combat_play":
                raise EngineError(
                    f"selected Monster node did not enter combat: {combat_state!r}"
                )
            combat_signature = _state_signature(combat_state)
            combat_save_response, combat_save_ms = combat_source.send(
                {"cmd": "write_continue_save", "path": str(combat_save_path)}
            )
            if not combat_save_response.get("success") or not combat_save_path.is_file():
                raise EngineError(f"combat checkpoint write failed: {combat_save_response!r}")

        # This probe is intentionally reported, not asserted: the CLI currently
        # writes a pre-room checkpoint for an in-progress combat.
        with _engine_factory(args, game_data_dir) as restore_probe:
            combat_restored, combat_load_ms = restore_probe.send(
                {"cmd": "load_save", "path": str(combat_save_path), "lang": "en"}
            )
            restored_combat_signature = _state_signature(combat_restored)
            result["combat_checkpoint_semantics"] = {
                "source": "normal_map_node_navigation",
                "source_map_load_ms": round(source_load_ms, 3),
                "source_enter_combat_ms": round(source_enter_ms, 3),
                "write_checkpoint_ms": round(combat_save_ms, 3),
                "load_ms": round(combat_load_ms, 3),
                "checkpoint_signature": combat_signature,
                "restored_signature": restored_combat_signature,
                "exact_visible_signature_match": restored_combat_signature
                == combat_signature,
                "interpretation": (
                    "exact_action_level_restore"
                    if restored_combat_signature == combat_signature
                    else "room_or_run_boundary_restore_only"
                ),
            }

    result["status"] = "pass"
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
    parser.add_argument("--encounter", default="SHRINKER_BEETLE_WEAK")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--branch-probes", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.branch_probes < 0:
        parser.error("--branch-probes cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_benchmark(args)
        exit_code = 0
    except Exception as exc:  # report a machine-readable failure for CI and users
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
