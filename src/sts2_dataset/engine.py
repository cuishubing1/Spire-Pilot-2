from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .constants import AUDIT_ROOT, DOTNET, ENGINE_PROJECT, THIRD_PARTY
from .types import AuditRef
from .util import canonical_json, read_zstd_json, sha256_bytes, write_zstd_json


class EngineProtocolError(RuntimeError):
    pass


class Sts2Engine:
    def __init__(self, config: dict[str, Any], run_id: str):
        self.config = config
        self.run_id = run_id
        self.process: subprocess.Popen[str] | None = None
        self.protocol_version: str | None = None
        self.seed: str | None = None
        self.character = config["character"]
        self.ascension = int(config["ascension"])
        self.actions: list[dict[str, Any]] = []
        self._replay_base_native_save: str | None = None
        self._replay_base_action_count = 0
        self._exchanges: list[dict[str, Any]] = []
        self._stderr_lines: Queue[str] = Queue()

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise EngineProtocolError("Engine is already running")
        env = dict(os.environ)
        game_data = Path(self.config["game_dir"]) / "data_sts2_windows_x86_64"
        env["STS2_GAME_DIR"] = str(game_data)
        env["STS2_LIB"] = str(THIRD_PARTY / "lib")
        self.process = subprocess.Popen(
            [str(DOTNET), "run", "--no-build", "--project", str(ENGINE_PROJECT)],
            cwd=str(THIRD_PARTY),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stderr is not None
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        ready = self._read_json_line()
        if ready.get("type") != "ready":
            raise EngineProtocolError(f"Unexpected engine greeting: {ready}")
        self.protocol_version = str(ready.get("version"))
        if self.protocol_version != self.config["sts2_cli_protocol"]:
            raise EngineProtocolError(
                f"Protocol mismatch: expected {self.config['sts2_cli_protocol']}, got {self.protocol_version}"
            )
        return ready

    def _drain_stderr(self) -> None:
        assert self.process and self.process.stderr
        for line in self.process.stderr:
            self._stderr_lines.put(line.rstrip())

    def stderr_tail(self, limit: int = 100) -> list[str]:
        lines = []
        while len(lines) < limit:
            try:
                lines.append(self._stderr_lines.get_nowait())
            except Empty:
                break
        return lines[-limit:]

    def _read_json_line(self) -> dict[str, Any]:
        assert self.process and self.process.stdout
        while True:
            line = self.process.stdout.readline()
            if line == "":
                tail = "\n".join(self.stderr_tail())
                raise EngineProtocolError(f"Engine exited before a JSON response. stderr:\n{tail}")
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise EngineProtocolError(f"Invalid engine JSON: {line[:300]}") from exc

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process is None:
            raise EngineProtocolError("Engine is not running")
        assert self.process.stdin
        self.process.stdin.write(canonical_json(request) + "\n")
        self.process.stdin.flush()
        response = self._read_json_line()
        self._exchanges.append({"request": request, "response": response})
        if response.get("type") == "error":
            raise EngineProtocolError(
                f"{response.get('message', 'Unknown engine error')} (request={canonical_json(request)})"
            )
        return response

    def drain_exchanges(self) -> list[dict[str, Any]]:
        result, self._exchanges = self._exchanges, []
        return result

    def reset(self, *, seed: str) -> dict[str, Any]:
        if self.process is None:
            self.start()
        self.seed = seed
        self.actions = []
        self._replay_base_native_save = None
        self._replay_base_action_count = 0
        return self.send(
            {"cmd": "start_run", "character": self.character, "ascension": self.ascension, "seed": seed, "lang": "en"}
        )

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        request = {"cmd": "action", "action": action["action"]}
        if action.get("args"):
            request["args"] = action["args"]
        response = self.send(request)
        self.actions.append({"action": action["action"], "args": action.get("args") or {}})
        return response

    def get_map(self) -> dict[str, Any] | None:
        response = self.send({"cmd": "get_map"})
        return response if response.get("type") == "map" else None

    def get_state(self) -> dict[str, Any]:
        return self.send({"cmd": "get_state"})

    def snapshot(self, step_id: int, expected_state_hash: str | None = None) -> AuditRef:
        if self.seed is None:
            raise EngineProtocolError("Cannot snapshot before reset")
        directory = AUDIT_ROOT / self.run_id
        directory.mkdir(parents=True, exist_ok=True)
        native_path = directory / f".{step_id:06d}.native.json"
        save_result = self.send({"cmd": "write_continue_save", "path": str(native_path)})
        if not save_result.get("success") or not native_path.exists():
            raise EngineProtocolError(f"Native checkpoint failed: {save_result}")
        native_json = native_path.read_text(encoding="utf-8")
        native_path.unlink()
        room_type = save_result.get("room_type")
        if room_type in {"MapRoom", None}:
            self._replay_base_native_save = native_json
            self._replay_base_action_count = len(self.actions)
        bundle = {
            "format": "sts2-audit-v1",
            "run_id": self.run_id,
            "step_id": step_id,
            "character": self.character,
            "ascension": self.ascension,
            "seed": self.seed,
            "actions": list(self.actions),
            "action_prefix_sha256": sha256_bytes(canonical_json(self.actions).encode("utf-8")),
            "expected_state_hash": expected_state_hash,
            "native_save": native_json,
            "native_save_room_type": room_type,
            "replay_base_native_save": self._replay_base_native_save,
            "replay_base_action_count": self._replay_base_action_count,
        }
        target = directory / f"{step_id:06d}.checkpoint.zst"
        digest = write_zstd_json(target, bundle, exclusive=True)
        return AuditRef(path=str(target.relative_to(AUDIT_ROOT.parent)).replace("\\", "/"), sha256=digest, format="sts2-audit-v1", step_id=step_id)

    def restore(self, audit_ref: AuditRef) -> dict[str, Any]:
        bundle_path = AUDIT_ROOT.parent / audit_ref.path
        bundle = read_zstd_json(bundle_path)
        self.restart()
        actions = bundle["actions"]
        # Native v0.107.1 saves are retained for audit, but loading one can consume
        # hidden RNG differently when a future Unknown map node is resolved. Exact
        # counterfactual restoration therefore uses the deterministic seed plus the
        # full action/control-command prefix.
        state = self.reset(seed=bundle["seed"])
        replay_save = AUDIT_ROOT / self.run_id / ".replay-stabilize.native.json"
        replay_save.parent.mkdir(parents=True, exist_ok=True)
        # The collector materializes and snapshots the initial reset observation
        # before the first policy action as well.
        state = self.get_state()
        if state.get("decision") == "map_select":
            self.get_map()
        self.send({"cmd": "write_continue_save", "path": str(replay_save)})
        for action in actions:
            if "cmd" in action:
                state = self.raw_command(action)
            else:
                state = self.step(action)
            # Reproduce Collector._materialize exactly at each decision boundary:
            # pump/export current state, export the visible map when applicable,
            # then serialize the native audit checkpoint. All three operations can
            # advance headless continuations in v0.107.1, so omitting any one makes
            # a long prefix diverge around Neow/event/treasure selectors.
            state = self.get_state()
            if state.get("decision") == "map_select":
                self.get_map()
            self.send({"cmd": "write_continue_save", "path": str(replay_save)})
        replay_save.unlink(missing_ok=True)
        return state

    def restart(self) -> None:
        self.close()
        self.process = None
        self.start()

    def close(self) -> None:
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            if process.poll() is None and process.stdin:
                process.stdin.write('{"cmd":"quit"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
        except Exception:
            process.kill()
        finally:
            if process.poll() is None:
                process.kill()

    def raw_command(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.send(request)
        if request.get("cmd") in {"set_player", "enter_room", "set_draw_order"}:
            self.actions.append(dict(request))
        return response
