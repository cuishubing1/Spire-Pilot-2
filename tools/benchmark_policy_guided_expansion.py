"""Benchmark policy-guided exact expansion of every legal combat action."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    EngineProcess,
    _game_data_dir,
    _skip_neow,
)
from run_combat_policy_online import _load_policy, _rank_actions, _resolve_checkpoint  # noqa: E402
from sts2_dataset.combat_engine_features import (  # noqa: E402
    candidate_preview_features,
    exact_transition_features,
)
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
)
from sts2_dataset.util import sha256_file, utc_now, write_json_atomic  # noqa: E402


def _engine(args: argparse.Namespace, game_data_dir: Path) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet.resolve(),
        engine_dll=args.engine_dll.resolve(),
        game_data_dir=game_data_dir,
        sts2_lib=args.sts2_lib.resolve(),
        timeout_s=args.timeout,
    )


def _monster_choice(state: dict[str, Any]) -> dict[str, Any]:
    try:
        return next(value for value in state.get("choices") or [] if value.get("type") == "Monster")
    except StopIteration as exc:
        raise EngineError("map state contains no Monster choice") from exc


def _enter_command(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": "action",
        "action": "select_map_node",
        "args": {"col": choice["col"], "row": choice["row"]},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    game_data_dir = _game_data_dir(args.game_dir)
    with tempfile.TemporaryDirectory(prefix="sts2_policy_expansion_") as temp_dir:
        entrance_save = Path(temp_dir) / "entrance.save"
        with _engine(args, game_data_dir) as source:
            state, _ = source.send({
                "cmd": "start_run",
                "character": "Ironclad",
                "ascension": args.ascension,
                "seed": args.seed,
                "lang": "en",
            })
            state, _ = _skip_neow(source, state)
            choice = _monster_choice(state)
            save_result, _ = source.send({"cmd": "write_continue_save", "path": str(entrance_save)})
            if not save_result.get("success"):
                raise EngineError(f"failed to write entrance save: {save_result!r}")
            root_state, _ = source.send(_enter_command(choice))
            if root_state.get("decision") != "combat_play":
                raise EngineError(f"Monster node did not enter combat: {root_state!r}")

        sample = headless_state_to_model_sample(
            root_state, transition_id="policy-expansion:root", combat_id="policy-expansion"
        )
        ranked, inference_ms = _rank_actions(
            model, tensorizer, sample, device=device, objective=None
        )
        ranked_by_index = {int(value["candidate_index"]): value for value in ranked}
        expansions: list[dict[str, Any]] = []
        with _engine(args, game_data_dir) as worker:
            # Pay initialization once. Every measured action uses reload_save.
            warm_state, _ = worker.send({"cmd": "load_save", "path": str(entrance_save), "lang": "en"})
            if warm_state.get("decision") != "map_select":
                raise EngineError(f"worker warmup restore failed: {warm_state!r}")
            for candidate in sample["candidates"]:
                started = time.perf_counter()
                restored, reload_ms = worker.send({
                    "cmd": "reload_save", "path": str(entrance_save), "lang": "en"
                })
                if restored.get("decision") != "map_select":
                    raise EngineError(f"worker reload failed: {restored!r}")
                branch_root, enter_ms = worker.send(_enter_command(choice))
                if branch_root != root_state:
                    raise EngineError("policy expansion did not reproduce the exact combat root")
                successor, action_ms = worker.send(candidate_to_headless_command(candidate))
                policy_row = ranked_by_index[int(candidate["candidate_index"])]
                expansions.append({
                    "candidate_index": candidate["candidate_index"],
                    "candidate": candidate,
                    "policy_probability": policy_row["policy_probability"],
                    "preview_features": candidate_preview_features(sample["observation"], candidate),
                    "exact_transition_features": exact_transition_features(
                        branch_root, successor, candidate
                    ),
                    "successor_decision": successor.get("decision"),
                    "reload_ms": round(reload_ms, 3),
                    "enter_combat_ms": round(enter_ms, 3),
                    "action_ms": round(action_ms, 3),
                    "total_branch_ms": round((time.perf_counter() - started) * 1000.0, 3),
                })
        branch_ms = [float(value["total_branch_ms"]) for value in expansions]
        return {
            "schema_version": "policy-guided-expansion-0.1.0",
            "generated_at": utc_now(),
            "status": "pass",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "seed": args.seed,
            "ascension": args.ascension,
            "device": device,
            "enemy_ids": [value.get("id") for value in root_state.get("enemies") or []],
            "candidate_count": len(expansions),
            "policy_inference_ms": round(inference_ms, 3),
            "mean_branch_ms": round(statistics.fmean(branch_ms), 3),
            "branches_per_second": round(1000.0 / statistics.fmean(branch_ms), 3),
            "expansions": sorted(
                expansions, key=lambda value: float(value["policy_probability"]), reverse=True
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", default="policy-guided-expansion-v1")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "artifacts" / "policy_guided_expansion.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps({
        key: report[key] for key in (
            "status", "seed", "enemy_ids", "candidate_count", "policy_inference_ms",
            "mean_branch_ms", "branches_per_second",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
