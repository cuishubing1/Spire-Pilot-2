"""Compare policy-only and per-action MCTS on the same real A0 combat."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch  # Keep Windows DLL load order consistent with training tools.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_policy_guided_mcts import (  # noqa: E402
    DEFAULT_CONFIG,
    POST_COMBAT_DECISIONS,
    SEARCH_DECISIONS,
    _candidate_command,
    _engine,
    _enter_command,
    _monster_choice,
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
from run_combat_policy_online import (  # noqa: E402
    _advance_initial_event,
    _load_policy,
    _rank_actions,
    _state_summary,
)
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
)
from sts2_dataset.util import (  # noqa: E402
    canonical_json,
    load_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_mcts_full_battle_comparison.json"


def _start_same_battle(
    engine: Any,
    *,
    seed: str,
    ascension: int,
    entrance_save: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state, _ = engine.send({
        "cmd": "start_run",
        "character": "Ironclad",
        "ascension": ascension,
        "seed": seed,
        "lang": "en",
    })
    map_state, _ = _advance_initial_event(engine, state)
    choice = _monster_choice(map_state)
    if entrance_save is not None:
        saved, _ = engine.send({"cmd": "write_continue_save", "path": str(entrance_save)})
        if not saved.get("success"):
            raise EngineError(f"failed to write combat entrance save: {saved!r}")
    combat_state, _ = engine.send(_enter_command(choice))
    if combat_state.get("decision") != "combat_play":
        raise EngineError(f"failed to enter combat: {combat_state!r}")
    return combat_state, choice, map_state


def _result(status_state: dict[str, Any], *, initial_hp: float, steps: list[dict[str, Any]]) -> dict[str, Any]:
    player = status_state.get("player") or {}
    final_hp = float(player.get("hp") or 0.0)
    decision = str(status_state.get("decision") or "")
    if decision == "game_over":
        status = "victory" if status_state.get("victory") else "death"
    elif decision == "combat_play":
        status = "step_limit"
    elif decision in POST_COMBAT_DECISIONS:
        status = "combat_won"
    else:
        status = f"unsupported_subdecision:{decision or 'unknown'}"
    return {
        "status": status,
        "final_decision": decision,
        "initial_hp": initial_hp,
        "final_hp": final_hp,
        "hp_loss": max(0.0, initial_hp - final_hp),
        "decision_count": len(steps),
        "rounds": max((int(row["before"].get("round") or 0) for row in steps), default=0),
        "steps": steps,
    }


def _run_policy_only(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    model: Any,
    tensorizer: Any,
    device: str,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    with _engine(args, game_data_dir) as engine:
        state, _, _ = _start_same_battle(
            engine, seed=args.seed, ascension=args.ascension
        )
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        root_signature = sha256_file_bytes(canonical_json(state).encode("utf-8"))
        for step in range(args.max_actions):
            if state.get("decision") != "combat_play":
                break
            sample = headless_state_to_model_sample(
                state,
                transition_id=f"policy-only:{step}",
                combat_id="comparison-policy",
            )
            ranked, inference_ms = _rank_actions(
                model, tensorizer, sample, device=device, objective=None
            )
            chosen = max(ranked, key=lambda row: float(row["policy_probability"]))
            command = candidate_to_headless_command(chosen["candidate"])
            before = _state_summary(state)
            state, engine_ms = engine.send(command)
            steps.append({
                "step": step,
                "before": before,
                "chosen_candidate": chosen["candidate"],
                "policy_probability": chosen["policy_probability"],
                "inference_ms": round(inference_ms, 3),
                "engine_ms": round(engine_ms, 3),
                "after": _state_summary(state),
            })
        result = _result(state, initial_hp=initial_hp, steps=steps)
        result["root_signature"] = root_signature
        result["total_inference_ms"] = round(sum(row["inference_ms"] for row in steps), 3)
        result["total_engine_ms"] = round(sum(row["engine_ms"] for row in steps), 3)
        return result


def sha256_file_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _run_mcts(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    config: dict[str, Any],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    max_depth = int(args.max_depth or config["puct"]["maximum_player_decision_depth"])
    with tempfile.TemporaryDirectory(prefix="sts2_full_combat_mcts_") as temp_dir:
        entrance_save = Path(temp_dir) / "entrance.save"
        with _engine(args, game_data_dir) as real_engine:
            state, choice, _ = _start_same_battle(
                real_engine,
                seed=args.seed,
                ascension=args.ascension,
                entrance_save=entrance_save,
            )
            initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
            root_signature = sha256_file_bytes(canonical_json(state).encode("utf-8"))
            with _engine(args, game_data_dir) as worker:
                warm, _ = worker.send({
                    "cmd": "load_save", "path": str(entrance_save), "lang": "en"
                })
                if warm.get("decision") != "map_select":
                    raise EngineError(f"worker warmup restore failed: {warm!r}")
                for step in range(args.max_actions):
                    if state.get("decision") not in SEARCH_DECISIONS:
                        break
                    before = _state_summary(state)
                    search = search_current_root(
                        worker=worker,
                        entrance_save=entrance_save,
                        enter_command=_enter_command(choice),
                        root_prefix=prefix,
                        root_state=state,
                        model=model,
                        tensorizer=tensorizer,
                        device=device,
                        objective=objective,
                        config=config,
                        budget=args.budget,
                        max_depth=max_depth,
                        search_seed=args.search_seed + step * 100003,
                    )
                    command = _candidate_command(search["chosen_candidate"])
                    state, engine_ms = real_engine.send(command)
                    prefix.append(command)
                    steps.append({
                        "step": step,
                        "before": before,
                        "chosen_candidate": search["chosen_candidate"],
                        "engine_ms": round(engine_ms, 3),
                        "after": _state_summary(state),
                        "search": search,
                    })
            result = _result(state, initial_hp=initial_hp, steps=steps)
            result["root_signature"] = root_signature
            search_times = [float(row["search"]["search_wall_ms"]) for row in steps]
            result["total_search_ms"] = round(sum(search_times), 3)
            result["mean_search_ms"] = round(statistics.fmean(search_times), 3) if search_times else None
            result["total_search_simulations"] = sum(
                int(row["search"]["effective_budget"]) for row in steps
            )
            result["total_search_engine_actions"] = sum(
                int(row["search"]["engine_action_count"]) for row in steps
            )
            result["total_prefix_replay_actions"] = sum(
                int(row["search"]["prefix_replay_action_count"]) for row in steps
            )
            return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if model.resource_value_head is None:
        raise EngineError("comparison requires combat_policy_value_v1")
    objective = CombatObjective.from_config(model.config)
    config = load_json(args.config.resolve())
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    policy = _run_policy_only(
        args,
        game_data_dir=game_data_dir,
        model=model,
        tensorizer=tensorizer,
        device=device,
    )
    mcts = _run_mcts(
        args,
        game_data_dir=game_data_dir,
        model=model,
        tensorizer=tensorizer,
        device=device,
        objective=objective,
        config=config,
    )
    if policy["root_signature"] != mcts["root_signature"]:
        raise EngineError("policy and MCTS runs did not start from the same combat state")
    return {
        "schema_version": "combat-mcts-full-battle-comparison-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "seed": args.seed,
        "ascension": args.ascension,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "budget_per_decision": args.budget,
        "max_search_depth": int(args.max_depth or config["puct"]["maximum_player_decision_depth"]),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "policy_only": policy,
        "mcts": mcts,
        "comparison": {
            "same_root": True,
            "hp_loss_delta_mcts_minus_policy": mcts["hp_loss"] - policy["hp_loss"],
            "decision_count_delta_mcts_minus_policy": mcts["decision_count"] - policy["decision_count"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", default="combat-mcts-full-v0")
    parser.add_argument("--search-seed", type=int, default=20260815)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        "seed": report["seed"],
        "budget_per_decision": report["budget_per_decision"],
        "policy_only": {
            key: report["policy_only"][key]
            for key in ("status", "hp_loss", "decision_count")
        },
        "mcts": {
            key: report["mcts"][key]
            for key in ("status", "hp_loss", "decision_count", "total_search_ms")
        },
        "comparison": report["comparison"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
