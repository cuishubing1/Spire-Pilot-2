"""Run Combat Policy V0 against the real Slay the Spire 2 engine.

This is an execution smoke test, not a strength benchmark.  The policy sees
only the visible state exported by Sts2Headless, chooses among engine-derived
legal actions, executes one action, and then observes the engine again.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Import torch before project modules that transitively import pyarrow.  On the
# pinned Windows environment the reverse DLL load order is not reliable.
import torch


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
)
from sts2_dataset.combat_model import (  # noqa: E402
    SUPPORTED_MODEL_VERSIONS,
    CombatObjective,
    CombatPolicyConfig,
    CombatPolicyTransformer,
    numpy_batch_to_torch,
)
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
    visible_intent_end_turn_hp_loss,
)
from sts2_dataset.combat_tensorizer import (  # noqa: E402
    CombatTensorizerV0,
    collate_combat_numpy,
)
from sts2_dataset.combat_tool import rank_combat_actions  # noqa: E402
from sts2_dataset.human import HumanRecordingError  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_LATEST = REPO_ROOT / "artifacts" / "combat_policy_v0" / "latest.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_policy_online_v0.json"


def _advance_initial_event(
    engine: EngineProcess, state: dict[str, Any]
) -> tuple[dict[str, Any], list[float]]:
    """Leave Neow without getting trapped in optional inspect subflows."""
    latencies: list[float] = []
    for _ in range(50):
        decision = state.get("decision")
        if decision == "map_select":
            return state, latencies
        if decision == "event_choice":
            options = [value for value in state.get("options") or [] if not value.get("is_locked")]
            if not options:
                raise EngineError("initial event exposed no unlocked option")
            # Neow can expose an optional first entry that returns to the same
            # menu.  The final unlocked entry is the progressing choice.
            command = {
                "cmd": "action",
                "action": "choose_option",
                "args": {"option_index": options[-1]["index"]},
            }
        elif decision == "card_reward":
            command = {"cmd": "action", "action": "skip_card_reward"}
        elif decision == "bundle_select":
            command = {"cmd": "action", "action": "select_bundle", "args": {"bundle_index": 0}}
        elif decision == "card_select":
            if int(state.get("min_select") or 0) == 0:
                command = {"cmd": "action", "action": "skip_select"}
            else:
                command = {"cmd": "action", "action": "select_cards", "args": {"indices": "0"}}
        else:
            command = {"cmd": "action", "action": "proceed"}
        state, latency = engine.send(command)
        latencies.append(latency)
    raise EngineError(f"failed to leave initial event; final state={state!r}")


def _resolve_checkpoint(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    latest = load_json(DEFAULT_LATEST)
    return Path(latest["checkpoint"]).resolve()


def _load_policy(checkpoint_path: Path, device: str):
    vocabulary_path = checkpoint_path.parent / "vocab.json"
    dataset_index_path = checkpoint_path.parent / "dataset_index.json"
    for required in (checkpoint_path, vocabulary_path, dataset_index_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("vocabulary_sha256") != sha256_file(vocabulary_path):
        raise HumanRecordingError("checkpoint vocabulary fingerprint mismatch")
    if checkpoint.get("dataset_index_sha256") != sha256_file(dataset_index_path):
        raise HumanRecordingError("checkpoint dataset index fingerprint mismatch")
    raw_config = dict(checkpoint["model_config"])
    checkpoint_model_version = raw_config.pop("model_version", None)
    if checkpoint_model_version not in SUPPORTED_MODEL_VERSIONS:
        raise HumanRecordingError("unsupported combat policy checkpoint version")
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )
    model = CombatPolicyTransformer(CombatPolicyConfig(**raw_config)).to(selected_device)
    model.checkpoint_model_version = str(checkpoint_model_version)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tensorizer = CombatTensorizerV0(load_json(vocabulary_path))
    return model, tensorizer, selected_device


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "decision": state.get("decision"),
        "round": state.get("round"),
        "energy": state.get("energy"),
        "player_hp": player.get("hp"),
        "player_block": player.get("block"),
        "hand": [card.get("id") for card in state.get("hand") or []],
        "enemies": [
            {
                "index": enemy.get("index"),
                "id": enemy.get("id"),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "intends_attack": enemy.get("intends_attack"),
                "intents": enemy.get("intents") or [],
                "powers": enemy.get("powers") or [],
            }
            for enemy in state.get("enemies") or []
        ],
    }


def _rank_actions(
    model: CombatPolicyTransformer,
    tensorizer: CombatTensorizerV0,
    sample: dict[str, Any],
    *,
    device: str,
    objective: CombatObjective | None = None,
) -> tuple[list[dict[str, Any]], float]:
    ranked, inference_ms, _ = rank_combat_actions(
        model,
        tensorizer,
        sample,
        device=device,
        objective=objective,
    )
    return ranked, inference_ms


def _objective_from_args(model: CombatPolicyTransformer, args: argparse.Namespace) -> CombatObjective | None:
    if model.resource_value_head is None:
        return None
    return CombatObjective.from_config(
        model.config,
        decision_value_scale=getattr(args, "decision_value_scale", None),
        hp_loss_weight=getattr(args, "hp_loss_weight", None),
        immediate_hp_loss_weight=getattr(args, "immediate_hp_loss_weight", None),
        death_penalty=getattr(args, "death_penalty", None),
        potion_cost=getattr(args, "potion_cost", None),
        max_hp_gain_weight=getattr(args, "max_hp_gain_weight", None),
    )


def _add_objective_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision-value-scale", type=float)
    parser.add_argument("--hp-loss-weight", type=float)
    parser.add_argument("--immediate-hp-loss-weight", type=float)
    parser.add_argument("--death-penalty", type=float)
    parser.add_argument("--potion-cost", type=float)
    parser.add_argument("--max-hp-gain-weight", type=float)


def _engine(args: argparse.Namespace, game_data_dir: Path) -> EngineProcess:
    return EngineProcess(
        dotnet=args.dotnet.resolve(),
        engine_dll=args.engine_dll.resolve(),
        game_data_dir=game_data_dir,
        sts2_lib=args.sts2_lib.resolve(),
        timeout_s=args.timeout,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint_path, args.device)
    objective = _objective_from_args(model, args)
    game_data_dir = _game_data_dir(args.game_dir)
    report: dict[str, Any] = {
        "schema_version": "combat-policy-online-run-0.1.0",
        "generated_at": utc_now(),
        "status": "running",
        "configuration": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "device": device,
            "game_data_dir": str(game_data_dir),
            "character": args.character,
            "ascension": args.ascension,
            "seed": args.seed,
            "encounter": args.encounter,
            "room_source": args.room_source,
            "max_steps": args.max_steps,
        },
        "steps": [],
    }
    try:
        with _engine(args, game_data_dir) as engine:
            report["engine_startup_ms"] = round(engine.startup_ms, 3)
            state, start_ms = engine.send({
                "cmd": "start_run",
                "character": args.character,
                "ascension": args.ascension,
                "seed": args.seed,
                "lang": "en",
            })
            state, neow_ms = _advance_initial_event(engine, state)
            report["start_run_ms"] = round(start_ms, 3)
            report["neow_actions"] = len(neow_ms)
            if args.room_source == "map":
                choices = state.get("choices") or []
                choice = next(
                    (value for value in choices if value.get("type") == "Monster"),
                    choices[0] if choices else None,
                )
                if choice is None:
                    raise EngineError("map exposed no selectable first-floor node")
                state, enter_ms = engine.send({
                    "cmd": "action",
                    "action": "select_map_node",
                    "args": {"col": choice["col"], "row": choice["row"]},
                })
                report["selected_map_node"] = choice
            else:
                state, enter_ms = engine.send({
                    "cmd": "enter_room",
                    "type": "combat",
                    "encounter": args.encounter,
                })
            report["enter_combat_ms"] = round(enter_ms, 3)
            if state.get("decision") != "combat_play":
                raise EngineError(f"enter_room did not produce combat_play: {state!r}")
            initial_hp = (state.get("player") or {}).get("hp")
            combat_id = f"online:{args.seed}:{args.encounter}"
            for step_index in range(args.max_steps):
                if state.get("decision") != "combat_play":
                    break
                before = _state_summary(state)
                sample = headless_state_to_model_sample(
                    state,
                    transition_id=f"{combat_id}:{step_index}",
                    combat_id=combat_id,
                )
                ranked, inference_ms = _rank_actions(
                    model, tensorizer, sample, device=device, objective=objective
                )
                selected = ranked[0]
                candidate = selected["candidate"]
                command = candidate_to_headless_command(candidate)
                state, engine_ms = engine.send(command)
                report["steps"].append({
                    "step": step_index,
                    "before": before,
                    "candidate_count": len(sample["candidates"]),
                    "selected": selected,
                    "top3": ranked[:3],
                    "command": command,
                    "inference_ms": round(inference_ms, 3),
                    "engine_ms": round(engine_ms, 3),
                    "after": _state_summary(state),
                })
            final_hp = (state.get("player") or {}).get("hp")
            final_decision = state.get("decision")
            if final_decision == "combat_play":
                status = "max_steps"
            elif final_decision in {"card_reward", "map_select", "proceed", "rewards"}:
                status = "combat_won"
            elif final_decision == "game_over":
                status = "combat_won" if state.get("victory") else "combat_lost"
            else:
                status = "combat_finished_unclassified"
            inference_values = [float(row["inference_ms"]) for row in report["steps"]]
            engine_values = [float(row["engine_ms"]) for row in report["steps"]]
            report.update({
                "status": status,
                "final_decision": final_decision,
                "actions": len(report["steps"]),
                "initial_hp": initial_hp,
                "final_hp": final_hp,
                "hp_loss": initial_hp - final_hp
                if isinstance(initial_hp, int) and isinstance(final_hp, int)
                else None,
                "mean_inference_ms": round(statistics.fmean(inference_values), 3)
                if inference_values else None,
                "mean_engine_ms": round(statistics.fmean(engine_values), 3)
                if engine_values else None,
                "final_state": _state_summary(state),
            })
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", default="COMBATPOLICYV0")
    parser.add_argument("--encounter", default="SHRINKER_BEETLE_WEAK")
    parser.add_argument(
        "--room-source", choices=("map", "controlled"), default="map",
        help="enter the first natural map combat or a controlled encounter fixture",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _add_objective_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(json.dumps({
        "status": report["status"],
        "actions": report.get("actions", len(report.get("steps", []))),
        "hp_loss": report.get("hp_loss"),
        "mean_inference_ms": report.get("mean_inference_ms"),
        "mean_engine_ms": report.get("mean_engine_ms"),
        "output": str(output),
        "error": report.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] not in {"error", "max_steps"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
