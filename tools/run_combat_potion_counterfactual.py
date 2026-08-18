"""Evaluate one potion by paired same-root turn-boundary search.

The use branch must consume the requested potion as its root action.  The hold
branch forbids that potion for the whole retained turn.  Both branches reuse
the same entrance save, executed prefix, visible-information determinizations,
search budget and zero potion shadow price.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch  # Keep Windows native runtime initialization order stable.


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_policy_guided_mcts import _cache_key, _engine  # noqa: E402
from benchmark_sts2_cli import (  # noqa: E402
    DEFAULT_DOTNET,
    DEFAULT_ENGINE_DLL,
    DEFAULT_GAME_DIR,
    DEFAULT_STS2_LIB,
    EngineError,
    _game_data_dir,
)
from run_combat_mcts_act_sweep import (  # noqa: E402
    _create_base_save,
    _prepare_scenario_save,
    _resolve_optional_precombat_selects,
)
from run_combat_policy_online import _load_policy  # noqa: E402
from run_combat_mcts_comparison import sha256_file_bytes  # noqa: E402
from run_combat_turn_boundary_eval import (  # noqa: E402
    _run_turn_boundary,
    turn_boundary_current_root,
)
from run_heldout_run_combat_comparison import (  # noqa: E402
    DEFAULT_COMBATS,
    DEFAULT_RUN_ID,
    DEFAULT_TARGETS,
    DEFAULT_TRANSITIONS,
    _load_scenarios,
)
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_engine_features import candidate_preview_features  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    headless_state_to_model_sample,
)
from sts2_dataset.combat_potion_evaluator import build_paired_potion_proposal  # noqa: E402
from sts2_dataset.combat_potions import POTION_SPECS_BY_ID  # noqa: E402
from sts2_dataset.util import sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "combat_policy_value_v2_v11_cuda"
    / "20260816T084535Z"
    / "model.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_potion_counterfactual.json"


def _entry(identifier: str) -> str:
    return identifier.split(".", 1)[-1]


def _state_potion_ids(state: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for potion in (state.get("player") or {}).get("potions") or []:
        if not isinstance(potion, dict) or not potion.get("id"):
            continue
        identifier = str(potion["id"])
        ids.add(identifier if identifier.startswith("POTION.") else f"POTION.{identifier}")
    return ids


def _prepare_damage_root(
    engine: Any,
    state: dict[str, Any],
    *,
    enemy_hp_at_most: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Reach a real low-enemy-HP root through one deterministic card action."""

    if state.get("decision") != "combat_play":
        raise EngineError("controlled damage prefix requires combat_play")
    sample = headless_state_to_model_sample(
        state,
        transition_id="potion-counterfactual:controlled-prefix",
        combat_id="potion-counterfactual",
    )
    enemies = {
        str(enemy.get("entity_ref")): enemy
        for enemy in sample["observation"].get("enemies") or []
        if isinstance(enemy, dict) and enemy.get("entity_ref") is not None
    }
    options: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for candidate in sample.get("candidates") or []:
        if candidate.get("action_type") != "play_card":
            continue
        target = enemies.get(str(candidate.get("target_ref")))
        if target is None:
            continue
        preview = candidate_preview_features(sample["observation"], candidate)
        damage = float(preview.get("total_damage") or 0.0)
        remaining = float(target.get("hp") or 0.0) + float(target.get("block") or 0.0) - damage
        if damage > 0.0 and 0.0 < remaining <= enemy_hp_at_most:
            options.append((remaining, candidate, preview))
    if not options:
        raise EngineError(
            f"no one-card controlled prefix can leave an enemy within 1..{enemy_hp_at_most} HP"
        )
    remaining, candidate, preview = max(
        options,
        key=lambda row: (row[0], str(row[1].get("candidate_id") or "")),
    )
    command = candidate_to_headless_command(candidate)
    after, _ = engine.send(command)
    if after.get("decision") != "combat_play":
        raise EngineError("controlled damage prefix unexpectedly left combat_play")
    return after, [command], {
        "candidate": candidate,
        "engine_preview": preview,
        "target_remaining_hp_plus_block": remaining,
    }


def _search(
    args: argparse.Namespace,
    *,
    workers: list[Any],
    entrance_save: Path,
    scenario: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    root_state: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    forbidden_potion_ids: set[str],
    required_potion_ids: set[str] | None = None,
) -> dict[str, Any]:
    return turn_boundary_current_root(
        workers=workers,
        entrance_save=entrance_save,
        enter_command={"cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]},
        root_prefix=root_prefix,
        root_state=root_state,
        model=model,
        tensorizer=tensorizer,
        device=device,
        objective=objective,
        root_top_k=args.root_top_k,
        beam_width=args.beam_width,
        max_player_actions=args.max_player_actions,
        policy_log_weight=args.policy_log_weight,
        continuation_policy_weight=args.continuation_policy_weight,
        minimum_value_advantage=args.minimum_value_advantage,
        minimum_end_turn_advantage=args.minimum_end_turn_advantage,
        unsupported_penalty=args.unsupported_penalty,
        determinization_count=args.determinizations,
        cvar_alpha=args.cvar_alpha,
        cvar_weight=args.cvar_weight,
        search_seed=args.search_seed,
        step=0,
        forbidden_potion_ids=forbidden_potion_ids,
        required_potion_ids=required_potion_ids,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    potion_id = args.potion_id
    spec = POTION_SPECS_BY_ID.get(potion_id)
    if spec is None:
        raise EngineError(f"unknown Ironclad-applicable potion: {potion_id}")
    if spec.evaluator != "paired_turn_boundary":
        raise EngineError(
            f"{potion_id} uses evaluator={spec.evaluator}; this tool only supports paired_turn_boundary"
        )

    checkpoint = args.checkpoint.resolve()
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if device != "cuda":
        raise EngineError(f"CUDA is required; resolved device={device!r}")
    if model.state_value_head is None:
        raise EngineError("paired potion evaluation requires a state value head")
    # The lower evaluator measures tactical consequence.  The upper agent owns
    # the run-level scarcity cost and must not leak it into this comparison.
    objective = CombatObjective.from_config(model.config, potion_cost=0.0)

    scenarios = _load_scenarios(args)
    if args.scenario_ids:
        wanted = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row["scenario_id"] in wanted]
    if len(scenarios) != 1:
        raise EngineError(
            f"paired evaluator requires exactly one scenario, got {len(scenarios)}; pass --scenario-id"
        )
    scenario = scenarios[0]
    scenario = {**scenario, "player": {**scenario["player"]}}
    scenario["player"]["potions"] = [_entry(potion_id)]
    source_hp = int(scenario["player"]["hp"])
    if args.player_hp is not None:
        if args.player_hp < 1 or args.player_hp > int(scenario["player"]["max_hp"]):
            raise EngineError(
                f"--player-hp must be within [1, {scenario['player']['max_hp']}]"
            )
        scenario["player"]["hp"] = int(args.player_hp)

    game_data_dir = _game_data_dir(args.game_dir)
    terminal_rollouts = None
    with tempfile.TemporaryDirectory(prefix="sts2_potion_pair_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        entrance_save = temp / "scenario.save"
        prepared = _prepare_scenario_save(
            args,
            game_data_dir=game_data_dir,
            base_save=base_save,
            scenario=scenario,
            path=entrance_save,
        )
        with ExitStack() as stack:
            root_engine = stack.enter_context(_engine(args, game_data_dir))
            workers = [
                stack.enter_context(_engine(args, game_data_dir))
                for _ in range(args.search_workers)
            ]
            root_state, _ = root_engine.send(
                {"cmd": "load_save", "path": str(entrance_save), "lang": "en"}
            )
            root_state, _ = root_engine.send(
                {"cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]}
            )
            root_state, root_prefix = _resolve_optional_precombat_selects(root_engine, root_state)
            if root_state.get("decision") != "combat_play":
                raise EngineError(f"scenario did not reach combat_play: {root_state!r}")
            controlled_prefix = None
            controlled_commands: list[dict[str, Any]] = []
            if args.prepare_enemy_hp_at_most is not None:
                root_state, commands, controlled_prefix = _prepare_damage_root(
                    root_engine,
                    root_state,
                    enemy_hp_at_most=args.prepare_enemy_hp_at_most,
                )
                controlled_commands.extend(commands)
                root_prefix.extend(commands)
            evaluation_root_signature = sha256_file_bytes(
                json.dumps(root_state, sort_keys=True).encode("utf-8")
            )
            potion_ids = _state_potion_ids(root_state)
            if potion_id not in potion_ids:
                raise EngineError(
                    f"set_player did not expose requested potion {potion_id}; got {sorted(potion_ids)}"
                )
            for worker in workers:
                cached, _ = worker.send(
                    {
                        "cmd": "cache_save",
                        "name": _cache_key(entrance_save),
                        "path": str(entrance_save),
                    }
                )
                if cached.get("type") != "ok":
                    raise EngineError(f"potion worker could not cache save: {cached!r}")

            use_report = _search(
                args,
                workers=workers,
                entrance_save=entrance_save,
                scenario=scenario,
                root_prefix=root_prefix,
                root_state=root_state,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                forbidden_potion_ids=potion_ids - {potion_id},
                required_potion_ids={potion_id},
            )
            hold_report = _search(
                args,
                workers=workers,
                entrance_save=entrance_save,
                scenario=scenario,
                root_prefix=root_prefix,
                root_state=root_state,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                forbidden_potion_ids=potion_ids,
            )
        proposal = build_paired_potion_proposal(
            potion_id=potion_id,
            use_report=use_report,
            hold_report=hold_report,
            state_fingerprint=evaluation_root_signature,
            target_index=args.potion_target_index,
        )
        if args.terminal_rollout:
            use_terminal = _run_turn_boundary(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                initial_prefix_commands=controlled_commands,
                forced_root_candidate=proposal["use_candidate"],
                expected_root_signature=evaluation_root_signature,
            )
            hold_terminal = _run_turn_boundary(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                initial_prefix_commands=controlled_commands,
                forced_root_candidate=proposal["hold_candidate"],
                expected_root_signature=evaluation_root_signature,
                forbidden_potion_ids_until_next_turn={potion_id},
            )
            reserve_terminal = _run_turn_boundary(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
                initial_prefix_commands=controlled_commands,
                forced_root_candidate=proposal["hold_candidate"],
                expected_root_signature=evaluation_root_signature,
                forbidden_potion_ids={potion_id},
            )
            terminal_rollouts = {
                "semantics": {
                    "intervention": (
                        "force_use_now_vs_force_best_hold_root_and_forbid_target_potion_until_"
                        "next_player_turn_then_same_planner"
                    ),
                    "rng": (
                        "same entrance save and seed; action-dependent RNG consumption may diverge "
                        "after the forced intervention"
                    ),
                    "upper_potion_shadow_price": "excluded",
                },
                "use_now": use_terminal,
                "hold_now": hold_terminal,
                "reserve_entire_combat": reserve_terminal,
                "final_hp_gain": float(use_terminal["final_hp"])
                - float(hold_terminal["final_hp"]),
                "hp_loss_reduction": float(hold_terminal["hp_loss"])
                - float(use_terminal["hp_loss"]),
                "same_terminal_status": use_terminal["status"] == hold_terminal["status"],
                "use_vs_reserve_final_hp_gain": float(use_terminal["final_hp"])
                - float(reserve_terminal["final_hp"]),
                "use_vs_reserve_hp_loss_reduction": float(reserve_terminal["hp_loss"])
                - float(use_terminal["hp_loss"]),
                "use_vs_reserve_same_terminal_status": (
                    use_terminal["status"] == reserve_terminal["status"]
                ),
            }
    return {
        "schema_version": "combat-potion-counterfactual-run-0.1.0",
        "generated_at": utc_now(),
        "status": "pass",
        "device": device,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "scenario": {
            **{key: value for key, value in scenario.items() if key != "player"},
            "snapshot": scenario["player"],
            "root": prepared,
            "evaluation_root_signature": evaluation_root_signature,
            "controlled_action_prefix": controlled_prefix,
            "controlled_overrides": {
                "player_hp": int(scenario["player"]["hp"]),
                "source_player_hp": source_hp,
                "is_synthetic_hp_override": args.player_hp is not None,
            },
        },
        "potion": spec.to_dict(),
        "objective": {
            "potion_cost": objective.potion_cost,
            "ownership": "lower_tactical_only_upper_shadow_price_excluded",
        },
        "search_configuration": {
            "root_top_k": args.root_top_k,
            "beam_width": args.beam_width,
            "max_player_actions": args.max_player_actions,
            "determinizations": args.determinizations,
            "search_workers": args.search_workers,
            "search_seed": args.search_seed,
        },
        "proposal": proposal,
        "terminal_rollouts": terminal_rollouts,
        "use_report": use_report,
        "hold_report": hold_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--potion-id", required=True)
    parser.add_argument("--scenario-id", dest="scenario_ids", action="append", required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--player-hp",
        type=int,
        help="Controlled diagnostic override; never treated as a recorded validation state.",
    )
    parser.add_argument(
        "--prepare-enemy-hp-at-most",
        type=int,
        help=(
            "Controlled diagnostic: execute one exact-preview card action before the paired "
            "root so an enemy remains alive at or below this HP+block threshold."
        ),
    )
    parser.add_argument(
        "--potion-target-index",
        type=int,
        help="Controlled diagnostic target filter for targetable potions.",
    )
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--combats", type=Path, default=DEFAULT_COMBATS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--seed", default="potion-counterfactual-v0")
    parser.add_argument("--search-seed", type=int, default=20260817)
    parser.add_argument("--root-top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-player-actions", type=int, default=3)
    parser.add_argument("--search-workers", type=int, default=2)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--policy-log-weight", type=float, default=0.0)
    parser.add_argument("--continuation-policy-weight", type=float, default=0.01)
    parser.add_argument("--minimum-value-advantage", type=float, default=0.0)
    parser.add_argument("--minimum-end-turn-advantage", type=float, default=0.0)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
    parser.add_argument(
        "--terminal-rollout",
        action="store_true",
        help="Force the paired root actions, then run the same planner to combat termination.",
    )
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument(
        "--reuse-turn-plan",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    if args.search_workers < 1:
        raise EngineError("search-workers must be at least 1")
    report = run(args)
    write_json_atomic(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "potion": report["potion"]["potion_id"],
                "scenario": report["scenario"]["scenario_id"],
                "evidence": report["proposal"]["tactical_evidence"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
