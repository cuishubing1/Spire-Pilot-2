"""Compare P1 policy-only and exact Top-k one-step lookahead in real combats."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch  # Keep Windows native runtime initialization order stable.


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_policy_guided_mcts import (  # noqa: E402
    POST_COMBAT_DECISIONS,
    SEARCH_DECISIONS,
    _cache_key,
    _candidate_command,
    _engine,
    _determinization,
    _expand_search_node,
    _leaf_outcome,
    _restore_search_root,
    _visible_draw_multiset,
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
    _enter_command,
    _first_a0_ironclad_snapshots,
    _prepare_scenario_save,
    _resolve_optional_precombat_selects,
    _run_policy,
    _scenario_specs,
)
from run_combat_mcts_comparison import _result, sha256_file_bytes  # noqa: E402
from run_combat_policy_online import _load_policy, _rank_actions, _state_summary  # noqa: E402
from sts2_dataset.combat_engine_features import exact_transition_features  # noqa: E402
from sts2_dataset.combat_lookahead import (  # noqa: E402
    COMBAT_ONE_STEP_VERSION,
    apply_exact_terminal_death_veto,
    apply_policy_advantage_gate,
    choose_one_step_candidate,
    one_step_takeover_ineligibility,
    policy_top_k,
    required_search_categories,
    regularized_one_step_score,
)
from sts2_dataset.combat_model import CombatObjective  # noqa: E402
from sts2_dataset.combat_online import (  # noqa: E402
    candidate_to_headless_command,
    first_card_select_candidate,
    headless_state_to_model_sample,
    visible_intent_end_turn_hp_loss,
)
from sts2_dataset.combat_search import (  # noqa: E402
    lower_tail_cvar,
    normalized_policy_entropy,
    risk_adjusted_root_score,
)
from sts2_dataset.legal_actions import enumerate_legal_actions  # noqa: E402
from sts2_dataset.util import load_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


DEFAULT_LATEST = REPO_ROOT / "artifacts" / "combat_policy_p1" / "latest.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "combat_one_step_act_sweep.json"


def _resolve_checkpoint(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    latest = load_json(DEFAULT_LATEST)
    path = Path(latest["checkpoint"])
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def one_step_current_root(
    *,
    worker: Any,
    entrance_save: Path,
    enter_command: dict[str, Any],
    root_prefix: list[dict[str, Any]],
    root_state: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    top_k: int,
    policy_log_weight: float,
    minimum_value_advantage: float,
    minimum_end_turn_advantage: float,
    unsupported_penalty: float,
    determinization_count: int,
    cvar_alpha: float,
    cvar_weight: float,
    search_seed: int,
    step: int,
    minimum_potion_policy_probability: float = 0.0,
    restore_mode: str = "cached_batch_auto_prepared",
    encounter_signature: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample = headless_state_to_model_sample(
        root_state,
        transition_id=f"one-step:root:{step}",
        combat_id="one-step",
        encounter_signature=encounter_signature,
    )
    root_visible_loss = visible_intent_end_turn_hp_loss(sample["observation"])
    retain_markers = ("BARRICADE", "BLUR", "CALIPERS")
    retain_identities = [
        str(value.get("id") or "").upper()
        for group in ("player_powers", "relics")
        for value in sample["observation"].get(group) or []
        if isinstance(value, dict)
    ]
    root_retains_block = any(
        marker in identity
        for identity in retain_identities
        for marker in retain_markers
    )
    ranked, root_inference_ms = _rank_actions(
        model, tensorizer, sample, device=device, objective=None
    )
    required_categories = required_search_categories(sample["observation"])
    semantic_ranked = policy_top_k(
        ranked,
        len(ranked),
        required_categories=required_categories,
    )
    shortlist = policy_top_k(
        ranked,
        top_k,
        required_categories=required_categories,
    )
    if determinization_count < 1:
        raise ValueError("one-step determinization count must be positive")
    if not 0.0 <= cvar_weight <= 1.0:
        raise ValueError("one-step CVaR weight must be in [0, 1]")
    visible_draw = _visible_draw_multiset(root_state)
    determinizations = [
        _determinization(visible_draw, search_seed=search_seed, simulation=index)
        for index in range(determinization_count)
    ]
    evaluations: list[dict[str, Any]] = []
    total_restore_ms = 0.0
    total_engine_ms = 0.0
    total_value_inference_ms = 0.0
    for branch_index, policy_row in enumerate(shortlist):
        candidate = policy_row["candidate"]
        worlds: list[dict[str, Any]] = []
        for world_index, (draw_order, determinization_id) in enumerate(determinizations):
            restored, restore_ms = _restore_search_root(
                worker=worker,
                entrance_save=entrance_save,
                enter_command=enter_command,
                root_prefix=root_prefix,
                draw_order=draw_order,
                mode=restore_mode,
            )
            total_restore_ms += restore_ms
            if restored != root_state:
                raise EngineError("one-step determinization changed the public root")
            successor, engine_ms = worker.send(_candidate_command(candidate))
            total_engine_ms += engine_ms
            successor_node = None
            if successor.get("decision") == "combat_play":
                successor_node = _expand_search_node(
                    successor,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    node_index=(branch_index * determinization_count) + world_index + 1,
                    reuse_entity_encoding=False,
                    encounter_signature=encounter_signature,
                )
                total_value_inference_ms += successor_node.inference_ms
            outcome = _leaf_outcome(
                successor,
                root_state=root_state,
                node=successor_node,
                objective=objective,
                determinization_id=determinization_id,
                unsupported_penalty=unsupported_penalty,
                depth=1,
            )
            worlds.append({
                "determinization_id": determinization_id,
                "successor_decision": str(successor.get("decision") or ""),
                "exact_transition": exact_transition_features(
                    root_state, successor, candidate
                ),
                "outcome": asdict(outcome),
            })
        values = [float(world["outcome"]["value"]) for world in worlds]
        mean_value = statistics.fmean(values)
        tail_value = lower_tail_cvar(values, cvar_alpha)
        risk_value = risk_adjusted_root_score(
            mean_value=mean_value,
            lower_tail_value=tail_value,
            mean_weight=1.0 - cvar_weight,
            cvar_weight=cvar_weight,
        )
        selection_eligible = all(
            world["successor_decision"] != "card_select" for world in worlds
        )
        score = regularized_one_step_score(
            value=risk_value,
            policy_probability=float(policy_row["policy_probability"]),
            policy_log_weight=policy_log_weight,
        )
        evaluations.append({
            "candidate": candidate,
            "policy_probability": float(policy_row["policy_probability"]),
            "selection_score": score,
            "selection_eligible": selection_eligible,
            "mean_value": mean_value,
            "lower_tail_cvar": tail_value,
            "risk_adjusted_value": risk_value,
            "root_visible_end_turn_hp_loss": float(
                (root_visible_loss or {}).get("hp_loss") or 0.0
            ),
            "root_retains_block": root_retains_block,
            "worlds": worlds,
        })
    policy_candidate_id = str(shortlist[0]["candidate"]["candidate_id"])
    for evaluation in evaluations:
        reasons = one_step_takeover_ineligibility(
            evaluation,
            policy_candidate_id=policy_candidate_id,
            minimum_potion_policy_probability=minimum_potion_policy_probability,
        )
        if reasons:
            evaluation["selection_eligible"] = False
            evaluation["selection_ineligible_reasons"] = list(reasons)
    exact_terminal_death_vetoes = apply_exact_terminal_death_veto(evaluations)
    try:
        search_choice = choose_one_step_candidate(evaluations)
        policy_choice = next(
            row for row in evaluations
            if row["candidate"]["candidate_id"] == shortlist[0]["candidate"]["candidate_id"]
        )
        chosen, fallback = apply_policy_advantage_gate(
            search_choice=search_choice,
            policy_choice=policy_choice,
            minimum_advantage=minimum_value_advantage,
            minimum_end_turn_advantage=minimum_end_turn_advantage,
        )
    except ValueError:
        chosen = next(
            row for row in evaluations
            if row["candidate"]["candidate_id"] == shortlist[0]["candidate"]["candidate_id"]
        )
        fallback = "all_shortlisted_actions_require_unsupported_subdecision"
    return {
        "schema_version": COMBAT_ONE_STEP_VERSION,
        "top_k": len(shortlist),
        "raw_candidate_count": len(ranked),
        "semantic_candidate_count": len(semantic_ranked),
        "deduplicated_candidate_count": len(ranked) - len(semantic_ranked),
        "required_categories": list(required_categories),
        "policy_log_weight": policy_log_weight,
        "minimum_value_advantage": minimum_value_advantage,
        "minimum_end_turn_advantage": minimum_end_turn_advantage,
        "minimum_potion_policy_probability": minimum_potion_policy_probability,
        "exact_terminal_death_vetoes": exact_terminal_death_vetoes,
        "determinization_count": determinization_count,
        "cvar_alpha": cvar_alpha,
        "cvar_weight": cvar_weight,
        "information_boundary": "visible_draw_multiset_determinization",
        "policy_candidate": shortlist[0]["candidate"],
        "chosen_candidate": chosen["candidate"],
        "fallback": fallback,
        "root_policy_entropy": normalized_policy_entropy(
            [float(row["policy_probability"]) for row in ranked]
        ),
        "root_inference_ms": round(root_inference_ms, 3),
        "successor_value_inference_ms": round(total_value_inference_ms, 3),
        "engine_restore_ms": round(total_restore_ms, 3),
        "engine_action_ms": round(total_engine_ms, 3),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "evaluations": evaluations,
    }


def _run_one_step(
    args: argparse.Namespace,
    *,
    game_data_dir: Path,
    entrance_save: Path,
    scenario: dict[str, Any],
    model: Any,
    tensorizer: Any,
    device: str,
    objective: CombatObjective,
    encounter_signature: str | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    with _engine(args, game_data_dir) as real_engine, _engine(args, game_data_dir) as worker:
        state, _ = real_engine.send({
            "cmd": "load_save", "path": str(entrance_save), "lang": "en"
        })
        state, _ = real_engine.send(_enter_command(scenario))
        state, precombat_prefix = _resolve_optional_precombat_selects(real_engine, state)
        prefix.extend(precombat_prefix)
        initial_hp = float((state.get("player") or {}).get("hp") or 0.0)
        root_signature = sha256_file_bytes(json.dumps(state, sort_keys=True).encode("utf-8"))
        cached, _ = worker.send({
            "cmd": "cache_save",
            "name": _cache_key(entrance_save),
            "path": str(entrance_save),
        })
        if cached.get("type") != "ok":
            raise EngineError(f"one-step worker could not cache save: {cached!r}")
        for step in range(args.max_actions):
            decision = str(state.get("decision") or "")
            if decision not in SEARCH_DECISIONS:
                break
            before = _state_summary(state)
            if decision == "combat_play":
                lookahead = one_step_current_root(
                    worker=worker,
                    entrance_save=entrance_save,
                    enter_command=_enter_command(scenario),
                    root_prefix=prefix,
                    root_state=state,
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    objective=objective,
                    top_k=args.top_k,
                    policy_log_weight=args.policy_log_weight,
                    minimum_value_advantage=args.minimum_value_advantage,
                    minimum_end_turn_advantage=args.minimum_end_turn_advantage,
                    unsupported_penalty=args.unsupported_penalty,
                    determinization_count=args.determinizations,
                    cvar_alpha=args.cvar_alpha,
                    cvar_weight=args.cvar_weight,
                    search_seed=args.search_seed + step * 100003,
                    step=step,
                    minimum_potion_policy_probability=(
                        args.minimum_potion_policy_probability
                    ),
                    restore_mode=getattr(
                        args,
                        "restore_mode",
                        "cached_batch_auto_prepared",
                    ),
                    encounter_signature=encounter_signature,
                )
                candidate = lookahead["chosen_candidate"]
                command = candidate_to_headless_command(candidate)
            else:
                actions = enumerate_legal_actions(state)
                candidate = first_card_select_candidate(state)
                command = candidate_to_headless_command(candidate)
                lookahead = None
            state, engine_ms = real_engine.send(command)
            prefix.append(command)
            steps.append({
                "step": step,
                "before": before,
                "chosen_candidate": candidate,
                "lookahead": lookahead,
                "engine_ms": round(engine_ms, 3),
                "after": _state_summary(state),
            })
    result = _result(state, initial_hp=initial_hp, steps=steps)
    result["root_signature"] = root_signature
    lookaheads = [row["lookahead"] for row in steps if row["lookahead"] is not None]
    result["total_lookahead_ms"] = round(
        sum(float(row["wall_ms"]) for row in lookaheads), 3
    )
    result["mean_lookahead_ms"] = round(
        statistics.fmean(float(row["wall_ms"]) for row in lookaheads), 3
    ) if lookaheads else None
    result["policy_action_change_count"] = sum(
        row["chosen_candidate"]["candidate_id"]
        != row["policy_candidate"]["candidate_id"]
        for row in lookaheads
    )
    result["lookahead_decision_count"] = len(lookaheads)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, tensorizer, device = _load_policy(checkpoint, args.device)
    if model.state_value_head is None:
        raise EngineError("one-step lookahead requires an independent state value head")
    objective = CombatObjective.from_config(model.config)
    snapshots = _first_a0_ironclad_snapshots(args.transitions.resolve())
    scenarios = _scenario_specs(snapshots, include_controls=False)
    if args.scenario_ids:
        wanted = set(args.scenario_ids)
        scenarios = [row for row in scenarios if row["scenario_id"] in wanted]
        if not scenarios:
            raise EngineError(f"no one-step scenarios matched: {sorted(wanted)}")
    game_data_dir = _game_data_dir(args.game_dir)
    started = time.perf_counter()
    results = []
    with tempfile.TemporaryDirectory(prefix="sts2_one_step_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        for index, scenario in enumerate(scenarios):
            entrance_save = temp / f"scenario-{index}.save"
            root = _prepare_scenario_save(
                args,
                game_data_dir=game_data_dir,
                base_save=base_save,
                scenario=scenario,
                path=entrance_save,
            )
            policy = _run_policy(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
            )
            lookahead = _run_one_step(
                args,
                game_data_dir=game_data_dir,
                entrance_save=entrance_save,
                scenario=scenario,
                model=model,
                tensorizer=tensorizer,
                device=device,
                objective=objective,
            )
            if root["root_signature"] != policy["root_signature"] or (
                policy["root_signature"] != lookahead["root_signature"]
            ):
                raise EngineError(f"one-step root mismatch: {scenario['scenario_id']}")
            results.append({
                **{key: value for key, value in scenario.items() if key != "player"},
                "snapshot": scenario["player"],
                "root": root,
                "policy_only": policy,
                "one_step": lookahead,
            })
            print(json.dumps({
                "completed": scenario["scenario_id"],
                "policy": {"status": policy["status"], "hp_loss": policy["hp_loss"]},
                "one_step": {
                    "status": lookahead["status"],
                    "hp_loss": lookahead["hp_loss"],
                    "changes": lookahead["policy_action_change_count"],
                },
            }, ensure_ascii=False), flush=True)
    return {
        "schema_version": "combat-one-step-act-sweep-0.1.0",
        "lookahead_version": COMBAT_ONE_STEP_VERSION,
        "generated_at": utc_now(),
        "status": "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args.seed,
        "top_k": args.top_k,
        "policy_log_weight": args.policy_log_weight,
        "minimum_value_advantage": args.minimum_value_advantage,
        "minimum_end_turn_advantage": args.minimum_end_turn_advantage,
        "minimum_potion_policy_probability": (
            args.minimum_potion_policy_probability
        ),
        "determinizations": args.determinizations,
        "cvar_alpha": args.cvar_alpha,
        "cvar_weight": args.cvar_weight,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "scenarios": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--seed", default="act-grid-v0")
    parser.add_argument("--search-seed", type=int, default=20260816)
    parser.add_argument("--scenario-ids", nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--policy-log-weight", type=float, default=0.05)
    parser.add_argument("--minimum-value-advantage", type=float, default=0.02)
    parser.add_argument("--minimum-end-turn-advantage", type=float, default=0.15)
    parser.add_argument(
        "--minimum-potion-policy-probability", type=float, default=0.0
    )
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--cvar-alpha", type=float, default=0.5)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-penalty", type=float, default=1.0)
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
        "scenario_count": len(report["scenarios"]),
        "wall_ms": report["wall_ms"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
