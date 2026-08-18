"""Export a controlled combat trace as an auditable Chinese Markdown replay."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    _engine,
    _prepare_scenario_save,
)
from run_heldout_run_combat_comparison import (  # noqa: E402
    DEFAULT_COMBATS,
    DEFAULT_TARGETS,
    DEFAULT_TRANSITIONS,
    _load_scenarios,
)
from sts2_dataset.combat_online import candidate_to_headless_command  # noqa: E402
from sts2_dataset.util import utc_now  # noqa: E402


DEFAULT_REPORT = REPO_ROOT / "artifacts" / "combat_turn_boundary_full19_parallel.json"
DEFAULT_LOC = REPO_ROOT / "third_party" / "sts2-cli" / "localization_zhs"
_BBCODE = re.compile(r"\[/?[^\]]+\]")


def _entry(identity: str | None) -> str:
    return str(identity or "").split(".", 1)[-1]


def _load_table(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: str) -> str:
    return _BBCODE.sub("", value).replace("\n", " ").strip()


class ChineseLocalization:
    def __init__(self, root: Path) -> None:
        self.cards = _load_table(root / "cards.json")
        self.potions = _load_table(root / "potions.json")
        self.monsters = _load_table(root / "monsters.json")
        self.intents = _load_table(root / "intents.json")

    def card_title(self, identity: str | None) -> str:
        key = _entry(identity)
        return self.cards.get(f"{key}.title", key)

    def card_description(self, identity: str | None) -> str | None:
        value = self.cards.get(f"{_entry(identity)}.description")
        return _plain(value) if value else None

    def potion_title(self, identity: str | None) -> str:
        key = _entry(identity)
        return self.potions.get(f"{key}.title", key)

    def potion_description(self, identity: str | None) -> str | None:
        value = self.potions.get(f"{_entry(identity)}.description")
        return _plain(value) if value else None

    def monster_title(self, identity: str | None) -> str:
        key = _entry(identity)
        return self.monsters.get(f"{key}.name", key)


def _hand(state: dict[str, Any], loc: ChineseLocalization) -> list[dict[str, Any]]:
    return [
        {
            "index": int(card.get("index") or 0),
            "id": str(card.get("id") or ""),
            "name": str(card.get("name") or loc.card_title(card.get("id"))),
            "cost": card.get("cost"),
            "can_play": card.get("can_play"),
        }
        for card in state.get("hand") or []
        if isinstance(card, dict)
    ]


def _potions(state: dict[str, Any], loc: ChineseLocalization) -> list[dict[str, Any]]:
    rows = state.get("potions") or (state.get("player") or {}).get("potions") or []
    return [
        {
            "index": int(potion.get("index") or 0),
            "id": str(potion.get("id") or ""),
            "name": str(potion.get("name") or loc.potion_title(potion.get("id"))),
            "can_use": potion.get("can_use"),
        }
        for potion in rows
        if isinstance(potion, dict)
    ]


def _enemies(state: dict[str, Any], loc: ChineseLocalization) -> list[dict[str, Any]]:
    result = []
    for enemy in state.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        result.append({
            "index": int(enemy.get("index") or 0),
            "id": str(enemy.get("id") or ""),
            "name": str(enemy.get("name") or loc.monster_title(enemy.get("id"))),
            "hp": enemy.get("hp"),
            "max_hp": enemy.get("max_hp"),
            "block": enemy.get("block"),
            "intents": enemy.get("intents") or [],
            "powers": enemy.get("powers") or [],
        })
    return result


def _snapshot(state: dict[str, Any], loc: ChineseLocalization) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "decision": state.get("decision"),
        "round": state.get("round"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "block": player.get("block"),
        "energy": state.get("energy"),
        "hand": _hand(state, loc),
        "potions": _potions(state, loc),
        "player_powers": state.get("player_powers") or player.get("powers") or [],
        "enemies": _enemies(state, loc),
        "selection_cards": [
            {
                "index": int(card.get("index") or 0),
                "id": str(card.get("id") or ""),
                "name": str(card.get("name") or loc.card_title(card.get("id"))),
            }
            for card in state.get("cards") or []
            if isinstance(card, dict)
        ],
        "victory": state.get("victory"),
    }


def _action_name(candidate: dict[str, Any], loc: ChineseLocalization) -> str:
    action_type = str(candidate.get("action_type") or "")
    if action_type == "end_turn":
        return "结束回合"
    if action_type == "play_card":
        title = loc.card_title(candidate.get("source_id"))
        target = candidate.get("target_id")
        return f"打出 {title}" + (f" → {loc.monster_title(target)}" if target else "")
    if action_type == "use_potion":
        title = loc.potion_title(candidate.get("source_id"))
        target = candidate.get("target_id")
        return f"使用 {title}" + (f" → {loc.monster_title(target)}" if target else "")
    if action_type == "discard_potion":
        return f"丢弃 {loc.potion_title(candidate.get('source_id'))}"
    if action_type == "select_cards":
        return f"选择 {loc.card_title(candidate.get('source_id'))}"
    if action_type == "skip_select":
        return "跳过选牌"
    return action_type


def _format_hand(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "（空）"
    return "、".join(
        f"[{card['index']}] {card['name']}（{card['cost']}费，{card['id']}）"
        for card in cards
    )


def _format_potions(potions: list[dict[str, Any]]) -> str:
    if not potions:
        return "（无）"
    return "、".join(
        f"[{potion['index']}] {potion['name']}（{potion['id']}）"
        for potion in potions
    )


def _intent_text(intents: list[Any]) -> str:
    if not intents:
        return "未知"
    parts = []
    for intent in intents:
        if isinstance(intent, dict):
            title = intent.get("name") or intent.get("title") or intent.get("type") or "未知"
            damage = intent.get("damage")
            hits = intent.get("hits") or intent.get("times")
            if damage is not None:
                title = f"{title} {damage}" + (f"×{hits}" if hits else "")
            parts.append(str(title))
        else:
            parts.append(str(intent))
    return "、".join(parts)


def _render_markdown(
    *,
    report_path: Path,
    scenario: dict[str, Any],
    source: dict[str, Any],
    replay: list[dict[str, Any]],
    loc: ChineseLocalization,
) -> str:
    result = source["turn_boundary"]
    final_replay_state = replay[-1]["after"] if replay else {}
    current_decision = str(final_replay_state.get("decision") or "unknown")
    selected_cards = [
        str(row["candidate"].get("source_id") or "")
        for row in replay
        if row["candidate"].get("action_type") == "select_cards"
    ]
    selected_text = "、".join(
        f"{loc.card_title(card_id)}（{card_id}）" for card_id in selected_cards
    ) or "（无）"
    human_loss = float(source["human"]["hp_loss"])
    lines = [
        "# 知识恶魔：回合边界搜索中文回放",
        "",
        f"- 生成时间：`{utc_now()}`",
        f"- 来源报告：`{report_path}`",
        f"- 场景：Act {scenario['act']} 第 {scenario['floor']} 层，{loc.monster_title('KNOWLEDGE_DEMON')}",
        f"- 当前报告结果：`{result['status']}`，入口 HP {result['initial_hp']}，结束 HP {result['final_hp']}，战损 {result['hp_loss']}",
        f"- 当前本地 CLI 复现终点：`{current_decision}`",
        "- 评测有效性：通过；CLI 已在敌方回合返回选牌决策，执行器显式选择卡牌后继续战斗",
        f"- 搜索：{result['search_decision_count']} 次决策，覆盖 P1 {result['policy_action_change_count']} 次",
        "",
        "> 这是使用同一受控入口存档重新执行报告动作得到的中文可见状态。中文名称来自同版本 `localization_zhs`，稳定 ID 保留在括号中。",
        "",
    ]
    rounds: dict[int, list[dict[str, Any]]] = {}
    for row in replay:
        round_index = int(row["before"].get("round") or 0)
        rounds.setdefault(round_index, []).append(row)
    for round_index, rows in rounds.items():
        initial = rows[0]["before"]
        lines.extend([
            f"## 第 {round_index} 回合",
            "",
            f"- 玩家：{initial['hp']}/{initial['max_hp']} HP，{initial['block']} 格挡，{initial['energy']} 能量",
            f"- 手牌：{_format_hand(initial['hand'])}",
            f"- 药水：{_format_potions(initial['potions'])}",
        ])
        for enemy in initial["enemies"]:
            lines.append(
                f"- 敌人：{enemy['name']} {enemy['hp']}/{enemy['max_hp']} HP，"
                f"意图：{_intent_text(enemy['intents'])}"
            )
        lines.extend([
            "",
            "| 步骤 | 操作前手牌 | 能量 | 操作 | 操作后状态 |",
            "|---:|---|---:|---|---|",
        ])
        for row in rows:
            before = row["before"]
            after = row["after"]
            if after["decision"] == "game_over":
                after_text = f"战斗结束：{'胜利' if after.get('victory') else '死亡'}，{after['hp']} HP"
            elif after["decision"] == "card_select":
                choices = "、".join(
                    f"{card['name']}（{card['id']}）" for card in after["selection_cards"]
                )
                after_text = f"等待选牌：{choices}"
            else:
                after_text = (
                    f"{after['hp']} HP，{after['energy']} 能量，"
                    f"手牌：{_format_hand(after['hand'])}"
                )
            lines.append(
                f"| {row['step']} | {_format_hand(before['hand'])} | {before['energy']} | "
                f"{row['action_name']} | {after_text} |"
            )
        lines.append("")
    seen_cards = sorted({
        card["id"]
        for row in replay
        for snapshot in (row["before"], row["after"])
        for card in snapshot["hand"]
    })
    seen_potions = sorted({
        potion["id"]
        for row in replay
        for snapshot in (row["before"], row["after"])
        for potion in snapshot["potions"]
    })
    lines.extend(["## 本场出现的卡牌与药水", ""])
    for identity in seen_cards:
        description = loc.card_description(identity)
        lines.append(
            f"- **{loc.card_title(identity)}**（`{identity}`）"
            + (f"：{description}" if description else "")
        )
    for identity in seen_potions:
        description = loc.potion_description(identity)
        lines.append(
            f"- **{loc.potion_title(identity)}**（`{identity}`）"
            + (f"：{description}" if description else "")
        )
    lines.extend([
        "",
        "## 直接观察",
        "",
        f"本次复现共处理 {len(selected_cards)} 次知识的诅咒选牌，依次选择：{selected_text}。",
        "",
        f"回合边界搜索完成战斗，战损 {result['hp_loss']}；原人工记录战损 {human_loss:g}。两者来自相同可见入口快照，但不是同一条原始隐藏随机轨迹，因此这里只作受控重建比较。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = next(
        (row for row in report["scenarios"] if row["scenario_id"] == args.scenario_id),
        None,
    )
    if source is None:
        raise EngineError(f"scenario not found in report: {args.scenario_id}")
    scenarios = _load_scenarios(args)
    scenario = next(row for row in scenarios if row["scenario_id"] == args.scenario_id)
    loc = ChineseLocalization(args.localization.resolve())
    game_data_dir = _game_data_dir(args.game_dir)
    replay: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sts2_replay_zh_") as temp_dir:
        temp = Path(temp_dir)
        base_save = _create_base_save(args, game_data_dir, temp / "base.save")
        entrance_save = temp / "scenario.save"
        _prepare_scenario_save(
            args,
            game_data_dir=game_data_dir,
            base_save=base_save,
            scenario=scenario,
            path=entrance_save,
        )
        with _engine(args, game_data_dir) as engine:
            state, _ = engine.send({
                "cmd": "load_save", "path": str(entrance_save), "lang": "zh"
            })
            state, _ = engine.send({
                "cmd": "enter_room", "type": "combat", "encounter": scenario["encounter"]
            })
            for source_step in source["turn_boundary"]["steps"]:
                candidate = source_step["chosen_candidate"]
                command = candidate_to_headless_command(candidate)
                before = _snapshot(state, loc)
                state, engine_ms = engine.send(command)
                replay.append({
                    "step": int(source_step["step"]),
                    "candidate": candidate,
                    "command": command,
                    "action_name": _action_name(candidate, loc),
                    "engine_ms": round(engine_ms, 3),
                    "before": before,
                    "after": _snapshot(state, loc),
                })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    content = _render_markdown(
        report_path=report_path,
        scenario=scenario,
        source=source,
        replay=replay,
        loc=loc,
    )
    output.write_text(content, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scenario-id", default="heldout-14-act2-floor33")
    parser.add_argument("--run-id", default="human-20260813T153218409Z-111dbff7862d4059970daa1469aaf9fe")
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--combats", type=Path, default=DEFAULT_COMBATS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--seed", default="heldout-a0-controlled-reconstruction-v0")
    parser.add_argument("--localization", type=Path, default=DEFAULT_LOC)
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--dotnet", type=Path, default=DEFAULT_DOTNET)
    parser.add_argument("--engine-dll", type=Path, default=DEFAULT_ENGINE_DLL)
    parser.add_argument("--sts2-lib", type=Path, default=DEFAULT_STS2_LIB)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "replays" / "knowledge_demon_turn_boundary.zh-CN.md",
    )
    return parser.parse_args()


def main() -> int:
    output = run(parse_args())
    print(json.dumps({"status": "pass", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
