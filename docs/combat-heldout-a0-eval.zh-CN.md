# A0 留出局战斗基线（Combat Dataset V1.1）

本轮使用 Combat Dataset V1.1 明确保留的完整 A0 测试局：

```text
run_id: human-20260813T153218409Z-111dbff7862d4059970daa1469aaf9fe
seed:   MYY13RPE9Z
result: human victory, Act 1--3, floor 48
```

P0、Value V1/V2 和 P1 均在 NVIDIA GeForce RTX 5080 上重新训练；本报告不使用此前误生成的
CPU checkpoint。

## 1. 逐场受控重建

每场战斗从 HumanRecorder 的首个可训练决策恢复玩家可见入口状态，包括 HP、最大 HP、金币、
牌组、遗物、药水和 Act，并通过 `sts2-cli` 启动对应 EncounterModel。P0 与 P1 加载同一个生成
存档，根状态哈希和引擎 RNG 根完全一致。P1 一步展开也从同一根开始，只使用玩家可见抽牌堆
多重集合构造两个 determinization。Human 战损来自原始记录，因此 Human 与网络之间不是
隐藏 RNG 完全一致的反事实比较。

| 指标 | Human | P0 | P1 | P1 + 一步展开 |
|---|---:|---:|---:|---:|
| 战斗数 | 19 | 19 | 19 | 19 |
| 存活完成 | 19 | 15 | 15 | 16 |
| 战斗内死亡 | 0 | 4 | 4 | 3 |
| 累计入口到结束战损 | 151 | 737 | 737 | 510 |
| 平均战损 | 7.95 | 38.79 | 38.79 | 26.84 |

分 Act 平均战损：

| Act | Human | P0 | P1 | P1 + 一步展开 |
|---|---:|---:|---:|---:|
| 1 | 5.50 | 25.13 | 25.13 | 21.88 |
| 2 | 7.57 | 46.43 | 46.43 | 25.71 |
| 3 | 13.50 | 52.75 | 52.75 | 38.75 |

P0 与 P1 共比较 373 个实际战斗决策，选择动作差异为 0；所选动作概率的平均绝对差约
0.00064，最大约 0.00280。P1 当前只训练精确候选特征适配器，没有形成足以改变在线动作排序
的改进。

一步展开在413个战斗决策中覆盖Policy 75次，使 Waterfall Giant 和 Soul Nexus 从死亡变为存活，
但仍死于 Terror Eel 和 Knowledge Demon，并使原本能击败 Aeonglass 的 Policy 变为死亡。总体
收益明确但不稳定，说明一步真实转移能修正即时资源错误，当前状态 Value 对多回合机制和部分
Boss仍会产生错误排序。全部规划耗时约126.3秒，平均约306毫秒/决策。

失败战的覆盖优势并不普遍贴近0.02门槛：Terror Eel 三次覆盖的平均预测优势约3.21，Knowledge
Demon五次约4.73，Aeonglass二十五次约0.70且最大5.43。仅提高门控阈值无法消除主要错误；更
可能的原因是单步叶节点Value无法正确评价回合结束、牌堆循环、能力铺设和Boss多回合机制。

19 场中有 18 场生成的敌人模型组合与人工记录一致。`BOWLBUGS_NORMAL` 在相同 encounter 内部
随机生成了 Nectar 而非原记录的 Silk，因此该场仅能用于 P0/P1 同根比较，不能作为严格的
Human 同敌人比较。

完整结果：

- `artifacts/heldout_a0_human_p0_p1_one_step_combat_comparison.json`

## 2. 固定局外计划的连续 A0

P1 从头开始运行原 Seed，路线、抓牌、事件、商店和营火动作采用人类记录计划，战斗由 P1
控制。所有战斗状态连续继承，不在战斗间恢复人工 HP 或牌组。

```text
status:       death
max Act:      1
max floor:    11
combats:      5
combat steps: 52
```

| 楼层 | 类型 | 引擎实际敌人 | 入口 HP | 结束 HP | 净变化 |
|---:|---|---|---:|---:|---:|
| 2 | 普通 | Nibbit | 80 | 68 | -12 |
| 3 | 普通 | Slimes | 68 | 63 | -5 |
| 6 | 普通 | Shrinker Beetle | 57 | 56 | -1 |
| 8 | 精英 | Bygone Effigy | 56 | 21 | -35 |
| 11 | 精英 | Byrdonis | 21 | 0 | -21 |

该运行正常执行至死亡，没有脚本阻塞。虽然 Seed 与非战斗计划固定，headless 引擎的 RNG 消耗
与原始游戏记录并不逐字节一致，因此实际遭遇序列不同。这个实验衡量的是固定计划下的连续
在线生存能力，而不是对人类轨迹的完全重放。

加入一步展开后，使用相同 Seed 和同一份局外计划重新运行：

```text
status:                   death
max Act:                  2
max floor:                28
combats:                  13
combat steps:             206
one-step action changes:  42
mean lookahead latency:    90.54 ms
```

它跨过此前第11层死亡点、击败第一幕Boss，并在第二幕第28层 Bowlbugs 战斗死亡。连续结果证明
独立战斗中的部分收益能够转化为跨战斗生存收益，但尚不足以完成第二幕。

固定计划兼容层共使用7次 fallback 和8次辅助子决策。其中 Membership Card 购买触发了
`sts2-cli` 空引用异常，按“引擎拒绝的局外购买”跳过；另一次因删牌没有发生而省略配套
`card_select`。这些均记录在 trace 中，但意味着本实验是“尽量遵循原计划”的连续评测，不是
完全相同局外状态的严格复现。

完整结果：

- `artifacts/combat_policy_p1_v11_cuda_fixed_plan_heldout_a0.json`
- `artifacts/combat_policy_p1_one_step_v11_cuda_fixed_plan_heldout_a0.json`

## 3. 当前判断

新增数据和 run-held-out 测试使泛化问题暴露得更真实：离线 Top-1 约 55% 并不能转化为可靠的
在线资源管理，差距随 Act 和战斗复杂度扩大。P1 的精确即时特征本身没有打破 P0 的动作排序；
而 `P1 + top-k 一步真实引擎展开` 已显示出实质但不稳定的收益。下一步不应直接扩大成高预算
MCTS，而应先分析 Aeonglass、Terror Eel 和 Knowledge Demon 的错误 Value 排序，并比较一步
展开的候选价值与真实后续战损。目前证据已排除“主要只是门控过松”，应优先建立短时域
回报目标或2--3个玩家决策的有限展开，再决定是否需要更大搜索。
