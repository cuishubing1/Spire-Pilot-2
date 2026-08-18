# Combat Mechanic Guidance V0

本实验验证结构化 Combat Directive 能否利用玩家可见的敌人机制，改善人工预训练战斗网络的
选择。机制不以自然语言直接输入网络，而是由规则适配器读取敌人 ID、Power、意图、生命和当前
合法动作，动态生成目标偏置、候选偏置或临时约束。

## 三类机制

### Bowlbug Rock：完全格挡后晕眩

触发依据为 `MONSTER.BOWLBUG_ROCK + POWER.IMBALANCED_POWER + 可见攻击意图`。指导器先根据
当前能量和手牌中公开的即时格挡数做一个小型背包计算；只有确认本回合能够完成全额格挡时，才
进入防御计划。计划开始后，在填满格挡缺口前暂时排除非格挡动作，避免网络中途改打攻击牌。
如果全防不可行，则完全退回原策略，允许直接击杀 Rock。

### Terror Eel：阈值前准备，触发后爆发

阈值直接读取 `POWER.SHRIEK_POWER.amount`；当前 A0 引擎实例为70，而不是写死75。实验版在前
两个回合禁止会过早跨过阈值的动作，之后提高伤害动作优先级，尝试利用触发回合与后续行动窗口
快速击杀。这是策略假设，不是确定性规则。

### Overgrowth Crawlers：优先 Shrinker Beetle

官方遭遇 `OVERGROWTH_CRAWLERS` 会生成 Shrinker Beetle 与 Fuzzy Wurm Crawler。两者同时存活
时，对 Shrinker 的目标偏置为正，对 Fuzzy 的偏置为负；Shrinker 死亡后指令自动失效。

## 受控测试

使用 `tools/run_combat_mechanic_guidance.py`，每个 profile 从同一 sts2-cli 存档入口独立运行。
牌组、遗物、药水和生命来自对应人工战斗的第一条状态；默认策略与机制策略只在 Directive 上
不同。当前 checkpoint 为人工数据预训练的 Policy/Value V2。

三个控制 Seed 的掉血变化定义为：

```text
机制指导掉血 - 默认掉血
```

负数代表改善。

| 场景 | Seed 1 | Seed 2 | Seed 3 | 平均 | 其他现象 |
|---|---:|---:|---:|---:|---|
| Bowlbugs，普通牌组 | 0 | 0 | -7 | -2.3 | 不可全防时回退；1个 Seed 新增晕眩 |
| Bowlbugs，防御牌组 | -21 | -8 | 0 | -9.7 | 2/3 Seed 成功触发 Rock 晕眩 |
| Terror Eel | 0 | +10 | -5 | +1.7 | 三次均通关；固定准备回合不稳定 |
| Shrinker + Fuzzy | -24 | -19 | -15 | -19.3 | 2次由死亡变通关；3/3 改善 |

## 结论

1. 目标优先级是当前最适合上层 Agent 传给 Combat Tool 的机制信息，Shrinker 场景表现稳定。
2. “完全格挡触发奖励”必须同时包含可行性门禁和回合内计划承诺；单纯提高防御牌概率会放弃
   更好的击杀路线，并可能增加掉血。
3. Terror Eel 的阈值能够精确识别，但“三回合内能否击杀”是跨回合能力估计问题。固定等待两
   回合只在部分抽牌顺序下有效，下一版应由短程 MCTS 或独立 burst-readiness Value 判断何时触发。
4. Directive 是可检验的策略建议，不是真值标签。规则被正确识别不代表对应建议一定提高结果。

这些结果仍是少量固定 Seed 的工程验证，不构成稳定强度结论。
