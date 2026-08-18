# Combat Policy V0 真实引擎测试

`tools/run_combat_policy_online.py` 将已训练的 Combat Policy V0 接入
`sts2-cli` 使用的真实《杀戮尖塔 2》程序集。每次决策都遵循以下闭环：

```text
读取玩家可见状态 -> 枚举合法动作 -> 网络评分 -> 执行一个动作 -> 重新读取状态
```

它是在线执行冒烟测试，不是搜索模拟器，也不是游戏实力评测。当前不会克隆战斗中间状态、
展开动作分支或使用人工兜底动作。

## 运行

默认从最新 checkpoint 启动一局铁甲战士，并进入 Seed 对应的第一个自然地图战斗：

```powershell
& '.\.venv\Scripts\python.exe' 'tools/run_combat_policy_online.py' `
  --device cpu `
  --ascension 0 `
  --seed NATURALV0A `
  --room-source map `
  --output 'artifacts/combat_policy_online_v0/natural_NATURALV0A.json'
```

若只想检查固定敌人的接口，可使用受控战斗：

```powershell
& '.\.venv\Scripts\python.exe' 'tools/run_combat_policy_online.py' `
  --room-source controlled `
  --encounter SHRINKER_BEETLE_WEAK
```

输出 JSON 保存每一步的可见状态摘要、候选动作数、Top-3 概率、最终动作、网络推理耗时、
引擎执行耗时和动作后的真实状态。原始人工 JSONL 不会被修改。

## 首次冒烟结果

2026-08-15 使用 checkpoint `20260815T055238Z` 完成：

- 1 场受控 Shrinker Beetle：9 个动作，战后净损失 1 HP；
- 3 场自然 A0 首层战斗：全部结束并进入卡牌奖励，净损失 0、2、6 HP；
- 2 场自然 A10 首层战斗：全部结束并进入卡牌奖励，净损失 8、9 HP；
- CPU 网络推理约 1.5--2.3 ms/动作，引擎动作约 6.8--14.9 ms/动作。

这些结果只验证端到端接口和基本执行能力。下一阶段需要建立覆盖普通、精英、Boss、
多敌人、药水和战斗内选牌的版本化场景集，才适合评价策略强弱与泛化能力。
