# Combat Policy V0 固定局外计划完整对局测试

## 协议

测试使用真实《杀戮尖塔 2》v0.107.1 引擎。Combat Policy V0 接管所有
`combat_play` 动作；每场战斗结束后的 HP、牌组、遗物、药水和金币由引擎自然继承，
不会在战斗之间重置。

局外控制器按 HumanRecorder 的 canonical 轨迹固定路线、事件、商店、休息点和卡牌选择。
若 AI 的战斗动作改变 RNG，使原记录对象不再出现，则使用预先固定的确定性 fallback，并在
结果中逐次记录。地图坐标不可 fallback；原路线不可达时测试应停止。

当前只有一局完整 Ironclad A0 轨迹，因此不能声称从验证/测试集中选出了三局独立 A0。
首次测试选择一局 A0、一局 A1 和一局 A2 的完整人工轨迹作为局外计划模板，三者都包含至少
一个 validation 或 test 战斗，并统一以 A0 执行。由于数据按战斗划分，每条完整轨迹还同时
含有 train 战斗，所以这不是严格的整局 held-out 评测。

## 首次结果

Checkpoint：`20260815T055238Z`。

| Seed | 来源模板 | 执行难度 | 结果 | 最远楼层 | 战斗数 | 网络动作 | fallback / support |
|---|---:|---:|---|---:|---:|---:|---:|
| UR87D1ZH5Q | A0 | A0 | 死亡 | Act 1，9层精英 | 5 | 61 | 4 / 1 |
| DC04XFCJKP | A1 | A0 | 死亡 | Act 1，8层精英 | 5 | 70 | 0 / 0 |
| 0KCSG2BCA3 | A2 | A0 | 死亡 | Act 1，9层普通战斗 | 5 | 49 | 1 / 2 |

三局均完成四场战斗，在第五场死亡，未进入 Act 2。关键 HP 轨迹：

- `UR87D1ZH5Q`：80 → 74 → 60 → 51 → 12 → 0；
- `DC04XFCJKP`：80 → 66 → 54 → 52 → 24 → 0；
- `0KCSG2BCA3`：80 → 72 → 74 → 68 → 26 → 0。

结果说明当前网络已经能够连续处理单敌人与多敌人战斗，但 HP 损失在中期明显累积，
尤其在第4场战斗分别损失39、28、42 HP，随后已没有足够生存余量。该实验不能证明
Boss能力，也不能把死亡全部归因于战斗网络：其中两局包含确定性局外 fallback，且当前网络
尚无 Value/Risk head、搜索或药水可靠性训练。

完整逐步日志位于 `artifacts/combat_policy_fixed_plan_a0.json`。

## 运行

```powershell
& '.\.venv\Scripts\python.exe' 'tools/run_fixed_plan_policy.py' `
  --device cpu `
  --ascension 0 `
  --output 'artifacts/combat_policy_fixed_plan_a0.json'
```
