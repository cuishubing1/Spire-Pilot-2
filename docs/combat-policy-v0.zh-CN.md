# Combat Policy Transformer V0

## 目标与边界

V0 是一个从人工战斗轨迹学习“在当前合法候选中选择人类动作”的行为克隆基线。它不是
World Model，也不在训练时调用模拟器；离线动作一致率只验证模型能否吸收人工策略，不能
直接解释为通关率或超越人类。

模型严格使用 Combat Observation V0 已记录的玩家可见字段：

- 全局：Act、楼层、进阶、回合、房间类型、生命、格挡、能量、金币和牌堆计数；
- 实体：手牌实例、无序抽牌/弃牌/消耗牌摘要、敌人及意图、遗物、药水、玩家 Power 和 Orb；
- 动作：Mod 枚举的 `play_card`、`use_potion`、`discard_potion` 与 `end_turn` 合法候选。

没有进入网络的内容包括隐藏牌序、RNG、audit state、未来信息、遭遇战名称、Boss 阶段标签、
人工思维链、动作价值和反事实结果。

## 网络结构

每个可见游戏实体由四部分相加得到 token：实体区域类型、实体身份 embedding、64维公开数值
特征、64维公开类别特征。敌人类型直接由 `MONSTER.*` 身份 embedding 表示；
`Monster/Elite/Boss`、卡牌类型与稀有度、目标类型、关键词和敌人意图类型来自记录本身。

共享的4层 Transformer（`d_model=128`、4个 attention head）编码全部实体。动作头不使用固定
卡牌分类表，而是组合全局 token、动作来源 token、目标 token、动作类型和目标类型，逐个评分
当前合法候选。padding 候选始终被 mask。模型共953,217个可训练参数。

目前不设遭遇专用网络，也不设 Boss adapter。训练集只有67场 Boss 战，且没有显式阶段标签；
V0 先用房间类型和当前敌人实体进行条件化，待确认共享模型的系统性 Boss 误差后再决定是否
专门化。

## 数据与训练

当前 Ironclad Combat Dataset V1.1 包含872场战斗、16,721个动作，并采用“测试集按完整
对局隔离、训练/验证按战斗细分”的混合划分：

| Split | 战斗 | 动作 |
|---|---:|---:|
| Train | 690 | 13,146 |
| Validation | 77 | 1,512 |
| Test | 105 | 2,063 |

测试集冻结5个此前未参与P0/P1训练的完整通关run，覆盖A0、A2、A3、A4和A6；每个run的
Act 1–3战斗全部进入test。其余战斗按Act以90%/10%划分train/validation。同一`combat_id`
不跨split，测试run也不得出现在其他split。词表只从train建立。

训练使用 AdamW、合法候选内 label smoothing、验证 NLL early stopping。配置位于
`config/combat_policy_v0.json`。

## 首轮离线结果（旧 Combat Dataset V1.0，历史记录）

以下结果来自旧的战斗级80%/10%/10%划分，仅作为历史开发记录；它们不能当作V1.1
run-held-out test的结果。V1.1需要重新训练P0/P1后再报告正式指标。

旧实验在第7轮取得最低验证 NLL，并在第11轮触发 early stopping：

| 指标 | Validation | Test |
|---|---:|---:|
| Top-1 动作一致率 | 57.43% | 57.82% |
| Top-3 动作一致率 | 86.98% | 89.83% |
| 按战斗宏平均 Top-1 | 58.85% | 58.03% |
| 合法动作率 | 100% | 100% |
| 随机合法候选期望 Top-1 | 18.00% | 15.89% |

测试集分项：

| 维度 | Top-1 |
|---|---:|
| Act 1 / Act 2 / Act 3 | 60.85% / 57.83% / 54.20% |
| 普通 / 精英 / Boss | 57.40% / 60.53% / 57.24% |
| 出牌 / 结束回合 / 使用药水 | 50.93% / 91.63% / 0.00% |

药水只有18个测试标签且0次命中，是当前最明显的监督短板；训练数据没有
`discard_potion` 正标签，因此不能声称模型学会了主动丢弃药水。

旧划分允许同一完整对局中的不同战斗进入不同split，因此上述数字可能受相似牌组和玩家风格
影响。V1.1已用完整run隔离test，但正式游戏强度仍需真实引擎在线评估。

## 运行

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[train]"
& .\.venv\Scripts\sts2-train.exe
```

训练产物写入 `artifacts/combat_policy_v0/<run_id>/`，包括 checkpoint、冻结词表、训练配置和
完整分项指标。重新评估最近一次 checkpoint：

```powershell
$latest = Get-Content .\artifacts\combat_policy_v0\latest.json | ConvertFrom-Json
& .\.venv\Scripts\sts2-train.exe --evaluate-checkpoint $latest.checkpoint --split test
```

新增人工记录后依次运行 import、validate、combat dataset、model examples 和 vocabulary 的增量
构建，再开始新训练；原始 JSONL 和旧 checkpoint 都不被修改。
