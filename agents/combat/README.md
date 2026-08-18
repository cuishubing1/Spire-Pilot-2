# Combat Agent

回合边界 Beam Search 的受控开发实验、并行执行与失败消融见
[`docs/combat-turn-boundary-search-v0.zh-CN.md`](../../docs/combat-turn-boundary-search-v0.zh-CN.md)。

上层 Agent 与战斗网络之间的第一版调用契约见
[`docs/combat-tool-v0.zh-CN.md`](../../docs/combat-tool-v0.zh-CN.md)。它由
`CombatDirective V0 + Combat Tool V0` 组成：上层只传递有类型的目标、资源预算和动作偏好，
下层继续在引擎枚举的合法动作中评分，并返回 Top-k、风险、分数拆解与重新规划信号。
针对 Bowlbug Rock、Terror Eel 和 Overgrowth Crawlers 的动态机制指令实验见
[`docs/combat-mechanic-guidance-v0.zh-CN.md`](../../docs/combat-mechanic-guidance-v0.zh-CN.md)。
候选精确引擎特征适配器与可见牌序 determinization 的 Top-k 一步展开见
[`docs/combat-policy-p1-one-step.zh-CN.md`](../../docs/combat-policy-p1-one-step.zh-CN.md)。

战斗 Agent 的目标是根据玩家可见的结构化状态，在合法动作集合中选择出牌、目标、
药水或结束回合动作。

当前计划路线：

1. 冻结 observation、semantic action 与 legal-action mask 规范。
2. 使用经过审计的 canonical 人工轨迹训练行为克隆基线。
3. 扩展为共享的实体级策略—价值—风险模型。
4. 在固定种子和真实引擎环境中评估，再逐步加入价值引导搜索。

当前已具备可执行的行为克隆主干、P1公开引擎候选特征适配器、独立状态 Value、类型化
Combat Tool 0.2.0 和实验性逐动作搜索。P1.3 默认仍由 Policy 选择动作；候选资源值只作诊断，
状态 Value 负责风险与重新规划信号。离线动作一致率、Value MAE 与单场搜索门禁都不等同于
真实游戏强度，搜索仅在固定根状态和回归场景中实验。

初版上层 Agent 可通过 `sts2-combat-tool` 的 JSON 请求接口调用下层网络；每次响应包含模型、
词表、数据索引和源数据指纹，并明确区分精确规则、on-policy 诊断值、实验性目标重排和未执行
搜索。当前推荐 checkpoint 为
`artifacts/combat_policy_p1_v13_cuda/20260817T151201Z/model.pt`。

## Combat Dataset V1

Combat Dataset V1.2 使用混合粒度划分。测试集按完整 `run_id` 隔离：测试对局中的全部
Act 1–3 战斗只能进入 test。排除测试对局后，其余战斗再以 `combat_id` 为单位，在每个 Act
内分别划分 train 和 validation。相同 `combat_id` 的全部动作始终属于同一个集合。

因此 test 可直接用于固定局外计划下的连续爬塔评测，且不会把同一牌组、Seed 或玩家风格的
其他战斗泄漏到训练集；train/validation 仍保留战斗级拆分，以充分利用有限人工数据。

配置位于 `config/combat_dataset_v1.json`。当前默认只选择 Ironclad、Act 1–3 和
`is_training_eligible=true` 的 `combat_play` transition。测试 run 由配置显式冻结；其余战斗
按 Act 以 90%/10% 分到 train/validation。增量构建保留已有归属，只将新增的非测试战斗
补入对应 Act 当前最缺少的集合。

```powershell
& .\.venv\Scripts\sts2-data.exe import-human "人工数据"
& .\.venv\Scripts\sts2-data.exe validate-human
& .\.venv\Scripts\sts2-data.exe build-combat-dataset
& .\.venv\Scripts\sts2-data.exe validate-combat-dataset
& .\.venv\Scripts\sts2-data.exe build-combat-examples
& .\.venv\Scripts\sts2-data.exe validate-combat-examples
& .\.venv\Scripts\sts2-data.exe build-combat-vocab
```

派生输出保存在被 Git 忽略的 `data/human/combat_v1/`：

- `combats.parquet`：每场战斗的 Act、楼层、split 与动作数；
- `transitions.parquet`：带 `combat_id`、split 和来源指纹的战斗动作；
- `manifest.json`：全局及逐 Act 的战斗数/动作数分布和文件哈希。

## Model Contract V0

模型输入契约由 `schemas/combat_model_v0.schema.json` 固定，当前版本为：

- Combat Observation `combat-observation-0.1.0`；
- Combat Action `combat-action-0.1.0`；
- Model Sample `combat-model-sample-0.1.0`。

Observation V0 只从 `observation_json` 投影玩家可见信息，包含全局战斗数值、手牌、无序
抽牌/弃牌/消耗牌摘要、敌人及意图、遗物、药水、Power 和 Orb。它会移除本地化名称、程序集
来源和引擎对象引用；`entity_ref`/`lineage_ref` 只作为当前状态内的实体绑定键，不能作为需要
学习的类别 ID。

Action V0 不使用固定长度的卡牌分类器，而是逐个评分当前合法候选：

```text
play_card(card instance, optional enemy)
use_potion(potion instance, optional enemy/self)
discard_potion(potion instance)
end_turn
```

合法候选中的 `target_index` 会先解析为敌人的稳定 `combat_id`。旧记录中自用药水的
`target_combat_id="0"` 会规范化为 `target_kind=self`，从而与没有显式目标参数的合法候选对齐。
每个训练样本必须且只能有一个候选与人类动作匹配，否则构建立即失败。

模型样本输出位于 `data/human/combat_v1/model_v0/samples.parquet`，包含投影后的 observation、
动态候选集合以及 `label_index`。Python 侧可使用
`sts2_dataset.combat_contract.iter_combat_model_samples(split)` 读取解码后的样本。

`combat_tensorizer.py` 提供不依赖训练框架的 NumPy 张量化层。词表只从 train split 建立，
后续新增训练实体只追加、不改变已有索引；验证集和测试集中的未见实体映射为 `<UNK>`。
卡牌在不同牌堆区域共享身份 ID，区域则由独立 entity-type 表示。每个样本输出实体类型、
实体身份、64维哈希数值特征、64维哈希类别特征、候选动作类型、来源/目标实体索引、mask
和行为克隆标签。类别通道只编码记录中真实存在的字段，例如房间类型、卡牌类型/稀有度/
目标类型、关键词、敌人意图类型和可见状态；不会假设遭遇名称、Boss 阶段或反事实价值标签。

## Combat Policy Transformer V0

`combat_model.py` 实现第一版局部战斗网络：4层、`d_model=128` 的共享实体
Transformer 编码玩家、手牌、牌堆、敌人、遗物、药水和状态实体；候选动作头结合全局
状态、来源实体、目标实体、动作类型与目标类型，对当前合法候选逐个评分。Padding 候选在
网络内部被强制 mask，因此模型不会把概率分配给非法动作。

V0 只训练行为克隆策略头，暂不把 Value、风险预测或搜索混入基线。PyTorch 是可选训练依赖：

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[train]"
& .\.venv\Scripts\sts2-train.exe
```

训练产物保存在被 Git 忽略的 `artifacts/combat_policy_v0/<run_id>/`。重新评估最近一次
checkpoint：

```powershell
$latest = Get-Content .\artifacts\combat_policy_v0\latest.json | ConvertFrom-Json
& .\.venv\Scripts\sts2-train.exe --evaluate-checkpoint $latest.checkpoint --split test
```

第一轮完整实验的设计、指标和限制见
[`docs/combat-policy-v0.zh-CN.md`](../../docs/combat-policy-v0.zh-CN.md)。

## Search V0（实验性）

当前已具备“人工策略先验 + sts2-cli 真实引擎前缀回放 + determinized PUCT”的单战斗根
节点实验。搜索保留访问次数、均值、CVaR、死亡率、战后生命、药水和最大生命等独立字段，
供后续诊断与蒸馏；这些统计暂不循环输入基础策略网络。首个32/64次固定根状态门禁均通过，
但叶节点仍使用临时动作条件资源估值，因此 Search V0 只能视为可执行原型，不能视为已验证
的强战斗 Agent。详细接口、速度和限制见
[`docs/sts2-cli-simulator-gate.zh-CN.md`](../../docs/sts2-cli-simulator-gate.zh-CN.md)。
当前 prepared-save、紧凑重放、共享实体编码和搜索阶段剖析见
[`docs/combat-search-performance-v0.zh-CN.md`](../../docs/combat-search-performance-v0.zh-CN.md)。

搜索现已接入整场战斗的逐动作重规划，并增加最大生命规则门禁：真实引擎差值直接计入，
学习到的未来正增长只能在 `Feed`、`Chosen Cheese` 等公开机制允许的上界内生效。同一场
A0战斗中，4次烟雾预算一度将战后净掉血从4降到0，但16/32次预算均回到4，尚未显示稳定
搜索收益。下一模型门禁是独立状态Value分布，而不是继续堆叠搜索预算。
