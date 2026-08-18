# Combat MCTS V0：预算与复杂度诊断

本记录只验证当前搜索原型的行为，不构成游戏强度结论。所有场景均由
`sts2-cli` 的真实引擎执行，Policy-only 与各搜索预算从同一份战斗入口存档开始。
Act 1–3 的牌组、生命、遗物与药水来自人工数据中同一条 A0 铁甲战士对局在各 Act
首场战斗的玩家可见快照；确定规则和即时资源变化由引擎计算。

## 1. 为什么简单战斗中预算增大后看起来回到 Policy

旧的 Act 1 简单战斗中，MCTS 与当前节点最高 Policy 先验动作的一致率为：

| 请求预算 | 掉血 | 最高先验动作一致率 | 平均搜索深度 | 平均可选根动作数 |
|---:|---:|---:|---:|---:|
| 4 | 0 | 53.8% | 1.46 | 3.92 |
| 16 | 4 | 75.0% | 3.17 | 2.08 |
| 32 | 4 | 100.0% | 3.91 | 2.36 |

预算4并不是可靠的深搜索。实现会保证每个根动作至少访问一次，因此请求预算4时实际
通常只有4–6次模拟；大多数动作恰好只有一个样本，根访问门槛也退化为1。此时一个低先验
动作可能仅凭一次偏乐观的 provisional leaf 估值胜出。预算16/32会给高先验动作更多重复
访问，同时访问门槛升高并排除只有一次样本的动作。简单初始牌组的 Policy 分布高度集中，
因此稳定后的访问分配容易与 Policy 一致。这是“低预算噪声消失 + 强先验”的结果，不是搜索
深度把动作归并成相同决策。

## 2. Act 1–3 代表场景

| 场景 | 敌人数 / 总最大生命 | 牌组规模 / 唯一卡 / 升级牌 | Policy | MCTS-4 | MCTS-32 | MCTS-64 |
|---|---:|---:|---:|---:|---:|---:|
| Act 1 Fuzzy Wurm Crawler | 1 / 56 | 10 / 3 / 0 | 胜，掉0 | 胜，掉0 | 胜，掉0 | 未跑 |
| Act 2 Bowlbugs | 3 / 108 | 21 / 15 / 5 | 胜，掉5 | 胜，掉53 | 胜，掉22 | 胜，掉36 |
| Act 3 Scrolls of Biting | 4 / 130 | 27 / 22 / 11 | 胜，掉52 | 死亡 | 胜，掉14 | 胜，掉36 |

复杂场景没有随着预算增大而回到 Policy。MCTS-32在当前搜索节点选择最高 Policy 先验
动作的比例只有 Act 2 的37.0%和 Act 3 的23.1%；MCTS-64进一步降到18.8%和16.7%。
所以表现的非单调变化不能用“动作越来越与Policy一致”解释。

更直接的原因是当前搜索仍未收敛：

- MCTS-32只有约2.2%–2.4%的 rollout 达到真实引擎终局；
- MCTS-64在 Act 2 达到7.7%，在 Act 3仍只有2.8%；
- 其余叶节点依赖人工数据训练的 action-conditioned resource head；
- 该 head 与 Policy 共享数据和表征，不是独立、充分校准的状态 Value；
- 根选择使用 25% lower-tail CVaR，样本量增加会改变尾部估计与可选动作集合；
- 搜索在多种可见牌堆确定化上优化期望/尾部风险，但整场测试只观察真实引擎中的一个
  隐藏抽牌实现，因此单场掉血本身具有方差。

这解释了为什么32次模拟可能优于64次：更多搜索改变了动作与风险估计，但当前叶价值与
终局覆盖仍不足以保证单调改进。

## 3. 怪物难度与牌组复杂度控制

固定 Act 2 牌组：

| 怪物 | Policy | MCTS-4 | MCTS-32 |
|---|---:|---:|---:|
| Act 1 Fuzzy Wurm Crawler | 胜，掉0 | 胜，掉0 | 胜，掉0 |
| Act 3 Scrolls of Biting | 胜，掉28 | 死亡 | 未完成：进入 `card_select` |

固定 Act 2 Bowlbugs：

| 牌组快照 | 规模 / 唯一卡 / 升级牌 | Policy | MCTS-4 | MCTS-32 |
|---|---:|---:|---:|---:|
| Act 1 | 10 / 3 / 0 | 死亡 | 死亡 | 死亡 |
| Act 2 | 21 / 15 / 5 | 胜，掉5 | 胜，掉53 | 胜，掉22 |
| Act 3 | 27 / 22 / 11 | 胜，掉32 | 胜，掉21 | 胜，掉1 |

当前样本只足以说明两点。第一，怪物规模增加会明显放大低预算搜索的不稳定性。第二，
更强的后期牌组能提供更多有效分支，搜索可能取得更大收益，但分支数也显著增加；固定预算
不能代表相同的每动作证据量。不能由这7个受控场景推出总体胜率结论。

## 4. 下一步门禁

1. 把请求预算改成“每个合法根动作的最小重复访问数 + 总预算”，禁止把每动作一次的
   breadth smoke test 当成可比较的搜索预算。
2. 在固定状态上记录预算8/16/32/64/128的根动作 Q、CVaR、置信区间与选择稳定性；只有
   排名稳定后才扩大整场测试。
3. 接通 `card_select` 等战斗内子决策，否则后期复杂牌组会产生未完成轨迹。
4. 将叶估值改为独立的 state-value / distributional value，并报告精确终局 rollout 比例；
   在此之前，MCTS V0仍属于引擎接入与搜索可行性原型。
5. 用多个固定 Seed 和成对同根测试报告均值、尾部与置信区间，避免用单场掉血判断预算优劣。

复现实验入口：`tools/run_combat_mcts_act_sweep.py`。原始报告位于：

- `artifacts/combat_mcts_act_sweep_smoke.json`
- `artifacts/combat_mcts_act_sweep_budget32.json`
- `artifacts/combat_mcts_act_sweep_budget64.json`
- `artifacts/combat_mcts_act_controls.json`

## 5. 第一轮门禁修复结果

搜索现已把 `minimum_root_visits_per_legal_action` 从1提高为2，并在PUCT接管前按不同
确定化对每个合法根动作强制做两次round-robin访问。请求预算4因此只保留为用户侧标签；
若根节点有10个动作，实际预算至少为20。

相同代表场景复测：

| 场景 | 旧 MCTS-4 | 根动作最少访问2次后的 MCTS-4 |
|---|---:|---:|
| Act 2 Bowlbugs | 胜，掉53 | 胜，掉42 |
| Act 3 Scrolls of Biting | 死亡 | 胜，掉49 |

重复采样消除了最严重的单样本失败，但 Act 2 仍明显弱于 Policy 的掉5血，说明根覆盖只是
必要门禁，无法替代更准确的叶 Value。

`card_select` 也已从 unsupported leaf 改为显式搜索节点：引擎枚举合法卡牌组合，V0使用
均匀先验，MCTS根据选择后的真实引擎状态继续搜索。此前中断的“Act 3怪物 + Act 2牌组”
场景现在可以完整结束，且搜索中 `unsupported_subdecision:card_select=0`：

```text
Policy deterministic fallback: 胜，掉28，16个玩家决策
MCTS-32:                      胜，掉57，33个玩家决策
card_select decisions:        2
```

这个结果只证明子决策的执行与回放闭环已经接通。均匀先验并不代表选择质量；完整人工数据中
共有734条可训练 `card_select`，其中337条位于战斗房间，可用于后续构建选择先验。当前更
关键的瓶颈仍是独立、校准的状态 Value，而不是继续单独增加搜索预算。

新增原始报告：

- `artifacts/combat_mcts_act_sweep_min2_budget4.json`
- `artifacts/combat_mcts_card_select_search.json`

## 6. 独立状态价值与低预算根选择

旧 action-conditioned resource head 只在每个状态的人类已选动作上获得监督，却被搜索用于评价
所有候选动作。为消除这一不匹配，V2 在冻结原策略和动作资源头的前提下，新增直接读取当前可见
状态的独立价值头，预测后续掉血比例、死亡概率、药水消耗和最大生命变化。它表示人工行为延续下的
`V^human(s)`，不是最优价值函数。

V2 使用 10,040 个训练状态，验证/测试分别为 1,306/1,278 个状态。测试集后续掉血比例 MAE 为
0.0518，优于旧动作头在人类已选动作上的 0.0671；但离线误差更低并不保证搜索分布外状态的排序
正确。使用相同的当前搜索实现与预算 32，旧叶估值和独立状态价值的结果为：

| 场景 | Policy | 旧叶估值 | 独立状态价值 |
|---|---:|---:|---:|
| Act 2 Bowlbugs | 掉 5 | 掉 51 | 掉 24 |
| Act 3 Scrolls of Biting | 掉 52 | 掉 56 | 掉 54 |

独立状态价值优于旧叶估值，但仍不足以稳定超过 Policy。进一步检查发现，原根选择直接按少量样本的
均值与 CVaR 排名，经常选择只有两次访问的低先验动作。当前低预算版本因此改为“访问次数优先，
风险分数破同分”；搜索仍可通过持续高价值改变访问分布，但不会轻信单个 rollout 离群点。

相同预算 32 下，访问次数优先的结果为：

| 控制场景 | Policy | MCTS |
|---|---:|---:|
| 代表 Act 1，初始牌组 | 掉 0 | 掉 0 |
| 代表 Act 2，Act 2 牌组 | 掉 5 | 掉 15 |
| 代表 Act 3，Act 3 牌组 | 掉 52 | 掉 11 |
| Act 1 怪物，Act 2 牌组 | 掉 0 | 掉 0 |
| Act 3 怪物，Act 2 牌组 | 掉 28 | 掉 29 |
| Act 2 怪物，Act 1 牌组 | 死亡 | 死亡 |
| Act 2 怪物，Act 3 牌组 | 掉 32 | 掉 9 |

当前证据表明，搜索在候选丰富且存在组合收益的复杂牌组上更可能带来提升；简单战斗没有额外收益，
明显不可胜的牌组也无法由有限搜索挽救，而原本很强的 Policy 决策仍可能被搜索破坏。因此下一阶段
不应只增加预算，而应使用真实引擎搜索结果构造状态价值/动作偏好监督，并加入“搜索证据不足时回退
Policy”的门禁。以上仅为 7 个受控单场场景，不构成胜率结论。

在代表性 Act 2/3 场景上额外使用三个确定化搜索 Seed 复测，Policy 轨迹保持不变：

| 场景 | Policy | MCTS Seed 20260815 | 20260816 | 20260817 |
|---|---:|---:|---:|---:|
| Act 2 Bowlbugs | 掉 5 | 掉 15 | 掉 19 | 掉 54 |
| Act 3 Scrolls of Biting | 掉 52 | 掉 11 | 掉 8 | 掉 19 |

方向在三次复测中一致，但数值方差仍很大。因此搜索教师数据必须保存确定化集合、访问数、Q/CVaR、
终局覆盖率及与 Policy 的差异；只有跨确定化稳定且证据充分的根动作才能作为候选改进标签，其余状态
应回退 Policy 或保留为不确定样本。

新增原始报告：

- `artifacts/combat_mcts_act_sweep_legacy_leaf_current_search_budget32.json`
- `artifacts/combat_mcts_act_sweep_state_value_v2_budget32.json`
- `artifacts/combat_mcts_act_sweep_state_value_v2_visit_root_budget32.json`
- `artifacts/combat_mcts_controls_state_value_v2_visit_root_budget32.json`
- `artifacts/combat_mcts_state_value_visit_root_seed20260816.json`
- `artifacts/combat_mcts_state_value_visit_root_seed20260817.json`

## 7. Search V0.2：配对确定化与 Policy 回退门禁

旧强制覆盖按动作 round-robin，但每次 simulation 都生成新的随机世界，导致动作 A 的第一个样本
和动作 B 的第一个样本并不共享抽牌确定化。Search V0.2 将根世界固定为8个，并把强制覆盖改为：

```text
world 0: action 0, action 1, ..., action N
world 1: action 0, action 1, ..., action N
PUCT:    在同一组8个world中循环
```

根统计先在每个 determinization 内求均值，再跨世界计算均值、死亡概率和 lower-tail CVaR，避免
某个被重复访问的世界获得额外统计权重。最终搜索动作若与最高人工 Policy 先验不同，还必须满足：

- 与 Policy 至少共享4个 determinization；
- 配对死亡概率不高于0.20；
- 配对死亡概率不比 Policy 高0.02以上；
- 配对风险分数不低于 Policy。

否则执行 Policy 动作，并在报告中保存具体回退原因。

固定代表场景结果：

| 场景 | Policy | 旧Search-32（三Seed） | V0.2-32（三Seed） | V0.2-128 |
|---|---:|---:|---:|---:|
| Act 2 Bowlbugs | 掉5 | 掉15 / 19 / 54 | 掉5 / 5 / 5 | 掉10 |
| Act 3 Scrolls of Biting | 掉52 | 掉11 / 8 / 19 | 掉36 / 36 / 36 | 掉23 |

32-budget 下，Act 2 每次都稳定回退到强人工 Policy，消除了原来5至54血的大幅搜索方差；Act 3
也从跨Seed波动变为固定掉36，但牺牲了旧搜索在这一个场景中的最好结果。128-budget 中，非Policy
候选与Policy平均共享约5.3个世界，Act 2/3分别有6/11个搜索动作通过证据门禁，因此系统没有永久
退化成 Policy-only。

值得注意的是，128-budget 几乎没有候选因为死亡风险超限被拒绝，说明当前死亡价值头的区分度与
校准仍不足。V0.2解决的是随机世界不公平和低证据动作覆盖问题，并没有修复状态价值排序。下一步应
冻结这套搜索证据格式，构建分布式终局 Value 与搜索教师样本，而不是继续放宽门禁。

新增原始报告：

- `artifacts/combat_mcts_paired_gate_budget32.json`
- `artifacts/combat_mcts_paired_gate_budget32_seed20260816.json`
- `artifacts/combat_mcts_paired_gate_budget32_seed20260817.json`
- `artifacts/combat_mcts_paired_gate_budget128.json`

## 8. 分布式终局 Value V3 门禁结果

V3 在冻结 V2 策略、动作资源头和状态价值头的前提下，只训练一个21档终局分布头：第0档为死亡，
其余20档表示战斗结束生命比例。全量数据包含10,040个训练状态、1,306个验证状态和1,278个
测试状态；训练7个 epoch 后早停，并使用验证集 NLL 拟合温度参数。

当前结果不满足接入搜索的门禁：

| 指标 | V3 | 朴素或现有基线 |
|---|---:|---:|
| 测试集校准 NLL | 2.8626 | 训练集类别频率 2.8412 |
| 测试集原始终局生命比例 MAE | 0.1610 | 当前生命比例 0.0813 |
| 测试集校准终局生命比例 MAE | 0.1774 | V2 后续掉血比例 MAE 0.0518 |
| 测试集死亡 Brier | 0.0124 | 13个死亡状态 / 1个死亡战斗 |

温度校准把 NLL 从2.9348改善到2.8626，但同时把终局生命期望 MAE 从0.1610拉差到0.1774。
此外，验证集67场战斗中没有死亡战斗，无法为死亡尾部提供可靠校准。结论是：保留 V3 checkpoint
作为失败基线，但不替换当前 MCTS 的 V2 状态价值。后续应在更多失败战斗到来后，重新冻结按
Act与终局类型共同分层的战斗划分；在此之前，不用反复调参掩盖数据覆盖问题。

V3 checkpoint：`artifacts/combat_policy_value_v3/20260815T172629Z/model.pt`。

## 9. 搜索教师证据 V0

搜索现在会为每个根节点生成 `combat-search-teacher-0.1.0` 记录，保存完整结构化根观察、合法候选、
Policy先验、访问分布、均值、CVaR、死亡率、终局覆盖率以及逐 determinization 统计。该记录只是
可审计证据，不会自动成为训练标签；之后可离线应用更严格的共享世界数、终局覆盖和稳定性门禁。

单步真实引擎 smoke test 中，根节点有6个动作，请求预算4被最低覆盖规则提升为12次模拟，每个
动作在相同的2个世界中得到一次评估。因为少于门禁要求的4个共享世界，搜索正确回退到 Policy；
生成记录通过 `schemas/combat_search_teacher_v0.schema.json` 验证。原始报告位于
`artifacts/combat_mcts_teacher_smoke.json`。

同一 Act 2 根状态的1,024次 simulation 测试达到约71.1 simulation/s，建立1,054个树节点，
最大到达8个玩家决策深度；21个根动作中11个覆盖至少4个 determinization。但1,024个叶节点
全部来自 `independent_state_value_leaf`，真实引擎终局覆盖仍为0%。这说明继续增加预算会增加树覆盖
与跨世界证据，但当前分支规模下并不会自动把搜索变成精确终局教师。教师筛选必须显式查看
`leaf_sources` 和 `terminal_fraction`，不能只按 simulation 数量判断质量。原始报告位于
`artifacts/combat_mcts_teacher_act2_budget1024.json`。
