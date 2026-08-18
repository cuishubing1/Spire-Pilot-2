# Combat P2 阶段总结、后续方向与接口文件

日期：2026-08-19

## 1. 当前定位

当前 P2 已经可以作为初版上层 Agent 调用的下层战斗能力，但还不能视为成熟的强战斗
Agent。它最适合承担的角色是：读取玩家可见的结构化战斗状态，在引擎枚举的合法动作中给出
稳定的 Policy 排序、风险诊断和重新规划信号；遇到特殊机制、高风险或资源决策时，再由上层
Agent 提供结构化约束，并由独立执行器按需启动 Encounter Residual 一步展开。

```text
玩家可见状态 + 合法动作
        ↓
模型样本投影与张量化
        ↓
精确引擎候选特征 + 共享实体 Transformer P2
        ↓
Policy 排序 + 状态风险/资源诊断
        ↓
Combat Directive 调整与合法性门禁
        ↓
Combat Tool 返回动作、Top-k、置信度与重新规划信号
        ↓
低风险/高置信度 ──→ 直接采用 P2 Policy
高风险/高分歧   ──→ Residual 辅助的一步展开
        ↓
真实 sts2-cli 引擎执行
```

当前代码中的安全基线仍是 **P2 Policy Only**；面向初版上层 Agent 的推荐运行形态则是
**P2 安全底座 + Residual 诊断 + 条件一步展开 + 自动回退**。触发器尚未完成，因此不能把该
组合写成已部署默认能力。turn-boundary search 和 MCTS 继续作为更高预算的实验能力。

## 2. 已完成的成果

### 2.1 数据与评测基础

- 当前 Combat Dataset 快照包含58个完整 episode、941场战斗和18,616条战斗 transition；
- train/validation/test 按752/84/105场战斗划分，Act 1–3 均有覆盖；
- test 额外冻结5个完整 run，可用于固定局外计划下的连续爬塔测试；
- observation 只保留玩家可见信息，合法候选由真实环境提供，隐藏 RNG 不进入网络输入；
- 原始人工记录保持不可变，训练样本、词表、Value target 和 split 均为可重建派生产物。

### 2.2 P2 战斗网络

P2 使用人工动作进行行为预训练，主干是4层、`d_model=128` 的共享实体 Transformer。玩家、
卡牌、敌人、遗物、药水和状态被编码为结构化实体；每个合法动作结合来源实体、目标实体和
动作类型单独评分，因此网络不会依赖固定手牌长度或固定敌人数。

在共享主干之外，P2 接入18维公开引擎候选特征。能够由引擎或确定规则直接得到的即时伤害、
格挡、击杀、药水和最大生命变化等信息，优先作为精确输入或规则门禁，而不是完全交给网络
近似。当前默认 checkpoint 为：

```text
artifacts/combat_policy_p2_cuda/20260818T125348Z/model.pt
```

离线验证集的人类动作 Top-1 为58.86%，测试集 Top-1 为57.20%，合法动作率为100%。这些指标
表示动作模仿能力，不等同于真实游戏强度。

### 2.3 真实引擎在线评测

P2 已在 validation 的84场战斗上完成真实引擎评测，其中83场可共同比较，Test Subject 因
连续特殊选择路径的 CLI 超时被标记为 `engine_unsupported`。

| 方法 | 死亡 | 总战损 | 相较人类总战损 |
|---|---:|---:|---:|
| 人类记录 | 2 | 749 | — |
| P2 Policy | 16 | 2,049 | +173.57% |
| P2 一步展开 | 12 | 1,816 | +142.46% |
| Encounter residual 一步展开 | 12 | 1,784 | +138.18% |

一步展开相对 P2 Policy 减少233点战损并净减少4次死亡，但也使16/83场恶化，在 Act 1 和
A9–A10 分组没有改善。因此当前结论不是“搜索无效”，而是现有 Value、预算和接管条件还不
足以保证搜索稳定优于 Policy。

从综合收益看，真正显著的改善来自一步展开。Encounter residual 一步展开比 P2 一步展开再
减少32点总战损，但两者死亡数相同，且79/83场结果完全一致，平均每场只相差约0.39 HP。
Residual 的额外推理成本相对真实引擎展开很低，因此值得保留为搜索先验、模型分歧信号或
轻量修正；但当前证据不足以说明 Residual Policy 本身已经稳定优于 P2。

### 2.4 引擎、搜索与吞吐边界

- 已通过 `sts2-cli` 完成战斗状态重建、动作执行、前缀回放和终局 rollout；
- 已实现一步展开、turn-boundary search、determinized PUCT/MCTS 原型及逐动作重规划；
- 多 worker 的纯恢复吞吐可达约3.5倍，但 CUDA 端到端搜索在 Act 1/2/3 仅约
  2.29/1.89/1.57倍，没有稳定通过继续投入的2倍门槛；
- 项目暂时停止继续开发引擎并行和树合并，也没有实现类似 Spire Pilot 的专用 C++ 高速
  模拟器；高预算全程搜索仍然昂贵。

### 2.5 失败分析与反事实闭环

验证失败轨迹表明，主要问题不是入口处存在大量未见卡牌或敌人，而是 Policy 在线执行后的小
误差不断累积，逐渐进入人工数据分布尾部；搜索在部分状态下又进一步放大这种偏移。

训练 split 上已经建立终局反事实采集闭环：让冻结 P2 自己运行，在早期异常根状态比较 Top-k
动作，并在共享 determinization 下使用同一个冻结 P2 continuation 运行到战斗结束。60场最早
根实验得到45个有效根、22个 P2 错误首选和109组成对标签。其监督含义是 `Q^P2(s,a)`，不是
最优 `Q*(s,a)`。

三种低成本调优已经完成门禁，但均未替换 P2：

- 全局候选末层提高反事实排序，却使人工 validation Top-1 下降4.01个百分点；
- 严格规则门控仍下降0.90个百分点，高于允许的0.5个百分点；
- Encounter Embedding 局部适配没有纠正错误首选，并下降1.50个百分点。

因此当前没有把被拒绝的 ranker、gate 或 encounter adapter 接入默认路径。

### 2.6 上下层连接能力

Combat Tool 0.2.0 已能返回合法动作、Top-k、Policy 分数、精确引擎预览、状态风险、资源诊断、
不确定性、模型与数据指纹，以及是否建议重新规划。Combat Directive 0.2.0 已能承载资源偏好、
药水约束、目标/动作偏好、风险阈值和有界搜索请求。

这里有一个明确边界：Combat Tool 当前可以表达和返回搜索请求，但其同步 Python 接口本身
不会执行搜索；一步展开或 MCTS 仍由独立的真实引擎 executor 负责。未经验证的自然语言怪物
知识也不能直接变成硬规则，必须先经过上层解释，再转换成有类型的偏好，或由确定规则适配器
生成可信的候选事实。

## 3. 当前默认能力与实验能力

| 状态 | 能力 |
|---|---|
| 当前安全基线 | P2 Policy Only、合法动作约束、公开引擎候选特征、状态风险与不确定性诊断 |
| 推荐集成形态 | P2 安全底座 + Residual 诊断 + 条件一步展开 + 失败或优势不足时回退 P2 |
| 可供上层调用 | Combat Tool、Combat Directive、机制事实、搜索请求与重新规划信号 |
| 待完成门禁 | 条件一步展开的可靠触发器，以及上层搜索预算到真实 executor 的连接 |
| 实验性 | 无条件一步展开、turn-boundary search、MCTS、药水反事实评估、机制型指令控制器 |
| 已拒绝替换 P2 | 全局反事实末层、规则 gate、Encounter Embedding/Residual 的直接接管 |
| 尚未实现 | 初版上层 LLM Agent、稳定的学习式搜索接管器、专用 C++ 高速模拟器 |

## 4. 后续方向

### 4.1 先冻结可调用的 P2 下层

初版上层 LLM Agent 不需要等待战斗网络达到 Spire Pilot 的强度。应先把 P2 Policy、Residual
诊断、Directive、一步展开和真实引擎执行连接成稳定的 Combat Tool。普通、低风险且 Policy
置信度高的状态直接行动；精英、Boss、特殊机制、高风险或 P2/Residual 明显分歧的状态允许
上层提高一步展开预算；展开失败、超时或价值优势不足时回退 P2。对 checkpoint、数据集和
实际启用的能力继续做指纹记录。

### 4.2 接入初版上层 LLM Agent

上层主要负责跨战斗资源预算、药水策略、特殊敌人机制、目标优先级和搜索预算，不直接生成
卡牌动作。自然语言知识需要先转换为下层能理解的结构化目标或可信机制事实。Boss、精英和
特殊机制战斗可以获得更高的上层介入程度和搜索预算；常规战斗继续由 P2 低成本执行。

### 4.3 继续修复下层的在线分布偏移

后续调优采用“不可变 P2 + 独立条件 residual + 学习式接管”，而不是直接微调整个 Policy。
反事实数据应同时包含三类根：P2 明显错误、P2 已经正确、多个动作实际近似平局。接管器需要
学习何时不要修改 P2，并在独立测试集同时通过错误纠正率、常规动作保持率和真实引擎战损门禁。

### 4.4 改进反事实数据质量

- 从训练 split 的代表性战斗采样，不只使用少量失败集中的敌人；
- 对生成卡牌、随机目标等高随机动作增加共享 determinization 数量；
- 将终局收益拆分为死亡、战损、药水、战斗轮数和明确的局外成长；
- 继续保留 `Q^P2` 与 `Q*` 的语义边界，不把有限 continuation 结果称为最优教师。

### 4.5 维持真实环境双层评测

每个候选版本先在独立单场战斗上评估，再在冻结完整 run 上进行连续测试。单场评测用于定位
战术问题；连续测试用于观察前期战损如何影响抓牌、休息、药水和后续战斗。固定人类局外路线
只能作为受控基线，不能替代未来上层 Agent 的完整爬塔能力评估。

## 5. 对应接口文件

本节只列模块边界和入口文件，不固定具体 JSON 字段或通信协议。后续协议演进应尽量保持这些
职责边界稳定。

### 5.1 下层核心接口

| 职责 | 接口文件 | 当前作用 |
|---|---|---|
| 对外 Combat Tool | `src/sts2_dataset/combat_tool.py` | 加载带指纹的 checkpoint，完成合法候选排序并返回诊断与重新规划信号 |
| 上层战斗指令 | `src/sts2_dataset/combat_directive.py` | 定义资源、动作偏好、风险门槛、机制事实和有界搜索请求 |
| 真实引擎运行适配 | `src/sts2_dataset/combat_online.py` | 将 CLI 状态转为模型样本，并把模型候选转为真实引擎命令 |
| 模型输入契约 | `src/sts2_dataset/combat_contract.py` | 从人工 transition 投影可见 observation、合法候选和监督标签 |
| 张量化边界 | `src/sts2_dataset/combat_tensorizer.py` | 将动态实体和候选集合转换为模型张量 |
| P2 网络与目标 | `src/sts2_dataset/combat_model.py` | 共享实体 Transformer、候选 scorer、状态/资源预测头 |
| 精确候选特征 | `src/sts2_dataset/combat_engine_features.py` | 计算可由公开状态和引擎规则直接得到的候选特征与规则门禁 |
| 稳定遭遇身份 | `src/sts2_dataset/combat_encounter.py` | 在整场战斗中维护初始敌人组合签名 |

### 5.2 搜索与特殊能力接口

| 职责 | 接口文件 | 当前作用 |
|---|---|---|
| 一步展开与接管门禁 | `src/sts2_dataset/combat_lookahead.py` | Top-k 展开、终局死亡 veto、Policy advantage gate |
| MCTS/搜索统计 | `src/sts2_dataset/combat_search.py` | 自适应预算、PUCT、CVaR、根动作覆盖和教师记录 |
| 机制指导适配 | `src/sts2_dataset/combat_mechanics.py` | 将少数已实现的敌人机制转换为动态 Directive/候选事实 |
| 药水目录 | `src/sts2_dataset/combat_potions.py` | 维护公共与角色药水的结构化能力分类 |
| 药水反事实评价 | `src/sts2_dataset/combat_potion_evaluator.py` | 汇总配对 rollout，形成药水使用建议和置信证据 |

### 5.3 可执行入口与评测接口

| 职责 | 接口文件 | 当前作用 |
|---|---|---|
| Combat Tool CLI | `src/sts2_dataset/combat_tool_cli.py` | 提供独立进程可调用的工具入口 |
| 单场在线运行 | `tools/run_combat_policy_online.py` | 在真实引擎中执行一场 Policy/实验搜索战斗 |
| 连续固定计划运行 | `tools/run_fixed_plan_policy.py` | 固定局外决策并连续执行战斗，用于整局受控评测 |
| 验证集全量消融 | `tools/run_validation_combat_ablation.py` | 重建 validation 战斗并比较 Policy、适配器和一步展开 |
| 失败回归集 | `tools/build_combat_failure_ratchet.py` | 从真实引擎结果冻结高战损、死亡和错误接管场景 |
| On-policy 终局采集 | `tools/collect_combat_on_policy_counterfactuals.py` | 在 train split 采集 P2 实际访问状态的终局反事实 |
| 反事实数据构建 | `tools/build_combat_counterfactual_dataset.py` | 将原始 rollout 转为可追溯的根状态和成对标签 |
| 实验 ranker 训练 | `tools/train_combat_counterfactual_ranker.py` | 训练独立候选排序或 encounter 条件实验模型 |
| 接管门禁评估 | `tools/evaluate_combat_counterfactual_gate.py` | 比较候选修正收益与常规人工动作能力损失 |

## 6. 推荐的后续开发入口

若开始接入初版上层 LLM Agent，优先阅读和保持稳定的文件顺序为：

1. `src/sts2_dataset/combat_tool.py`：上层最终调用的能力边界；
2. `src/sts2_dataset/combat_directive.py`：上层能够表达的战斗目标；
3. `src/sts2_dataset/combat_online.py`：下层如何连接真实游戏；
4. `src/sts2_dataset/combat_model.py`：P2 能够提供和不能提供的预测；
5. `src/sts2_dataset/combat_lookahead.py` 与 `combat_search.py`：需要额外预算时的实验执行路径。

现阶段最合适的系统表述是：

> **以人类行为预训练的 P2 作为安全战术底座，在高风险或高分歧状态条件触发 Residual 辅助的
> 一步展开，由上层 LLM Agent 提供跨战斗资源目标、特殊机制指导和搜索预算；真实引擎负责
> 合法性、精确规则、展开执行和最终结果验证。**
