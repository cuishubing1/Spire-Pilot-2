# Combat Directive / Combat Tool V0

本接口连接上层长期规划 Agent 与下层战斗策略网络。V0 的目标不是让上层用自然语言直接控制出牌，
而是把宏观意图转换成可验证、可记录、可回放的结构化约束。

## 分层边界

上层负责给出：

- 本场战斗对生命、药水和永久成长的价值权重；
- 药水预算与需要保留的药水；
- 禁止、鼓励或抑制的动作类型；
- 需要优先处理的稳定目标引用；
- 风险、分歧或高不确定状态下的重新规划阈值。

下层负责：

- 读取玩家可见的结构化战斗状态；
- 在 sts2-cli / HumanRecorder 枚举的合法动作中逐个评分；
- 合并策略先验、资源价值、精确公开规则特征和上层指令；
- 返回最终动作、Top-k、分数拆解、风险与是否需要重新规划。

诸如“该敌人必须先破甲”“此动作会触发反伤”一类机制，不作为自由文本直接输入网络。
它们必须先由版本锁定的规则适配器转换成 `CandidateMechanicFactV0`，绑定具体 `candidate_id`，
并标明 `source=engine_rule`。事实可以调整候选分数或将其硬排除；LLM 生成的未经验证结论不能进入该通道。

## CombatDirective V0

Schema：`schemas/combat_directive_v0.schema.json`。

```json
{
  "schema_version": "combat-directive-0.2.0",
  "scope": "current_combat",
  "expires_at": "combat_end",
  "objective": {
    "hp_loss_weight": 1.0,
    "potion_cost": 0.2,
    "max_hp_gain_weight": 1.0
  },
  "resource_policy": {
    "max_potion_uses": 1,
    "potion_uses_so_far": 0,
    "preserve_potion_ids": [],
    "acceptable_hp_loss_fraction": 0.25
  },
  "action_preferences": {
    "forbidden_action_types": [],
    "forbidden_candidate_ids": [],
    "action_type_biases": {"use_potion": 0.5},
    "target_biases": {},
    "candidate_biases": {}
  },
  "replan_policy": {
    "death_probability": 0.2,
    "normalized_entropy": 0.8,
    "top_probability_gap": 0.08
  },
  "search_policy": {
    "mode": "policy_only",
    "budget_class": "low",
    "max_wall_ms": 750,
    "determinizations": 2,
    "allow_policy_override": true,
    "mechanic_plan_id": null
  }
}
```

偏置是决策 logit 的小幅先验，不是绝对命令。真正的硬限制只来自合法动作集合、显式禁止字段和
可信规则事实。若上层排除了全部候选，工具会退回完整合法集合并返回
`directive_conflict_fallback`，要求上层重新规划。

## Combat Tool V0

Schema：`schemas/combat_tool_v0.schema.json`。主要返回字段包括：

- `chosen_action`：最终结构化合法动作；
- `top_k` / `ranked_actions`：候选排序与概率；
- `score_breakdown`：Policy、资源效用、动作/目标/候选偏置、规则偏置；
- `engine_preview`：当前公开状态下可由规则精确计算的即时特征；
- `predicted_risk`：学习到的战斗结果风险，以及已被公开意图规则替换的即时承伤；
- `capabilities`：逐项说明当前调用支持、仅诊断、实验性或不可用的能力；
- `directive_effects`：显式报告哪些目标权重实际参与重排、哪些被忽略及原因；
- `model_provenance`：checkpoint、词表、数据索引和源数据 manifest 指纹；
- `request_replan`：指令冲突、高熵、候选接近、死亡或掉血预算超限等原因。

当前 V0 是可调用的同步 Python 接口，并未定义 LLM Prompt。未来无论上层采用 LLM、规则规划器
还是人工控制，都必须生成同一个 Directive Schema，保证下层网络和评测协议不随上层实现变化。

同步 JSON 入口为：

```powershell
sts2-combat-tool --checkpoint <model.pt> --device cuda --request <request.json>
```

请求遵循 `schemas/combat_tool_request_v0.schema.json`，其中环境适配器提供模型 `sample`，上层只
负责可选的 `directive`。LLM 不应自行构造或修改玩家不可见状态和合法动作集合。

### 类型化搜索请求

Directive 0.2.0 新增 `search_policy`。上层只能选择 `policy_only`、`one_step` 或
`turn_boundary`，以及 `low/medium/high` 预算等级、1–8个确定化和50–10000 ms时限；不能直接
指定任意 simulation 数。Combat Tool 会在 `search_request` 中回显请求、推荐模式和触发原因，
并明确返回 `search_executed=false`。当前纯网络接口不会伪装成已经执行搜索，后续由真实引擎
执行器消费该请求并补充实际耗时、预算和回退原因。

## 当前限制

- 目标偏置依赖当前战斗内的稳定 `target_ref`，战斗结束后自动失效；
- 敌人机制规则适配器只有通用候选事实接口，尚未建立逐敌人的规则库；
- 风险与远期掉血仍来自现有人工数据预训练的价值头，不等于精确终局搜索；
- 状态 Value 用于当前风险与重新规划；候选资源头只在人工实际动作上接受过监督，默认仅作
  `diagnostic_on_policy`，其目标重排必须显式设置正的 `decision_value_scale`，并标为实验性；
- V0 不把 MCTS 输出循环作为网络输入；搜索仍是独立、可选的重新规划工具。

P1.3 checkpoint 为
`artifacts/combat_policy_p1_v13_cuda/20260817T151201Z/model.pt`。它保留 V2 候选资源头但冻结
参数，且默认价值缩放为0；测试集策略 Top-1 为55.50%，与未保留资源头的 P1.2 相同。真实
Act 1–3 根状态检查见 `artifacts/combat_tool_v02_p1_v13_real_roots.json`。

## 首轮真实引擎敏感性检查

`tools/run_combat_tool_sensitivity.py` 使用 sts2-cli 构造三个固定 A0 战斗根状态；同一场景的
牌组、抽牌、敌人和合法动作完全相同，只替换 Directive。2026-08-16 使用当前 V2 checkpoint
得到：

| 场景 | 默认动作 | Directive 现象 |
|---|---|---|
| Act 1 Fuzzy Wurm Crawler | Strike | 初始牌组且无药水，三种资源策略均不改动作 |
| Act 2 Bowlbugs | Rage | 禁用药水后3个药水动作全部不可选；降低药水阈值使药水概率质量由0.0085升至0.0285 |
| Act 3 Scrolls of Biting | Armaments | 降低药水阈值使药水概率质量由0.0391升至0.1226；集火最低血敌人后改选对该目标使用 Dominate |

三场中规则门禁后的 `max_hp_delta` 均为0，原始网络噪声不再参与选择。该检查只证明接口能在
真实合法动作上按预期改变约束和偏好；它不证明这些宏观指令能提高整场战斗结果。后续强度实验
应选择机制型敌人、低血/药水槽已满等状态，分别执行完整战斗并与默认 Directive 做同根对照。

同一脚本增加 `--full-combat` 后，对每个配置从同一入口独立执行整场战斗，首轮结果为：

| 场景 | 默认 | 保留药水 | 降低药水阈值 | 集火最低血敌人 |
|---|---:|---:|---:|---:|
| Act 1 Fuzzy Wurm Crawler | 0 HP / 3回合 | 0 HP / 3回合 | 0 HP / 3回合 | 不适用 |
| Act 2 Bowlbugs | 28 HP / 4回合 | 28 HP / 4回合 | 28 HP / 4回合 | 31 HP / 4回合 |
| Act 3 Scrolls of Biting | 57 HP / 4回合 | 57 HP / 4回合 | 57 HP / 4回合 | 50 HP / 5回合 |

所有配置均获胜，且均未实际使用药水。集火在 Act 2 变差、在 Act 3 改善，说明 Directive 应被视为
待下层策略检验的偏好，而不是天然正确的动作标签。`request_replan` 在这些完整战斗中被多次触发；
当前脚本只记录信号，下一阶段可将高风险/高分歧状态路由到现有 MCTS，再比较“直接执行”和
“触发式搜索”的同根结果。

敌人机制指导的首轮受控实验见
[`combat-mechanic-guidance-v0.zh-CN.md`](combat-mechanic-guidance-v0.zh-CN.md)。

P1 已将18维公开引擎候选特征并入 Policy 的残差适配器，并在 Tool 外增加 Top-k 一步真实
引擎展开。它使用可见抽牌堆 determinization 和独立状态 Value，不再依赖未选择动作的资源头
作为反事实排序依据。设计与首轮门禁见
[`combat-policy-p1-one-step.zh-CN.md`](combat-policy-p1-one-step.zh-CN.md)。
