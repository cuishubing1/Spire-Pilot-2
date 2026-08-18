# sts2-cli Compact Prefix Replay V0

## 目标

搜索在每个模拟开始时都要从战斗入口存档重建当前根状态。旧路径在重放前缀的每一步后都会生成完整 `CombatPlayState`，包括本地化文本、卡牌动态数值以及每张攻击牌对每个存活敌人的伤害预览；这些中间结果随即被下一步覆盖。

V0 增加 `restore_combat.prefix_projection=compact`：中间动作仍完整推进动作队列、异步 continuation、选牌器、回合与终局逻辑，但战斗仍在进行时只返回轻量决策标记；前缀和抽牌顺序设置完毕后，再调用一次完整状态投影。

协议同时增加可选 `suffix`。它用于 turn-boundary/beam search 中已经确定的候选动作序列，执行顺序为“入口前缀 → determinization 抽牌顺序 → 候选后缀 → 最终投影”。这不会用于需要观察每个中间状态后再选动作的 MCTS。

旧路径仍保留。搜索侧先以 `cached_batch_auto` 自动选择投影：前缀与候选后缀合计少于 4 个动作时使用 legacy 投影，达到 4 个动作后使用 compact 投影，避免空前缀额外做一次轻量投影再做完整投影。

CLI 0.3.2 使用隔离的 prepared-save 缓存。`cache_save` 首次接收入口存档时完成 JSON 解析与 schema 检查，并通过游戏自身的 packet serializer 保存二进制入口描述；后续 `restore_combat.reuse_prepared_save=true` 每次从 packet 反序列化出新的 `SerializableRun` 对象图，再构造独立 `RunState`。不能直接复用最初解析出的对象：游戏加载过程会修改其中的楼层、RNG或容器状态。当前默认模式为 `cached_batch_auto_prepared`，原有 JSON 解析路径继续保留为正确性对照和回退路径。

## 正确性边界

- 不跳过 `WaitForActionExecutor`、同步上下文 Pump、选牌器或终局处理。
- 只省略中间 `combat_play` 的可见状态序列化；最终状态、合法动作与精确引擎结果不变。
- `legacy`、`cached_batch` 和 `cached_batch_compact` 继续可显式选择，便于回退和 A/B。
- prepared save 只复用不可变 packet 字节，不复用 `SerializableRun`、战斗、随机数或动作队列实例。
- 隐藏 RNG 只用于真实引擎复现与搜索 determinization，不进入策略网络观察。

## 阶段性结果

在固定 Act 1–3 入口、相同动作前缀下，legacy 与 compact 的最终 JSON 状态逐字段相等。前缀长度为 8 时：

| 场景 | legacy restore p50 | compact restore p50 | 整次 restore 加速 | prefix replay 加速 |
|---|---:|---:|---:|---:|
| Act 1 | 7.464 ms | 4.762 ms | 1.57x | 2.16x |
| Act 2 | 17.346 ms | 8.066 ms | 2.15x | 3.59x |
| Act 3 | 18.150 ms | 11.357 ms | 1.60x | 2.48x |

一场固定 Act 3 战斗的 12 步 CUDA turn-boundary 搜索对照中，三种策略的动作序列和战损完全一致；搜索耗时由 4989.524 ms 降至 4297.757 ms（1.16x），完整评测墙钟时间由 10833.970 ms 降至 9989.547 ms（1.08x）。

接入 batched suffix 后，对固定 Act 1/2/3 各一场、每场 8 步的联合对照中，Policy、一步展开和 turn-boundary 的动作序列及战损仍逐项一致。turn-boundary 搜索累计耗时由 8240.640 ms 降至 7022.081 ms（1.17x），分 Act 加速分别为 1.14x、1.21x、1.17x；完整三场评测墙钟时间由 23065.122 ms 降至 21206.391 ms（1.09x）。单场 Act 3 的 12 步对照达到 1.24x 搜索加速。

prepared-save 在 Act 1–3、128/256 simulations、每组两次交错顺序的 CUDA 配对测试中，所有根动作、树节点数和引擎动作数均保持一致。相对 `cached_batch_auto`：

| 场景 | 128 search 加速 | 256 search 加速 | restore 阶段平均加速 |
|---|---:|---:|---:|
| Act 1 | 1.21x | 1.06x | 1.49x |
| Act 2 | 1.12x | 1.03x | 1.26x |
| Act 3 | 1.10x | 1.02x | 1.21x |
| 全部配对均值 | 1.14x | 1.04x | 1.32x |

prepared-save 的收益在浅预算和较短战斗中更明显；256 simulations 时，完整搜索加速已经收敛到 1.02–1.06x。优化后 256 预算中，入口恢复约占搜索墙钟的 12.5%（Act 1）、13.9%（Act 2）和 17.0%（Act 3），模型推理约占 17.5%、27.8% 和 35.4%。剩余时间主要在真实引擎动作推进与 Python 搜索调度，而不是存档 JSON 解析。

因此，紧凑重放与 prepared-save 值得作为低风险默认优化保留，但继续重写恢复路径的边际收益已经有限。若要再显著提高可投入的搜索预算，下一阶段应优先测量并优化“引擎动作推进”和“模型推理/搜索调度”；直接实现完整 C++ 模拟器仍不符合当前投入产出比。任何进一步优化都必须保持最终状态逐字段等价、同种子树结构一致和 prepared-save 交替复用无污染三项门禁。
