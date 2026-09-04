# PM-v2.1 对 PM-v1 问题的系统解决审计

更新时间：2026-07-14
状态：代码硬化与 no-API 验证阶段；尚未产生 PM-v2.1 实验结果

## 1. 结论先行

PM-v2.1 已把 PM-v1 的主要可修复结构性问题改成了 fail-closed
合同，但当前不能写“V2 已经学会了最优路由”。两件事必须区分：

1. **设计问题是否被代码和门禁修复**；
2. **新 PM 是否在未见用户上实际学会条件化的 quality–risk–cost
   routing，并超过强同预算固定动作**。

第一项可以通过静态审计、单元测试、artifact lineage 和 no-API
preflight 证明。第二项只能由完整 development labels、用户隔离 internal
test、人工 judge calibration、EvoEmo disjoint holdout 和 forced-swap 结果证明。
在这些证据产生前，状态只能是“具备可检验性”，不能是“经验问题已解决”。

## 2. PM-v1 → PM-v2.1 问题矩阵

| PM-v1 问题 | PM-v2.1 处理 | 成功判据 | 当前状态 |
|---|---|---|---|
| 外部 M0=0、RS≈96.4% | 九类 resource-need cases；M0/R0/RS 都有直接标签；learned-only M0、R0、M0+R0 门禁 | internal/external learned decisions 均通过 M0、R0、entropy、max-share 门 | 代码已实现；经验待测 |
| `MSE+RS` 近常量塌缩 | 最大动作占比、动作熵、distinct actions；severe-OOD/no-feasible fallback 从 learned decisions 中剔除 | 任何 fallback 不得伪装成“学会 abstain”；最大占比≤0.50 | 代码已实现；经验待测 |
| cost 只作 `epsilon=0` tie-break | 统一 utility：quality LCB − risk penalty − normalized resource cost；所有 tuning candidates 用同一冻结评估标尺 | 调参与在线选择使用同一 estimated-cost 口径；候选不能靠缩小自身 penalty 作弊；observed tokens 独立校准 | 代码已实现；经验待测 |
| omission/strategy risk 阈值形同关闭 | 七个 risk heads、逐维 applicability、逐维 UCB 阈值 | 任一适用风险越界动作不可行；无可行动作单独记录 | 代码已实现 |
| Overall 与 Emotional Support 完全相同 | judge schema 删除 Overall；六维独立评分；冻结 composite 只作训练 utility | 不存在 LLM Overall 字段；weights SHA、维度常量/复制/高相关门禁通过 | 代码已实现；新标签待测 |
| 单一 silver judge | 两个互相独立的 development judge families；逐维 median/MAD；full 前先做 4-call schema smoke、720-call train-only pilot 与 18-item 双人盲审 | 两个 raw family 必须分别逐维通过 human gate，median 不能掩盖坏 family | 代码已实现；endpoint/human 证据待测 |
| 训练 judge 与最终 judge 不独立 | development=Gemini/DeepSeek；external=OpenAI/Anthropic | freeze 检查四个 family 集合和 endpoint descriptor | 配置已冻结；API compatibility 待测 |
| 没有人工校准 | full 前 18-item train-only 双人 spot-check；freeze 前另做 72-item、按 regime 平衡、双人盲标 | 六 response/七 risk 原始维度逐一检验；禁止 Overall；两个 raw family 分别过门 | 代码已实现；需要真人标注 |
| 36 个 unique texts / template overlap | 52 users×九个不同状态；user、semantic family、normalized current text 三重 split 隔离；跨 split word/char fixed-hash 近重复门 | split manifest 无交叉且 near-duplicate audit PASS | 代码已实现；generated texts 待测 |
| train/test seed 污染 | ESConv split manifest 按 index 严格 join；只取 train 且排除 EvoEmo overlap | test/validation/overlap seed 行数均为 0，所有源和 manifest 有 SHA | 代码已实现 |
| current/future memory | `created_session < session_index` 强验证 | 当前或未来 memory 一律拒绝 | 代码已实现并测试 |
| synthetic stale/conflict oracle 进入特征 | stale/conflict/needed-source/regime 只存在 evaluator contexts | PM state schema 拒绝 evaluator-only 字段 | 代码已实现并测试 |
| evaluator context 与模型 state 混放 | memory backend、PM state、evaluator contexts 三个物理文件 | exact state/card/context ID 和 map SHA 对齐 | 代码已实现 |
| synthetic 标签格式合法但语义不成立 | 付费 sweep 前做 3 split×9 regime 平衡 semantic-sanity 盲审；至少双人判断 family/regime/needed sources | 逐字段 affirmative/agreement、逐 regime、逐 split YAML gate 全部 PASS，且 attestation lineage 相同 | 代码与测试已实现；正式 states 待人工标注 |
| fitted TF-IDF OOV | fixed word/char hashing；可选预计算 embedding | feature config hash 冻结；external OOV 不会变成全零 | 代码已实现 |
| 线性 action prior | nonlinear bootstrap outcome heads + state×action interactions | full action matrix 上预测与 interval coverage 通过 | 代码已实现；经验待测 |
| bootstrap 只给未校准 epistemic std | 12-user calibration 对每个 user 的 states×actions 取最大残差，拟合 head-wise conformal radius；`z` 预注册 | 16-user internal holdout 的 head-wise user-block Wilson evidence lower bound ≥0.60，即至少 14/16 blocks；target coverage 仍为 0.90 | 代码已实现；经验待测 |
| evaluator 定义的需求在合法 PM 特征中可能不可见 | exact deployable feature builder 上做 train-only user-group CV，覆盖 regime/source/memory need 及 strategy/memory helpful-vs-harmful 方向 | 任一信号门失败则在付费 action pilot 前停止；plan、schema smoke、script06/full gate 均重算 | 代码和防篡改测试已实现；正式 states 待测 |
| source relevance 不可见 | 只暴露 deployable query-to-source-catalog centroid，不暴露 item/top-score | development/external constructor golden parity | 代码已实现并测试 |
| strategy catalog 20→12,429 导致 OOD | development 绑定真实 strategy bank hash/count/token；count 常量不进入 OOD 特征 | strategy metadata 在 development/external 完全一致 | 代码已实现并测试中 |
| memory catalog 尺度 domain shift | retrieval-capacity-aware count/token/age 特征；external no-API OOD preflight | severe-OOD fallback≤0.10，不能靠常量 fallback 通过 diversity | 代码与回归已实现；外部数据待测 |
| fixed top-k 强塞零相关证据 | memory/strategy retrieval 使用 exclusive minimum relevance threshold；0 拒绝零重合 | 零相关 catalog 可返回空 | 代码已实现并测试 |
| PM 只选 source、不能保证 item precision | retriever 可在已选 source 内 abstain；风险标签评价实际 selected context | item-level 最优性仍不宣称 | 部分缓解，方法边界保留 |
| 内部只看点估计 | user-cluster paired bootstrap CI vs cost-matched fixed；deployment tradeoff 与 learned advantage 分开 | external 前 quality/support/utility CI lower 均须严格 `>0`，risk CI upper 同时过门；全 tie 失败 | 代码已实现；经验待测 |
| cost-matched baseline 可能并不同成本 | calibration 选 fixed 后，在 calibration/internal 都检查 observed tokens；external 再做 dry-run 与生成后双重容差 | 相对 PM 成本偏差≤10%，否则禁止称 cost-matched | 代码已实现；实际外部 tokens 待测 |
| pairwise 顺序偏差 | 主评测 single-candidate pointwise；关键比较 dual-order forced-swap | order disagreement≤0.25；不一致 preference 解析为 tie | 代码已实现；结果待跑 |
| 未 pilot 就 full-run 烧钱 | 1-call source-grounded generator pilot + 双人九例 semantic gate、4-call development schema smoke、180/720 development pilot、4-call external pointwise smoke；每阶段 exact plan/价格/预算/accepted SHA | generator structural/semantic 两道 gate 均 PASS 才允许 52 calls；所有 clients 在首个 ledger `STARTED` 前验证；失败即停止 | 代码与崩溃/篡改回归已实现 |
| API crash/retry 可能重复计费 | append-only ledger 在 HTTP 前 fsync `STARTED`，成功结果持久化并可重建输出 | unknown attempt 视为已花费；不能重发；provider usage 必须为正且不超冻结 bound | 代码与恢复测试已实现 |
| 外部为未评分 turns 浪费 generation | freeze 内容寻址 turns `[3,8]` 的 204-unit exact universe；10-turn 只做 no-API diagnostic | 每 condition 只允许 204 个生成 call IDs，summary/attestation 精确证明矩阵 | 代码已实现；freeze 后待跑 |
| 外部 optional stopping 污染结论 | 40-unit forced-swap 作技术/futility pilot，并从 full confirmatory units 排除 | full 报告只使用 disjoint units，保存 exclusion SHA | 代码已实现；结果待跑 |
| external 不同 policy 不同 seeker 世界线 | 继续使用已 attested fixed seeker tracks | exact scenario×seed×turn 矩阵 | 代码已实现 |

## 3. “PM 学会使用不同 policy”的硬定义

不能用“动作空间包含 16 个动作”或“输出里出现过多个动作”作为证据。
PM-v2.1 只有同时满足以下条件，才能说学到了条件化 routing：

1. 统计对象只包含 `fallback_type=none` 的 learned decisions；
2. M0、R0、M0+R0 都达到预注册最低比例；
3. 动作熵、distinct actions 和最大单动作占比同时过门；
4. `context_only`、`memory_harmful`、`strategy_helpful`、
   `strategy_harmful` 以及 MP/MS/ME 精确来源 regime 分别过门；
5. 付费前已证明这些 evaluator targets 在 exact deployable feature space 中至少
   具有 train-only user-group CV 可观测性；
6. 质量/风险/成本结果不是由某一个 synthetic user 或 family 主导；
7. 相对 calibration-selected cost-matched fixed 的 quality、emotional support、
   utility paired user-cluster CI lower bounds 全部严格大于零；全 tie 不算优势；
8. 同一 fixed action 在 calibration 与 internal 的 observed-token 偏差均不超过
   10%，且风险/coverage 门同时通过；
9. EvoEmo no-API preflight 不发生 OOD 或 no-feasible 大规模 fallback；
10. disjoint external holdout 不出现新的近常量塌缩。

任一项失败，都应写“PM-v2.1 没有通过条件路由验证”，而不是调松阈值后重跑。

## 4. quality–risk–cost 的定义

训练标签保留六个原始 response dimensions 和七个 resource-use risk
dimensions。六维 composite 是预注册、确定性的归一化加权公式：

```text
0.30 emotional_support
+ 0.20 personalization
+ 0.15 memory_appropriateness
+ 0.15 factual_grounding
+ 0.10 temporal_consistency
+ 0.10 non_intrusiveness
```

它不是 Overall，也不能在论文中冒充人类 holistic quality。在线决策使用：

```text
quality lower bound
- risk_weight × applicable risk upper bound
- cost_weight × within-state normalized estimated resource cost
```

observed input tokens 不可在选择前获得，所以不直接进入在线 decision；它用于校准
estimated cost、选择同预算 fixed baseline、报告真实资源使用，并设置估算—观测相关
门禁。这一区分避免了把未来生成日志偷喂给 PM。

## 5. API 运行顺序与停止规则

所有步骤默认停止，不默认继续：

1. 完成全仓 compile/test、release preflight 和 seed lineage；
2. 对 synthetic bundle generation 做 exact no-API dry-run；
3. 只运行 1-call source-grounded synthetic-generator compatibility pilot；失败即停止；
4. 对该九例 pilot 做双人、八维、all-affirmative semantic review；
5. 两道 pilot gate 均 PASS 后才生成完整 52-user development bundles（24/12/16 user split）；
6. 做 3 split×9 regime、至少双 annotator 的无 API semantic-sanity 审计；
7. semantic-sanity PASS 后，重算 train-only deployable-feature observability，再
   创建按九个 regime 平衡的 compatibility matrix；
8. 先运行 4-call development judge schema smoke；
9. 再运行 180-call action pilot 与 720-call 双-family judging；schema、MAD、
   dimension independence、方向分离和 same-oracle headroom 必须全 PASS；
10. 对 18 个 train-only pilot items 做双人盲审，两个 raw judge families 分别逐维过门；
11. 才能运行 full 7,488 action sweep 与 29,952 development judging；
12. internal learned-advantage gate 与 72-item human audit 全过门后，才能创建 study freeze；
13. external 三个 PM-v2 conditions 各只生成冻结的 204 units，并验证预计/实际成本匹配；
14. 先运行 40-unit、双顺序、双 family forced-swap；
15. 若技术门失败，或 PM 相对同预算 fixed 的 Support CI upper 仍低于
    -0.10，停止 full external judging；
16. forced-swap PASS 后先跑 4-call external pointwise schema smoke；
17. 若继续，full external judging 必须排除 pilot units，避免 optional stopping
    污染 confirmatory 结果。

## 6. 当前仍不能由代码“解决”的问题

以下属于证据范围，不是再加一个单元测试就能解决：

- 真人是否长期感到被理解；
- 回复是否改善后续情绪轨迹、信任或参与度；
- 临床风险识别与医疗安全；
- 真实 privacy consent；
- 在线用户反馈、RL 或 POMDP 长期回报；
- PM 是否在 EvoEmo 之外的自然用户分布继续有效。

因此，即使 PM-v2.1 全部通过，也只能主张“在冻结的单步资源分配协议下，
条件化 routing 相对强 fixed baselines 的质量–resource-risk–cost 表现”，不能主张
真实长期心理效果或临床安全。

## 7. 当前证据状态

截至本文件更新时间：

- 历史 generator compatibility calls：五次失败、一次 v4 structural success；v5 本轮 calls=`0`；judge calls=`0`；
- strict ESConv seed extraction：`PASS`，878 个 train-only、非 EvoEmo-overlap seeds，
  output SHA256 `1f74d24f491963584268e095148c44689684064e2c82697e21ae4ab91606dba6`；
- v5 1-call generator compatibility dry-run：`PASS`，最大估计成本
  `0.00608925 USD`，cost SHA256
  `b52a540a0fe3ed985885f0a4cf3c93fe60d3c9472e545284183ac052154eb92f`；
- v5 52-user generation dry-run：`PASS`，52 expected/max calls，最大
  估计成本 `0.3163494 USD`，cost SHA256
  `dde4ec80d57a8cfb2ea3da115ea9f3b0046367499728d8ff64f935b9057963a7`；
- full repository tests：`212 collected，全部 PASS`；`git diff --check` 通过；
- release preflight：`API_PILOT_READY`，syntax/static-leakage/data/config/
  family-independence/pytest 全 PASS；`confirmatory_ready=false` 是因为现有旧 study
  freeze 已被当前代码变更作废，且 PM-v2.1 的 states/checkpoint/freeze 尚未生成，
  不能把该状态误读为 confirmatory ready；
- 当前环境 `OPENAI_API_KEY`、`NVIDIA_API_KEY`、`GEMINI_API_KEY`、
  `ANTHROPIC_API_KEY` 均缺失；dry-run ledger 均不存在，证明本轮没有开始物理调用；
- PM-v2.1 development states/labels/checkpoint：尚未生成；
- human audit：尚未进行；
- PM-v2.1 external generation/evaluation：尚未进行。

所以最准确的当前结论是：

> PM-v2.1 已把本次审计识别到的 PM-v1 结构性失败变成不可静默绕过的测试与冻结门禁；
> 它尚未通过经验数据证明这些问题已经在结果层面解决。
