# MetaCom PM-v1 失败、限制与根因完整复盘

> **ARCHIVED / 只读历史证据。** 本文只解释旧 PM-v1 的结果与根因，不再维护当前问题状态、
> 执行顺序或待办。其可迁移问题已合并到
> `PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`；当前路线以
> `PM_V1_5_CORE_CHAIN_PLAN_ZH.md` 为准。后续不得在本文追加新的活跃问题清单。

更新时间：2026-07-14  
适用对象：旧版监督式 Policy Manager（仓库 `main` 中的 V3.3 / PM-v1 链路）  
文档性质：研究复盘、负面结果解释、PM-v2 设计依据；不是为了否定全部旧工作，也不是新的实验结果。

---

## 1. 先给出总判断

PM-v1 不能简单概括成“项目失败”或“模型完全无效”。更准确的结论是：

1. **工程和因果边界基本成立**：PM 确实在检索前选择资源，最终版本未发现已经证实的数据泄漏、未来信息读取或直接把 judge 标签喂给 PM 的问题。
2. **核心正向发现成立**：全历史或高资源注入不等于更好的回复；PM 相对 Full History 大幅降低 token，并保持更高或接近的回复质量。
3. **核心学习型优势未被证明**：PM-v1 没有稳定超过同预算固定 `ME+R0`，也没有超过 Context Only 或 Raw Session Top-k 的 fixed-input 回复质量。
4. **名义上的质量—风险—成本权衡并未真正落实到最终决策**：最终 `epsilon=0`，成本只用于最高质量候选内部的 tie-break；omission 和 strategy risk 阈值为 `1.0`，实际基本没有约束力。
5. **PM 没有学会真正的 abstention**：EvoEmo 1020 个状态中 `M0` 选择率为 0%，RS 开启率约 96.4%，动作明显集中到 `MSE+RS`。
6. **数据和标签确实是主要原因之一**：开发数据表面规模较大，但只有 36 个唯一 current-user texts，user-fold 没有同时隔离 semantic family 和表达文本；训练标签完全由 LLM silver judging 产生，并且后来发现 `Overall` 与 Emotional Support 完全重合、部分风险维度退化。
7. **PM-v1 的能力边界被高估了**：它是 coarse source router，不是具体 memory item selector；固定 top-k lexical retrieval 没有 relevance threshold，使高资源动作很容易携带弱相关历史。
8. **外部评价本身也有失败和限制**：旧 bundle pairwise 协议因强顺序偏差正式失败；V4 虽然技术有效，但 `Overall` 构念失效、仅评价固定输入单步回复，不能证明长期用户状态改善。

因此，PM-v1 最可靠的研究结论应写成：

> 旧版实验支持“资源并非越多越好”以及“检索前资源调度具有研究价值”，但当前监督式 PM 尚未证明其条件路由优于强同预算固定动作，也未学会可靠地关闭记忆和 Strategy RAG。

---

## 2. “失败”必须分成五种，不应混为一谈

| 类型 | 具体情况 | 是否还能用于论文 |
|---|---|---|
| 正式协议失败 | EvoEmo 10-turn bundle pairwise 的 AB/BA consistency gate 未通过 | 只能作为诊断，不能作为 confirmatory result |
| 指标失效 | V4 `Overall` 与 Emotional Support 重合，不能代表整体质量 | 原 Overall 不应再作为主指标 |
| 真实负面结果 | PM 没有超过 Context Only、Session Retrieval、同预算 `ME+R0` | 必须诚实报告，可作为重要研究发现 |
| 结构性限制 | PM 不读取具体 snippets，无法负责 item-level evidence precision | 可作为方法边界和 future work |
| 研究范围限制 | 单步 fixed-input、LLM judge、无真人、无长期闭环 | 不能外推为真实长期支持或临床效果 |

这一区分非常重要。协议或指标失败意味着证据不可用；负面结果则是有效证据，只是结论不利于原假设。

---

## 3. 旧版本的完整失败时间线

### 3.1 早期内部链路问题：多数后来已修复

旧开发过程中曾发现：

- 非 M0 selected-set omission 标签 M2b 可以静默缺失；
- validation selection 缺少完整 provenance / attestation；
- 早期 EvoEmo interactive comparison 会因不同 policy 产生不同 seeker 世界线；
- Strategy Bank 曾处于 pilot/development 状态；
- external evaluation 和 study freeze 门禁不完整；
- 一些文档曾把 PM 写得像 RL/POMDP，超过实际方法范围。

这些问题在最终 V3.3 外部实验前已通过 fail-closed、fixed seeker tracks、attestation、study freeze 和文档边界逐步修复。相关记录：

- `snap_reports/METACOM_V33_GITHUB_REVIEW_FIX_REPORT_20260628_ZH.md`
- `project/tests/test_m2b_fail_closed.py`
- `project/tests/test_freeze_gate.py`
- `project/tests/test_scientific_safeguards.py`

因此，这些历史问题不能直接解释最终外部负面结果，但说明旧链路是在多轮补丁后才达到可评估状态，而不是从一开始就有完整、稳定的研究协议。

### 3.2 第一次正式 EvoEmo response evaluation：pairwise protocol 失败

旧协议把 10 个 fixed-context cases 打成一个长 bundle，再对 PM 与 baseline 做 AB/BA 双顺序比较。

正式结果：

| Comparison | Orientation consistency |
|---|---:|
| PM vs Best Fixed | 0.647 |
| PM vs Full History | 0.745 |
| PM vs Session Retrieval | 0.402 |
| PM vs Strong Rule | 0.569 |

预注册门槛为 `0.80`，全部失败。

同时该协议消耗：

- 816 次 judge calls；
- 约 12.27M prompt tokens；
- 估算约 31.57 USD。

这次失败说明：

- judge 对候选位置高度敏感；
- 长 bundle prompt 增加了认知负担和顺序效应；
- 未先进行足够 pilot 就直接 full-run，造成不必要成本；
- 该批 pairwise 数据不能作为正式论文质量结论。

完整记录：`project/docs/METACOM_V33_EVOEMO_PAIRWISE_GATE_FAILURE_ZH.md`。

### 3.3 V4 fixed-input absolute scoring：技术成功，但测量构念出现问题

V4 改为 single-turn fixed-input absolute scoring，204 个 unit 全部完成，技术链路和 artifact attestation 有效。

但是后续离线检查发现：

- `Overall` 与 `Emotional Support` 在结果中完全相同；
- 所谓 Overall 实际只是 support 分的复制或同义输出；
- 因而它不能被解释为综合回复质量。

条件均值如下：

| Condition | 原 Overall / Emotional Support |
|---|---:|
| Session Retrieval | 3.750 |
| Context Only | 3.745 |
| Best Fixed | 3.662 |
| Strong Rule | 3.662 |
| PM | 3.657 |
| Full History | 3.618 |

PM 的 paired delta：

| Comparison | PM - baseline | 95% cluster CI |
|---|---:|---:|
| PM vs Strong Rule | -0.005 | [-0.059, 0.044] |
| PM vs Best Fixed | -0.005 | [-0.064, 0.054] |
| PM vs Full History | +0.039 | [-0.064, 0.147] |
| PM vs Session Retrieval | -0.093 | [-0.167, -0.025] |
| PM vs Context Only | -0.088 | [-0.167, -0.015] |

可得出的结论只有：

- PM 与 Rule / Best Fixed 的单步支持质量接近；
- PM 对 Full History 更省且不更差；
- PM 不是回复质量最优；
- Context Only 和 Session Retrieval 在该 judge 构念下更强。

完整记录：

- `project/docs/METACOM_V33_EVOEMO_RESPONSE_V4_RESULT_ZH.md`
- `project/outputs/evoemo_response_v4/response_analysis_summary.md`

### 3.4 资源风险 audit：不同抽样给出不同印象

普通 sampled audit 中，50 个 PM 样本几乎全部 acceptable，容易产生“PM 风险很低”的印象。

但 balanced paired stress pilot 中：

| Policy | n | Major issue | Mean evidence misuse |
|---|---:|---:|---:|
| PM | 12 | 3 | 0.583 |
| Rule | 12 | 2 | 0.417 |
| Fixed All Structured | 12 | 2 | 0.333 |
| Session Retrieval | 12 | 1 | 0.000 |

PM 的 risky cases 全部集中在 `MSE+RS`。

这说明：

- audit 结论对 sampling strata 非常敏感；
- 普通样本不足以证明 PM 普遍安全；
- stress sample 也不能被当作总体风险率；
- “资源风险”应拆成 selected evidence relevance、actual response misuse、unnecessary exposure、staleness、omission 等，而不能只用一个 Overall Risk 排名。

完整记录：`project/docs/METACOM_V33_PM_EVIDENCE_PRECISION_DIAGNOSTIC_ZH.md`。

### 3.5 同预算 `ME+R0` 补充实验：PM 的 learned routing 优势未成立

用户提供的本地 forced-swap 双顺序盲测结果：

| Metric | PM | ME+R0 | PM - ME+R0 |
|---|---:|---:|---:|
| Support | 3.863 | 4.013 | -0.150 |
| Personalization | 3.038 | 3.175 | -0.138 |

偏好结果：

- PM wins：2；
- ME+R0 wins：11；
- ties：27；
- order disagreement：1。

Support delta CI 为 `[-0.250, -0.050]`，Personalization delta CI 为 `[-0.238, -0.050]`。

这一结果说明：

- 之前 `ME+R0` 的优势不是纯粹的 order bias；
- 多数样本仍然是 tie，因此不是“PM 被全面碾压”；
- 最准确的结论是：**PM-v1 没有显示出超过同预算固定 Event Memory policy 的稳定优势**。

该结果来自用户提供的本地脚本和运行日志：

- `scripts/17s_eval_cost_matched_forced_swap_probe.py`
- `outputs/evoemo_cost_matched_forced_swap_probe/`

当前仓库主分支快照中尚未包含该补充输出；正式引用前应把脚本、summary、raw rows 和 cost hash 一并提交并冻结。

---

## 4. 数据设计：表面规模大，但有效语义多样性不足

旧 synthetic development 数据：

```text
source states: 192
runtime cards: 1728 = 192 states × 9 counterfactual inventory variants
users: 16
semantic families: 12
unique current texts: 36
```

相关记录：`snap_reports/METACOM_V33_CURRENT_RESEARCH_REPORT_ZH.md`。

### 4.1 1728 cards 不等于 1728 个独立语言状态

1728 cards 主要来自：

- 192 个 source states；
- 每个 state 做 9 个 inventory counterfactual variants。

因此大量 card 共享：

- 相同 current-user text；
- 相同 recent dialogue；
- 相同 semantic situation；
- 仅 memory availability / inventory metadata 不同。

这对研究 availability counterfactual 有价值，但不能替代真实语言和用户状态多样性。

### 4.2 只有 36 个唯一 current-user texts

这意味着 PM 很容易学习：

- 某个固定表达通常对应哪类动作；
- 某些 synthetic 模板与 memory source 的共现；
- 某种语义 family 的平均资源偏好。

即使 user-fold 不重叠，同一 semantic family、相似句式和相同生成模板仍可能同时出现在训练与验证中。

这不是传统意义上的“同一行泄漏”，但属于：

> semantic/template overlap 导致的乐观内部泛化估计。

### 4.3 user-fold 没有联合隔离 semantic family 和 normalized text

最终 selection 采用：

```text
fold_mode: user
feature_mode: text_metadata / text_metadata_stable
n_folds: 5
```

用户不重叠并不等于语义不重叠。内部 validation 仍可能看到训练中出现过的：

- 同一 semantic family；
- 高度相似的 current turn；
- 相同 resource-need pattern；
- 相同 synthetic generator 风格。

这解释了为什么内部 validation 的 M0/R0 分布看似多样，但到 EvoEmo 后迅速退化。

### 4.4 没有显式、均衡的 resource-need regimes

旧数据不是按以下状态强制均衡构造的：

- context only；
- memory harmful；
- profile needed；
- summary needed；
- event needed；
- multi-source needed；
- strategy helpful；
- strategy harmful；
- genuinely ambiguous。

因此即使 `M0+R0` 在动作空间中存在，也不代表数据中有足够多“它确实应该赢”的状态。

同理，`R0` 存在不代表存在足够多 Strategy RAG 会造成模板化、过度指导或不必要结构化的正例。

### 4.5 各状态允许动作数量不同

旧数据中的 action availability：

```text
2 actions: 192 cards
4 actions: 768 cards
8 actions: 576 cards
16 actions: 192 cards
```

只有 192 / 1728 cards 暴露完整 16 动作。结果是：

- 某些动作只在少数 inventory-rich states 上被训练；
- 不同动作有效样本数不平衡；
- `MPMSME` 等动作与“资源很多的特殊状态”绑定；
- fixed action 跨状态比较更困难；
- PM 很难学习统一的全动作决策边界。

### 4.6 synthetic → EvoEmo 存在明显 domain shift

EvoEmo 有：

- 18 个长期用户；
- 401 个历史 sessions；
- 自然表达、时间线、跨 session 主题演化；
- 远大于 synthetic users 的 memory inventory；
- 更复杂的情绪状态与真实语言变体。

旧版通过 metadata cap / stable mode 缓解了数值尺度 OOD，但没有解决：

- 语言语义 OOD；
- resource-need pattern OOD；
- synthetic label preference 与 EvoEmo judge construct 不一致。

---

## 5. 标签设计：不是直接作弊，但 silver labels 的构念和区分度不足

### 5.1 `Overall` 失效

旧版让 LLM 同时输出：

- Overall；
- Emotional Support；
- 其他维度。

结果 `Overall` 与 Emotional Support 完全相同，说明 judge 没有形成真正独立的 holistic construct。

影响：

- 主表把 support 分误称为 Overall；
- 质量结论看起来比实际更宽；
- PM 的训练和论文解释无法区分“情感承接好”与“综合回复好”。

根因：

- 两个字段定义过近；
- 同一个 prompt 同时要求多个相关维度；
- LLM 倾向复用已有分数；
- 没有 duplicate-dimension fail-closed gate。

### 5.2 风险标签存在退化和 schema/verdict 不一致

balanced audit 中曾出现：

```text
verdict = major_issue
overall_risk = 0
selected_evidence_misuse = 0
```

这说明 judge 的自然语言 verdict、主风险字段和 source-set appropriateness 之间没有稳定映射。

旧版部分 risk dimensions 还表现出低方差或近常量。这会导致 Ridge risk heads 学到接近常数，而不是状态条件风险。

### 5.3 训练标签来自单一 silver judging pipeline

训练主 judge 为 Gemini Flash-Lite，最终外部 judge 为 GPT-4o，family 独立性比使用同一模型好；但训练标签仍然是纯 LLM silver labels：

- 没有两种 judge family 对每条 action outcome 交叉校准；
- 没有逐维 MAD / disagreement 过滤；
- 没有人工 blind audit 作为冻结门禁；
- judge 偏好可能被训练成 PM policy。

因此，PM 学到的是：

> 在 synthetic development distribution 上，如何预测该 judge pipeline 的 action outcome labels。

它不等同于学习真实用户长期偏好。

### 5.4 pairwise 标签容易受顺序影响

虽然内部 measurement protocol 使用 dual-order debiasing，并把不一致 pair 设为 tie，但外部长 bundle pairwise 仍然严重 orientation-sensitive。

这说明：

- LLM pairwise preference 并非稳定 ground truth；
- 长上下文和多 case bundle 会放大位置偏差；
- 训练中剩余的 pairwise noise 可能削弱 response ranker。

### 5.5 tie 被排除会损失“多个动作接近”的信息

旧 response ranker 训练只使用非 tie pairs。大量动作本来可能是实际等价或近似等价的，但 tie 不进入主要分类学习。

后果：

- 模型被迫学习微小、噪声性的 winner；
- selection 容易形成 winner-take-all；
- 无法直接学习“质量接近时优先更低风险、更低成本”。

---

## 6. 特征表示与模型：内部能拟合，但外部条件路由能力不足

### 6.1 主模型使用 fitted TF-IDF

旧 FeatureBuilder 的主文本表示来自训练 states 拟合的 TF-IDF vocabulary。

问题：

- EvoEmo 中的新词和新表达会成为 OOV；
- 语义相近但词面不同的表达不能稳定映射；
- 36 个 unique current texts 很容易形成模板词特征；
- fitted vocabulary 强化内部 validation，弱化外部自然语言泛化。

### 6.2 最终外部模型是 `text_metadata_stable`，不读取 catalog semantic relevance

最终 EvoEmo readiness 记录：

```text
feature_mode: text_metadata_stable
```

这意味着 PM 主要看到：

- current/recent text；
- source availability；
- count / age / estimated tokens；
- session depth。

它**不读取 actual memory text，也不使用 source catalog relevance**。

因此对于两个具有相同 inventory count/age 的状态，PM 很难判断：

- Event Memory 具体是否与当前主题有关；
- Summary Memory 是相关模式还是噪声；
- Profile Memory 是有用边界还是无关个人信息。

这不是泄漏缺失，而是可观测信息不足。

### 6.3 线性 Logistic / Ridge heads 表达能力有限

旧 PM 使用：

- pairwise Logistic Regression 预测 response preference；
- Ridge 回归预测 misuse、omission、memory quality、strategy risk / quality。

即使加入 state × action gating，模型仍主要依赖线性可加关系。真实决策可能需要：

- 否定、转折和语义组合；
- 当前情绪 × 历史事件类型；
- session depth × source relevance；
- strategy repetition × 当前支持阶段；
- 多来源之间的互补或冲突。

线性模型容易学到整体 action prior，而不是细粒度状态交互。

### 6.4 多个 head 的分数没有统一校准

旧 response score 是 pairwise logit 的 sigmoid；风险 head 是 clipped Ridge prediction。它们不是同一个概率空间，也没有 uncertainty interval。

因此：

- `0.35` 的 misuse threshold 不一定具有可迁移意义；
- `1.0` 的 omission/strategy threshold 实际相当于关闭门禁；
- 无法知道两个 action 的 predicted quality 差异是否只是模型不确定性。

### 6.5 OOD 检测主要处理 feature range，而不是结果可靠性

旧版后来加入 metadata OOD 和 stable caps，这是有价值的防护。但：

- clip/cap 只能限制数值越界；
- 不能证明语义在训练分布内；
- 不能证明 action ranking 在新语言中可靠；
- READY 只表示代码允许推理，不表示 PM 有外部效度。

---

## 7. 最终选择规则：质量优先，风险弱过滤，成本几乎没有参与

最终 `outputs/selection.json` 的 PM 配置：

```text
epsilon = 0.0
tau_misuse = 0.35
tau_omission = 1.0
tau_strategy = 1.0
confirmatory_allow_constraint_fallback = false
```

### 7.1 `epsilon=0` 是最关键的结构性原因

旧选择逻辑：

1. 过滤风险超阈值动作；
2. 找安全动作中的最高 predicted response score；
3. 只保留与最高分差不超过 epsilon 的动作；
4. 在候选中按 estimated cost 选择更便宜动作。

当 `epsilon=0` 时：

- 通常只有预测质量最高的动作进入候选；
- cost 只有在完全同分时才生效；
- 所谓 quality–risk–cost 实际接近 quality winner-take-all。

因此旧版不能严谨声称：

> PM 学习并联合优化了质量、风险与成本。

更准确的是：

> PM 先按有限风险门禁筛选，再以预测回复质量为主，成本仅用于近乎不存在的并列决策。

### 7.2 omission 和 strategy risk 门禁实际上被关闭

风险 prediction 被限制在 `[0,1]`，而：

```text
tau_omission = 1.0
tau_strategy = 1.0
```

只有超过 1.0 才会被过滤，而模型输出不可能超过 1.0。

因此最终真正起作用的主要只有：

```text
tau_misuse = 0.35
```

结果是：

- M0 omission 没有真实参与动作过滤；
- Strategy overuse / omission 没有真实参与动作过滤；
- RS 很容易因为微小预测质量优势被打开。

### 7.3 大多数外部状态几乎所有动作都被视为安全

EvoEmo readiness：

```text
14 safe actions: 930 states
15 safe actions: 41 states
16 safe actions: 49 states
```

这意味着风险 heads 在外部选择中几乎没有产生有效区分。

### 7.4 成本使用的是估算 action cost，不是直接训练的 observed total cost utility

旧 `estimated_action_cost` 根据：

- 每来源固定 top-k；
- inventory estimated tokens；
- strategy 固定 token；
- retrieval call penalty；

估算动作成本。

它不是直接由每个 action outcome 的实际总 input tokens 与质量、风险共同优化得到的 utility。因此实际成本只间接影响 tie-break。

---

## 8. 外部动作分布说明 PM 出现 policy collapse

EvoEmo 1020 states 的 PM 动作分布：

| Action | Count |
|---|---:|
| MSE+RS | 521 |
| MPMS+RS | 165 |
| MP+RS | 128 |
| MS+RS | 72 |
| MPE+RS | 62 |
| ME+RS | 34 |
| MSE+R0 | 27 |
| ME+R0 | 3 |
| MPMS+R0 | 3 |
| MS+R0 | 2 |
| MP+R0 | 2 |
| MPMSME+RS | 1 |

由此可得：

- `M0`：0 / 1020；
- `RS`：983 / 1020，约 96.4%；
- `MSE+RS`：521 / 1020，约 51.1%。

这不是“PM 学会根据状态自由使用 16 种动作”，而是：

> 外部分布下，PM 强烈偏向打开 Strategy RAG，并集中选择多来源 memory action。

内部 validation 中 PM 曾选择：

- `M0+R0`: 91；
- `M0+RS`: 337；
- 多种 R0 / RS 动作。

内部与外部的巨大差异说明：

- internal validation 的 action diversity 没有迁移；
- user-fold validation 高估了条件路由泛化；
- 数据和特征无法稳定识别 external abstention 状态。

相关 artifact：`project/outputs/evoemo_pm_readiness_stable_frozen.json`。

---

## 9. Retrieval 层：PM 选的是 source，不是具体证据

### 9.1 固定 top-k lexical retrieval 无 relevance threshold

旧 MemoryRetriever 使用词面相关性排名并固定返回：

- MP top-2；
- MS top-2；
- ME top-3。

没有：

- minimum relevance threshold；
- “没有合格 item 则返回空”；
- staleness / conflict reranking；
- privacy / intrusiveness filtering。

所以一旦 PM 选择 `MSE`，系统会强制从 MS 和 ME 返回固定数量的 snippets，即使候选质量很弱。

### 9.2 source action 粒度过粗

`MSE+RS` 同时表示：

- 调用 Summary Memory；
- 调用 Event Memory；
- 打开 Strategy RAG。

它不能表示：

- MS 有用但 ME 无合格条目；
- 只返回一条 Event Memory；
- 当前 Strategy Bank 没有合格卡；
- 使用 source 但对所有候选 abstain。

因此 PM 的 source-level correct 不等于 evidence-level correct。

### 9.3 生成质量标签混合了三层误差

一个 action outcome 的最终 judge 分数同时受以下影响：

1. PM 是否选对 source；
2. retriever 是否找对 item；
3. frozen generator 是否正确使用 item。

旧模型只学习最终 outcome，无法清楚归因。若 `MSE+RS` 得分低，原因可能是：

- source 不适合；
- source 适合但 item 错；
- item 合适但 generator 忽略或误用；
- judge 对回复风格有偏好。

这会给 PM 训练带来高噪声 credit assignment。

---

## 10. 为什么 `ME+R0` 在当前 EvoEmo 上特别强

`ME+R0` 的优势有合理机制解释，但不能被外推为普遍最优策略。

### 10.1 EvoEmo subsequent topics 与具体历史事件天然对齐

EvoEmo 的后续主题通常与过去某个 concrete episode 相关。ME 保存 seeker 的具体事件表达，因此比：

- 过于稳定的 MP；
- 过度压缩的 MS；
- 全部历史；

更容易提供直接、具体且不过量的个性化信息。

### 10.2 R0 避免了 Strategy RAG 的模板化和过度结构化

旧 PM 96.4% 打开 RS。Strategy cards 可能带来：

- 重复反映情绪；
- 过早建议；
- 模板化支持；
- 回复长度增加；
- 当前 turn 已经足够清楚时的不必要结构。

`ME+R0` 保留个性化事件信息，同时避免额外策略约束，刚好符合当前 fixed-input judge 对自然承接和非侵入性的偏好。

### 10.3 同预算 fixed action 没有 routing error

当 external distribution 本身使 ME 在多数样本有用时，固定 `ME+R0` 不需要预测“什么时候用”。

PM 则多了一层可能出错的 routing：

- 错开 MS；
- 错开 RS；
- 错选多来源；
- 无法选择 M0。

若动态选择带来的收益不足以覆盖 routing error，固定动作就会更强。

### 10.4 多数 forced-swap 样本为 tie

27 / 40 为 tie，说明两者在很多状态接近。结论应是：

- PM 尚未证明动态路由增益；
- 不是 ME+R0 在所有状态绝对最优；
- 需要更有区分力的数据、人工校准和真实的 resource-need states。

---

## 11. Context Only 和 Session Retrieval 为什么会更高

### 11.1 fixed-input judge 强烈奖励当前轮连贯性

当前 user turn 和 recent dialogue 已经包含足够信息时，额外 memory 可能：

- 重复旧内容；
- 引入无关事实；
- 降低自然度；
- 显得过度个性化。

Context Only 因为没有个人历史风险，容易在 emotional support、factual grounding 和 non-intrusiveness 上得高分。

### 11.2 Context Only 高分不等于长期记忆无用

它只能说明：

> 在被抽样的固定输入 turn 上，当前上下文足以写出一条高分回复。

它不能说明：

- 系统能长期记住用户；
- 用户跨 session 感到被理解；
- 需要事实回忆时仍然正确；
- 长期关系、信任和状态跟踪不需要 memory。

### 11.3 Raw Session Retrieval 保留了丰富原始语境

Session Top-k 可能比结构化 MP/MS/ME 更强，因为它：

- 保留上下文细节；
- 不依赖摘要抽取质量；
- 对 subsequent topic 有较高检索命中率。

但它成本更高，也可能带来隐私和冗余风险。因此其高质量并不等于它是最佳部署 policy。

---

## 12. 外部评价范围限制

### 12.1 fixed-input 是单步因果比较，不是长期闭环

fixed seeker tracks 解决了不同 policy 产生不同世界线的混淆，这是正确设计。但代价是：

- 后续 seeker turn 不会根据 PM response 发生真实反应；
- 不能评价 trust、engagement、情绪轨迹和长期帮助；
- 不能评价错误记忆对后续多轮关系的累积影响。

因此它是：

> causal one-step response stress test

而不是完整 longitudinal outcome evaluation。

### 12.2 只抽样 turn 3 和 turn 8

V4 只评 204 units，即 102 个 track/scenario 上的两个 turn。它没有覆盖全部 1020 turns 的回复质量。

风险：

- turn 3/8 可能偏向某种支持阶段；
- 可能错过真正需要长期记忆的 turn；
- action difference 和 response difference 不一定在这两个位置最大。

### 12.3 用户数量小，cluster 数有限

EvoEmo 只有 18 个用户、34 个后续主题。虽然使用 scenario/user cluster bootstrap 比把每个 turn 当独立样本更合理，但：

- user-level power 仍有限；
- 个别用户或主题可能主导均值；
- 不能代表真实人群分布。

### 12.4 最终主 judge 仍是 LLM

V4 最终 judge 为 GPT-4o。没有：

- 人类情感支持专家；
- 真实用户偏好；
- 临床专家；
- 多 family pointwise aggregate。

因此结果只能解释为：

> 当前 judge rubric 下的模型评分。

### 12.5 multi-candidate absolute scoring 仍有 candidate-set effect

V4 已平衡 candidate order，但同一个 prompt 中存在多个候选。某个候选的绝对分仍可能受：

- 其他候选质量；
- 候选数量；
- 位置；
- judge 的相对比较心理；

影响。

### 12.6 latency 只是 observational diagnostic

现有生成日志显示 PM inference 本身很快，但 total latency 受：

- provider scheduling；
- 网络；
- retrieval path；
- 输出长度；
- API 时段；

影响。旧实验不是 randomized serving benchmark，不能声称 PM 普遍更快。

### 12.7 sampled audit 不能估计总体安全率

普通 audit 和 stress audit 结果不同，说明抽样目的必须明确：

- representative sample 才能估计 prevalence；
- stress sample 用于发现边界；
- 二者不能混成一个总体安全结论。

---

## 13. PM-v1 并不是 RL，也没有在线适应

PM-v1 的真实方法是：

1. 对 synthetic state 枚举 legal actions；
2. 固定 generator 生成 counterfactual responses；
3. LLM judge 产生 response/risk labels；
4. 监督学习 state–action outcome；
5. 在线对每个 state 选择一个动作。

它没有：

- 环境 transition model；
- 用户后续反应 reward；
- episode return；
- policy gradient；
- actor–critic；
- exploration；
- 在线更新；
- POMDP belief state。

因此不能声称：

- PM 通过强化学习学会长期支持；
- PM 优化长期用户状态；
- PM 会根据实际用户反馈持续适应。

---

## 14. 风险概念也必须收紧

旧版的 risk 主要是：

- memory misuse；
- unnecessary exposure；
- stale/conflicting use；
- unsupported personal claim；
- omission；
- strategy overuse / omission。

这不是：

- 自伤风险识别；
- 临床安全；
- 精神疾病诊断；
- 医疗干预安全；
- 真实隐私合规认证。

论文中应写 `resource-use risk`、`memory-use risk` 或 `evidence-use risk`，不要笼统写成 clinical safety。

---

## 15. 哪些问题不是最终失败的主要原因

### 15.1 没有证据表明最终结果由直接数据泄漏造成

最终 V3.3 已有：

- future-memory audit；
- memory backend 与 runtime state 物理分离；
- action-mask validation；
- artifact attestation；
- study freeze；
- fixed-input tracks；
- external evaluator-only context 隔离；
- static leakage tests。

因此当前更合理的判断是：

> 失败主要来自数据覆盖不足、语义泛化不足、silver labels 构念问题和选择规则失衡，而不是已经证实的作弊或 future information leakage。

### 15.2 生成器随机性不是主要解释

已有 generator variance diagnostic 通过，且正式生成使用低温度/固定 seed。生成器仍有噪声，但不足以解释 M0=0%、RS=96.4% 和同预算 fixed 持续更强等系统性现象。

### 15.3 order bias 不能完全解释 `ME+R0` 优势

forced-swap 后趋势仍存在，且 order disagreement 只有 1 / 40。因此 PM 的同预算劣势不是纯 evaluator position artifact。

---

## 16. PM-v1 仍然成立的贡献

即使存在以上失败，以下贡献仍然有效：

1. **问题定义有效**：长期 ESC 中需要独立的 pre-retrieval resource allocator。
2. **16-action action space 有解释性**：memory source subset 与 Strategy RAG on/off 可以明确拆分。
3. **因果边界清楚**：PM 在 evidence retrieval 前决策，不偷看 snippets。
4. **更多资源不一定更好**：Full History 成本最高且回复分最低之一。
5. **资源成本差异显著**：PM 相对 Full History 输入 token 约下降 90.8%。
6. **PM inference 开销很小**：主要部署成本来自 retrieval 与 generation，不是 PM 本身。
7. **负面结果有研究价值**：简单 `ME+R0` 可以击败复杂 supervised router，揭示 routing calibration 的真实难度。
8. **工程审计框架有价值**：freeze、attestation、fixed-input、budget gate、OOD gate 和 sampled audit 可被后续工作复用。

---

## 17. PM-v1 不能再提出的主张

禁止写：

- PM 全面提升回复质量；
- PM 优于所有 fixed / rule baselines；
- PM 已经联合优化质量、风险和成本；
- PM 学会了什么时候不用资源；
- PM 精确选择了具体 memory evidence；
- PM 证明长期记忆提升所有 ESC 回复；
- PM 是 RL / POMDP / adaptive long-term policy；
- PM 提升了真实用户长期心理状态；
- PM 已达到临床安全。

可写：

- PM 与 Rule / Structured Fixed 在 fixed-input support quality 上接近，并降低输入资源；
- PM 明显优于 Full History 的资源效率，且 Full History 并未带来更高回复质量；
- Context Only 的强表现说明 abstention 是核心决策，而 PM-v1 尚未学会；
- 同预算 `ME+R0` 的优势说明 learned routing 仍需更好的数据、标签、calibration 和人类反馈；
- 高资源动作的 evidence precision 需要独立的 post-retrieval selector。

---

## 18. 根因汇总表

| 观察到的问题 | 直接原因 | 更深层原因 |
|---|---|---|
| M0 外部选择率 0% | 训练/选择没有足够 abstention 信号 | 数据未平衡 context-only / memory-harmful regimes；外部语义 OOD |
| RS 选择率 96.4% | tau_strategy=1.0，RS 风险不约束；微小质量优势即可开启 | Strategy-off 正例不足；标签偏向结构化支持 |
| cost 几乎不生效 | epsilon=0，cost 只用于完全同分候选 | 多目标问题被实现成 quality-first lexicographic rule |
| PM 集中 MSE+RS | 线性模型学习 action prior；source relevance不可见 | TF-IDF + metadata 难以泛化；synthetic pattern 偏差 |
| ME+R0 胜 PM | Event Memory 与 EvoEmo topic 对齐；R0 避免策略过度 | PM 的动态 routing error 大于条件路由收益 |
| Context Only 高分 | 当前轮信息已足够；无记忆风险 | fixed-input judge 偏好局部连贯和非侵入性 |
| Session Retrieval 高分 | raw session 保留丰富语境 | structured memory 抽取/粒度损失；PM 选错来源 |
| PM audit risk 较高 | 固定 top-k 返回弱相关 snippets | PM 只选 source，不选 item；无 evidence threshold |
| Overall 与 support 相同 | judge 复用相关评分 | rubric 构念未独立；无 duplicate-dimension gate |
| pairwise gate 失败 | 强 position/orientation bias | 10-case bundle 太长、候选比较负担过高 |
| 内部动作多样、外部 collapse | user-fold 泛化过于乐观 | semantic/text overlap；36 unique texts；domain shift |
| 风险 heads 缺少区分 | silver risk labels 低方差/不一致 | 单一 judge pipeline、无人工 calibration |

---

## 19. PM-v2 必须针对性解决的事项

PM-v2 不是简单调 `epsilon`，必须逐项回应旧版根因：

| PM-v1 问题 | PM-v2 对应设计 |
|---|---|
| 36 个 unique texts、semantic overlap | train/calibration/internal-test 同时 user、semantic family、normalized text disjoint |
| 没有均衡 resource regimes | 每个 synthetic user 强制九类 resource-need cases |
| 不是所有 state 都有 16 actions | 每个 development state 必须同时存在 MP/MS/ME，完整 sweep 16 actions |
| Overall 复制 support | schema 完全删除 Overall；只保留六个独立维度 |
| 风险维度退化 | response/risk duplicate、constant、reliability fail-closed |
| 单一 silver judge | 至少两个 independent judge families + median + MAD |
| 无人工校准 | reportable freeze 前强制 blind human audit |
| fitted TF-IDF OOV | fixed word hashing + char hashing + optional semantic embedding |
| 线性点估计 | bootstrap nonlinear outcome ensembles + uncertainty |
| cost 只 tie-break | `quality LCB - risk weight × risk UCB - cost weight × normalized cost` |
| M0/R0 没学会 | explicit memory-benefit gate、strategy-benefit gate、M0/R0 coverage gate |
| 内部无强 fixed 门禁 | calibration-selected cost-matched fixed + internal reportability gate |
| 只看平均指标 | action diversity、M0 rate、R0 rate、regime alignment 同时检查 |
| OOD 只数值 cap | semantic + metadata OOD，严重 OOD 明确 fallback 并记录 |
| PM 与 fixed 不同运行路径 | fixed-action checkpoints 使用与 PM-v2 完全相同的 EvoEmo runtime |
| 外部 multi-candidate effect | single-candidate pointwise multi-family evaluation |

---

## 20. 后续研究仍无法仅靠 PM-v2 解决的限制

即使 PM-v2 通过全部内部 gate，以下问题仍然需要第二篇研究或额外人类实验：

- 真实用户下一轮反应；
- 多轮情绪轨迹和信任变化；
- 长期改善是否真实发生；
- 用户是否感到被理解；
- 高风险心理状态的识别与退出；
- 临床专家认可的安全阈值；
- 个体隐私偏好和真实 consent；
- simulator 是否迎合 supporter；
- 真实部署中的延迟和成本稳定性。

这些属于用户模拟器、POMDP/RL、专家校准和真人验证的研究范围，不能再由单步 supervised PM 代替。

---

## 21. 最终研究结论

PM-v1 最有价值的结果不是“一个成功击败所有 baseline 的 PM”，而是明确暴露了以下事实：

1. 长期 ESC 中，always-on memory / Strategy RAG 并不可靠；
2. Context Only 很强，所以 abstention 必须成为第一类动作，而不是异常 fallback；
3. 一个简单同预算 `ME+R0` 可以比复杂 PM 更强，说明条件路由比动作空间枚举困难得多；
4. 数据规模不能只按 card/action rows 计算，语言、语义和 resource-need 多样性才决定可学习性；
5. LLM judge 的字段名不等于有效测量构念，Overall、risk 和 preference 都必须有独立性与人工校准；
6. quality、risk、cost 必须同时进入统一决策 utility，不能把 cost 放在最后打破并列；
7. coarse resource routing 与 fine-grained evidence selection 必须分层；
8. 外部评测前必须有 internal cost-matched baseline gate，否则会在高成本实验后才发现 policy 没有增益。

因此，PM-v1 应被定位为：

> 一个完成了问题定义、因果边界、工程链路和关键负面发现的第一代监督式资源调度器；它证明了资源分配问题存在，但也证明了旧数据、标签、特征和 selection rule 不足以学习可靠的外部条件路由。

---

## 22. 主要证据与 artifact 索引

### 数据与方法

- `snap_reports/METACOM_V33_CURRENT_RESEARCH_REPORT_ZH.md`
- `project/src/metacom_pm/features.py`
- `project/src/metacom_pm/training.py`
- `project/src/metacom_pm/policies.py`
- `project/src/metacom_pm/selection.py`
- `project/outputs/selection.json`
- `project/outputs/evoemo_pm_readiness_stable_frozen.json`

### 外部评价

- `project/docs/METACOM_V33_EVOEMO_PAIRWISE_GATE_FAILURE_ZH.md`
- `project/docs/METACOM_V33_EVOEMO_RESPONSE_V4_RESULT_ZH.md`
- `project/docs/METACOM_V33_EVOEMO_RESPONSE_V4_NOTE_TO_GPT55_ZH.md`
- `project/outputs/evoemo_response_v4/response_analysis_summary.md`
- `project/outputs/evoemo_response_v4/response_statistical_summary.json`
- `project/outputs/evoemo_response_v4/response_resource_summary.json`

### 风险与 evidence precision

- `project/docs/METACOM_V33_PM_EVIDENCE_PRECISION_DIAGNOSTIC_ZH.md`
- `project/outputs/evoemo_response_v4/pm_resource_saving_subgroups.md`
- `project/outputs/evoemo_balanced_paired_audit/pm_evidence_precision_diagnostic.json`

### 历史修复与边界

- `snap_reports/METACOM_V33_GITHUB_REVIEW_FIX_REPORT_20260628_ZH.md`
- `project/docs/CLAIM_BOUNDARIES_CN.md`
- `project/tests/test_scientific_safeguards.py`
- `project/tests/test_freeze_gate.py`
- `project/tests/test_m2b_fail_closed.py`

### 本地补充证据，待正式提交

- `scripts/17s_eval_cost_matched_forced_swap_probe.py`
- `outputs/evoemo_cost_matched_forced_swap_probe/`

---

## 23. 文档维护规则

后续发现新的 PM-v1 负面结果时，应按以下格式追加：

1. 结果是协议失败、指标失败、真实负面结果还是范围限制；
2. 是否经过 freeze / attestation；
3. 样本和统计单位；
4. 与哪个 baseline 比较；
5. 是否有 order / candidate-set / sampling bias；
6. 是否能归因给 PM、retrieval、generator 或 judge；
7. 对论文 claim 的具体影响；
8. PM-v2 是否已经针对该问题设置 fail-closed gate。

禁止把探索性 post-hoc diagnostic 重新包装成预注册 confirmatory finding。
