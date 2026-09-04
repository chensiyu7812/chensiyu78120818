# PM-v1.5 统一需求语义、可学习数据与裁判方案 V3

> **2026-07-28 范围更新：** 本文保留为 V2.0 的细粒度诊断/多组件方法设计与 V1.5
> 的历史研究依据；快速 V1.5 不再要求完成五类/全因子 SupportNeed fit 或继续补标，
> 当前唯一执行入口见 `PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`。已完成的 39-group
> 结果可以作为候选特征证据，但不能被包装成“已理解用户需求”。

更新时间：2026-07-27
用途：定义 PM 应学会的行为、如何从可见对话形成非临床的支持需求表征、BAAI 语义表示如何
与 PM 正确连接、Strategy Bank 必须提供什么、能产生可辨识监督的数据，以及透明的分工评价
标准。本文不是运行许可或状态清单；它回答的是“理解什么、学什么、用什么数据学、裁判究竟
按什么判”。它完整继承 V2 的 matched-treatment 与拆职裁判设计，并曾取代 V2 成为当时活跃
的实质方法方案。

## 1. PM 到底要学会什么

PM 是同一个 pre-retrieval source-level router，不是 ESConv PM 和 EvoEmo PM 两个模型。对
任一当前状态 \(s\)，它选择：

```text
action = (memory subset ∈ {M0, MP, MS, ME, ...}, strategy mode ∈ {R0, RS})
```

目标行为按以下顺序定义：

1. 遵守来源可用性。不可用的 memory source 永远不能选择；ESConv 是 memory 全不可用的
   合法边界情形，而不是另一项任务。
2. 预测每个 MP/MS/ME/RS 组件相对当前背景动作的**增量价值**，而不是预测一个动作的绝对
   “好看分数”。
3. 只有在预期支持质量增益足以覆盖证据误用风险、不必要暴露风险和确定性成本时才开启组件。
4. 证据不足、多个组件互相干扰或不确定性过高时，保守选择 M0 和/或 R0。
5. 学习 source×source、source×RS 交互；不能假定三个单源分别有益就意味着三源同时有益。
6. 只选择来源类型和 RS 开关，不声称 PM 选择了最优 item；item-level retrieval 的错误必须
   与 PM source decision 分开报告。

因此第一篇论文可检验的核心不是“PM 找到全局最优回复”，而是：

> 在冻结生成器、prompt、检索器和可见观测下，一个共享 PM 能否从小样本有噪监督中学到
> 可重复的组件增量方向，并在不降低回复质量的前提下减少不必要的 memory/RS 使用、风险和
> 成本。

## 2. 为什么旧数据学不出来

已完成的真实诊断给出三段因果证据：

- 干净的单条 oracle memory 能被同一生成器吸收：helpful `9/12` 为正，harmful `4/6`
  非正；说明生成器不是完全没有利用能力。
- 旧 production treatment 的 helpful 单源行全部混入 irrelevant item；多源 helpful 行同时
  塞入六项 evidence。真实多源分解中 `7/9` 个单源为正、`8/9` 个单源优于 all-three，
  说明 evidence dilution/interference 是主要原因。
- 旧标签从同一 state 的 16 个含噪绝对分数中取 argmax，产生 winner's curse；即使出现
  non-M0 pseudo-oracle，也不等于存在稳定可学习的组件边际值。

所以旧失败不能只归因给模型，也不能只归因给裁判。旧训练样本的 treatment 本身没有稳定
产生预期效果，裁判又被要求从微小、混杂的回复差异中输出过细的绝对分数。

## 3. 新训练数据必须长什么样

### 3.1 一个训练单元

最小因果单元是同一 state 下的 matched pair，不是孤立的 state-action 行：

```text
(same state, same generator, same prompt, same seed, same decoding)
control evidence surface  vs  exactly-one-component treatment surface
```

除研究操纵的一个组件外，其他内容逐字一致。标签目标是 treatment effect，而不是 action 的
绝对得分。

### 3.2 必须覆盖的干净对照格

| 组件 | 背景动作 | control | treatment | 期望回答的问题 |
|---|---|---|---|---|
| MP | M0+R0、M0+RS | 无 memory | 恰好一条相关 MP | profile 是否带来增量价值 |
| MS | M0+R0、M0+RS | 无 memory | 恰好一条相关 MS | summary 是否带来增量价值 |
| ME | M0+R0、M0+RS | 无 memory | 恰好一条相关 ME | event 是否带来增量价值 |
| placebo | 同上 | 无 memory | 恰好一条无关但自然的同源 item | PM 能否拒绝无关来源 |
| harmful | 同上 | 无 memory | 恰好一条陈旧/冲突/诱导暴露 item | PM 能否避免有害来源 |
| RS | memory unavailable/available | R0 | 一张相关 strategy card | 何时开启 RS |
| RS mismatch | memory unavailable/available | R0 | 一张不合时宜 strategy card | 何时保持 R0 |
| interaction | 已证明 uptake 的单源背景 | 单源 | 第二个单源或 RS | 是否存在增益、稀释或冲突 |

正例 treatment 禁止同时混入 helpful 与 irrelevant/harmful item。多源行只能在各单源已证明
uptake 后生成，不能用 all-three 作为第一批训练正例。

### 3.3 数据内容要覆盖的能力，而不是复刻外部答案

训练状态应覆盖同一决策函数的不同合法区域：

- memory 全不可用、只判断 R0/RS；
- 某一来源可用且有明确增量；
- 来源可用但无关，应 abstain；
- 来源陈旧、冲突或会造成不必要暴露；
- 用户明确要求“先听、不建议”与明确请求方案的 readiness 差异；
- 单源有益、多源互扰；
- 大 catalog count、不同 age range、不同 session index 的 observable metadata 形状。

外部形状增强只能使用 outcome-free metadata 分布，不能使用 ESConv test、EvoEmo 或
ES-MemEval-derived 的答案、偏好或最终成绩。目的不是“背考试题”，而是让训练覆盖考试会
考的能力和输入形状。

### 3.4 扩量前的生成器 uptake 门

每个组件先在 train-only 小样本验证 treatment 是否真的改变了回复：

- helpful treatment 大多数方向为正；
- placebo 接近零；
- harmful 多数非正；
- treatment effect 不能只由回复长度解释；
- 多源必须分别报告相对 control 和最佳单源的增量。

若回复本身没有形成可见差异，停止生成更多标签；此时换裁判也救不了训练。

## 4. 裁判职责必须拆开

### 4.1 岗位 A：支持质量的盲化成对比较

输入只包含 visible dialogue 和两个匿名回复。**不展示 selected memory、strategy、
authorized hidden context、action ID、regime 或预期赢家。**

只输出一个目标：

```text
support_preference ∈ {A, B, tie, abstain}
decisive_criterion
response_a_excerpt
response_b_excerpt
confidence ∈ {1,2,3}
```

判定优先级固定如下，前一项的明显违反不能被后一项的文采补偿：

1. 用户本轮明确边界：例如“先听、不建议”“只问一个问题”“现在想要具体步骤”。
2. 对话连续性：不重复询问 history 已明确的信息，不改变人物、时序或事实。
3. 情绪贴合：承认当前体验，不过度表扬、诊断、正常化或推断未表达的感受。
4. 互动负担：问题聚焦、一次一个任务；不给未被邀请的多步骤作业。
5. 帮助深度：只有用户准备好时，具体、可拒绝、与当前困扰相符的建议才加分。
6. 表达质量：自然、简洁、非模板化；长度本身不加分。

选择 `tie`：两条回复在上述优先级下实质等价或存在不可排序的权衡。
选择 `abstain`：可见信息不足、两条都严重有问题、或无法应用规则。
禁止另设一个与 support 高度重复的 `overall_preference`。

### 4.2 岗位 B：逐回复证据审计

这不是 A/B 偏好，也不输出总体分。先从回复中抽取每个可核验的个性化/记忆依赖主张，再逐条
映射可见证据：

```text
claim_excerpt              # 回复原文精确子串
evidence_excerpt           # visible dialogue 或 selected evidence 精确子串
status ∈ {
  supported,
  unsupported,
  stale_or_conflicting,
  unnecessarily_exposed,
  insufficient
}
severity ∈ {0,1,2,3}
```

规则：

- exact excerpt 由代码验证，不接受“current_user_text says...”这类重述；
- 没有个性化主张时返回 `no_personalized_claim`，不强迫编造多行 `no_violation`；
- “使用了 selected evidence”不自动加分；必须反映在最终回复且适合当前需要；
- 未选择 memory 不等于不存在 omission。omission 只能在独立的 matched treatment 中，通过
  “加入该证据是否改善回复”判断，不能仅凭 selected context 为空推断；
- `insufficient` 不是零风险证据，也不进入负例。

### 4.3 岗位 C：strategy/readiness 审计

只回答一个问题：当前回复的结构化推进程度是否匹配用户 readiness？

```text
strategy_use ∈ {
  appropriate,
  overstructured_or_premature,
  useful_strategy_omitted,
  no_strategy_needed,
  abstain
}
```

判据依次看：用户明确请求、当前情绪强度、是否已被充分承认、建议是否可拒绝、是否一次推进
过多任务。不得因为 action=RS 就推定“用了策略”，也不得因为 action=R0 就推定“遗漏策略”。

### 4.4 岗位 D：确定性事实、风险和成本

以下优先用代码而非 LLM：

- 引用是否为输入文本精确子串；
- 人物、时间和数字的显式矛盾；
- memory/strategy 是否实际进入 prompt；
- token、retrieval call、item count 和费用；
- action 合法性与来源可用性。

高风险但无法规则化的少数项再交给岗位 B/C；N/A 必须保持 N/A，不能编码为 0。

## 5. Prompt 应怎样写

每个 prompt 必须包含：

1. 单一岗位和单一决策问题；
2. 完整判定优先级；
3. `tie` 与 `abstain` 的边界；
4. 一正一负一边界的简短示例，但示例不得来自当前评测状态；
5. 精确输出字段和最大长度；
6. 明确禁止依据：action、模型身份、memory 数量、回复长度、预期 regime；
7. AB/BA 顺序平衡；同一 judge 两顺序不一致时不产生硬标签。

质量 prompt 不得展示 hidden authorized context 或 selected evidence。证据审计 prompt 可以看
evidence，但不应同时比较总体支持质量。所有候选应以相同信息条件被比较。

## 6. 双裁判差异大时怎么处理

不能默认“二比一多数即真”。三个相关 LLM 可以稳定地一起错。采用岗位级、置信度感知合成：

| 结果 | 训练处理 |
|---|---|
| 同一 judge 的 AB/BA 一致，且两家合格 judge 同意 | 高权重弱标签 |
| 两家同意 tie | tie/小效应样本；不强造赢家 |
| 一家 abstain，另一家 AB/BA 一致且该岗位经人评校准 | 低权重单家标签 |
| 两家方向冲突 | soft target 0.5 / 高不确定性；不进入硬正负例 |
| 同一 regime/组件系统性冲突 | 测量故障；抽取少量人工 active-adjudication |
| 回复 treatment 没有 uptake | 数据/生成机制失败；停止 judge 调优 |

每个 judge 的可靠性必须按岗位分别估计，不能有一个全局“Claude/Gemini 准确率”。少量人评
用于：

- 校准不同岗位的错误率；
- 审查高影响冲突；
- 检验模型是否遵守 rubric；
- 不直接把一个研究者的 12 条判断包装成全量 gold。

当样本足够时，可用带 abstention 的弱监督标签模型估计潜在偏好；若 judge 数、人工 anchor
或条件独立性不足以辨识误差率，则保留 family-specific soft labels，而不是输出伪金标签。

## 7. PM 的训练目标

禁止回到“六个质量分 + 七个风险分合成一个标量，再对 16 动作取 max”。建议两阶段：

### 阶段 1：组件增量头

对 MP/MS/ME/RS 分别学习：

```text
P(component improves support | pre-action state)
P(component causes evidence/readiness risk | pre-action state)
uncertainty(component effect)
```

训练目标来自 matched pair。冲突/abstain 是低权重或无标签，不当负例。共享主干、强正则、
user/dialogue group CV；source×source 与 source×RS interaction 在证据不足时收缩到零。

### 阶段 2：受约束动作选择

在合法动作中选择：

```text
expected support gain
- applicable evidence/readiness risk
- deterministic cost
- uncertainty penalty
```

若没有组件超过预先设定的保守门，选择 M0+R0。cost 是选择器输入，不是让 LLM 裁判估计。

## 8. 训练前必须看到的实质证据

这不是“文件齐全”检查，而是可学习性检查：

1. 每个 MP/MS/ME/RS 至少有清晰正例、placebo/零效应和有害/不适用例；
2. clean paired treatment 的回复确实产生 uptake；
3. quality judge 在新鲜人工 anchor 上遵守判定优先级，且 AB/BA 不一致样本不会变硬标签；
4. evidence exact-quote 有效，omission 不再由“selected context 为空”机械推断；
5. 按 user/dialogue 留出的 train-only CV 同时优于 M0+R0 和同观测透明 rule；
6. learned policy 不是通过放宽风险阈值才获得非 M0 动作；
7. ESConv 形状下能判断 RS，memory-available 形状下能判断 source 与 RS；同一 checkpoint、
   不使用 domain ID；
8. EvoEmo/ES-MemEval-derived observable metadata 不再全量 OOD。

满足这些条件只能提高“下一次确实可学”的可信度，不能数学保证一定成功。若 clean treatment
仍无 uptake，或 group-held-out CV 仍不胜透明 rule，应停止 learned PM 主张，而不是再次扩大
调用规模。

## 9. 先纠正“理解用户需求”的方法身份

### 9.1 不是临床诊断模块

本研究需要的是 **Support-Need Assessment（支持需求评估）**，不是心理疾病诊断，也不是
推断用户未表达的隐私、人格或病因。它只对当前可见对话形成一组可撤回的交互假设：

- 用户现在更需要被倾听、共同探索、一个轻量建议，还是结构化计划；
- 用户在处理什么类型的目标，以及对话进行到什么阶段；
- 当前痛苦/紧迫程度是否允许继续普通支持流程；
- MP/MS/ME 或 Strategy Bank 中是否存在与当前目标匹配、可能产生增量价值的资源；
- 这些判断有多不确定，是否应保守 abstain。

任何危机/安全升级必须由独立、经过专门验证的 safety policy 处理，不能让小样本 PM 学习
“情绪崩溃时选哪个 RAG action”。PM 只处理普通 ESC 范围内的资源调度。

### 9.2 需求不是一个五分类答案

同一句“能给我一点建议吗”至少包含三个不同问题：

1. **readiness**：用户是否允许建议；
2. **goal**：用户要解决的是情绪梳理、信息获取、关系沟通、决策还是实际任务；
3. **resource opportunity**：当前 Strategy Bank 是否真的有与该 goal 和对话阶段相符的卡。

readiness 为 `light_suggestion` 不等于 RS 有益；用户请求帮助也可能只需要 generator 本身已
有的常识，或 Bank 根本没有合适内容。反过来，用户没有显式说“请给建议”，一个恰当的
reflection/restatement strategy 仍可能改善回复。因此 RS 标签必须来自
“readiness × Bank opportunity × matched response effect”，不能来自“是否出现 advice
request”。

## 10. 当前 BAAI→Step-0→PM 接口的真实审计

### 10.1 当前接口做了什么

当前冻结的 `BAAI/bge-small-en-v1.5` 是 33M 参数、384 维、512-token 上限的英文通用
embedding model。项目使用精确 revision、CLS pooling、L2 normalize，并产生：

- 完整 visible state embedding；
- current-user-turn embedding；
- MP/MS/ME query-to-source-centroid similarity；
- 8 个 Strategy family centroid similarity；
- 5 个手写 advice-readiness anchor centroid similarity。

`PMV2FeatureBuilder` 再把完整 state embedding 在 train users 上 PCA 到 48 维，并加入
word/char hash、Step-0 统计量，以及这些文本特征与 MP/MS/ME/RS action bits 的交互。PM
从未在决策前看到原始 memory/card、item ID、生成回复或 judge label；这一边界是正确的。

BAAI 官方模型卡把该模型定位为 retrieval/semantic-similarity embedding，并明确提醒：
v1.5 的相似度通常集中在约 `[0.6, 1]`，绝对分数大于 0.5 不代表语义相关，阈值必须在
具体任务分布上校准；cross-encoder reranker 通常比 bi-encoder 更准确但更慢。因此本研究
不能把 cosine 最大值解释为“模型理解了用户需求”，也不能用一个未校准的全局阈值决定
是否开 RS。

### 10.2 已经观测到的失败

现有 20 条 outcome-free natural-language readiness challenge 只有 `14/20` full-context
top-1 正确。6 个错误不是随机边角：

- 4 个明确接受“一条小建议”的状态被完整上下文稀释为 `ambiguous` 或 `listen_only`；
- “先聊、之后也许要想法”的状态被判成 `explore_first`；
- 修辞性 “What am I even supposed to do?” 被判成 `explore_first`，而 current-turn
  单独视图又偏 `structured_plan`。

多个错误的 top1-top2 margin 只有 `.0027–.0292`。这说明当前 nearest-centroid 机制既会
发生 context dilution，也没有把不确定性作为一等输出。它可以继续作为连续特征，但不能
继续被口头解释成可靠的需求理解。

### 10.3 V3 的接口改法

保留 BAAI snapshot 作为候选共享语义底座，但不再让它直接产生 readiness/action。新增一个
真正的、多视图 `SupportNeedObservation`：

```text
SupportNeedObservation
  visible_views
    current_user
    last_assistant
    recent_history
    session_summary
    current_vs_context_delta

  explicit_boundaries
    advice_rejected
    advice_requested
    one_small_step_requested
    listen_first_requested
    question_or_task_burden_limit

  support_mode_distribution
    listen
    explore
    comfort_reassure
    light_guidance
    structured_planning

  goal_distribution                 # multi-label
    be_heard
    make_sense
    stabilize
    decide
    act

  dialogue_phase_distribution
    exploration
    comforting
    action

  nonclinical_urgency_distribution
    routine
    elevated
    acute

  memory_source_opportunity[MP/MS/ME]
    available
    eligible_source_summary
    semantic_fit
    recency_compatibility
    expected_cost
    opportunity_probability
    uncertainty

  strategy_family_opportunity[family]
    eligible_card_count
    semantic_fit
    mode_phase_goal_compatibility
    directive_burden_fit
    opportunity_probability
    uncertainty

  uncertainty
    predictive_entropy
    group_bootstrap_dispersion
    semantic_OOD
    metadata_OOD
    abstain
    abstain_reasons
```

其中：

- BAAI 只提供表示，不提供 gold；
- current turn、last assistant turn、recent history 和 summary 分开编码/汇总，不能先拼成长
  段后只留一个向量；
- `ambiguous` 不再与真正支持模式并列。旧 `ADVICE_READINESS_IDS` 中缺失的
  `comfort_reassure` 成为真实 mode；不确定由 entropy/OOD/abstain 单独表达；
- support-mode/goal/phase/urgency 是强正则、低容量、可校准的 partial-label heads，不是
  nearest-anchor argmax；
- 显式请求/拒绝、问号、上一轮 assistant 是否提出方案等 observable cues 可以作为特征或
  弱标签函数，但不得直接写成最终 RS action；
- goal 与 mode 分开：例如用户目标可以是 `decide`，当前适合的 mode 仍可能是
  `explore`，而不是立刻 `structured_planning`；
- phase 是可回退、可混合的分布，不得只由 turn index 推导；
- nonclinical urgency 只控制互动负担与是否交给独立 safety policy；不得当作疾病诊断或
  memory/RS 正标签；
- memory opportunity 同时依赖可用性、source-level representation、age/cost 与需求分布，
  但仍不得读取具体 item、top item score 或 oracle；
- strategy opportunity 必须来自 Strategy Bank V2 的真实 eligible pool；不能继续使用
  raw 11,590-card 全库 family centroid 代替“可用卡”；
- 输出必须保留分布、margin 和 abstain，不能只留一个离散标签；
- safety escalation 独立于 PM，触发时绕过普通资源选择。

### 10.4 实现不是“让 BAAI 自己判断”，而是固定表示器上的小模型

V3 Phase 0 的候选实现固定为：

```text
frozen BAAI multi-view embeddings
  + deterministic visible cues
  + source/Bank aggregate metadata
        ↓
outcome-blind train-only low-rank projection
        ↓
regularized partial-label heads
        ↓
calibrated distributions + uncertainty + abstain
```

具体边界：

1. `prepare_visible_semantic_state` 扩展为明确的四视图输出；现有 current/full 两视图可复用，
   但单一 `text_embedding` 不再是全部接口；
2. BAAI 权重先保持冻结。降维只在 train-visible、outcome-blind 文本上拟合，并只在
   `{4, 8, 12}` 个共享维度中用 user/dialogue-group CV 选择；
3. 主候选只允许 L2/elastic-net logistic、ridge/ordinal heads 等低容量模型；旧高维 HGB
   不作第一候选；
4. 每个 head 的有效自由度必须受独立 train group 数约束，默认要求
   `learned feature dimensions <= floor(n_effective_groups / 5)`；不满足时减少维度或退回
   transparent rule；
5. 多任务共享只发生在低维表示层；mode、goal、phase、urgency、MP/MS/ME/RS opportunity
   各有独立输出、损失、分母和 uncertainty，禁止再压成一个 readiness/utility score；
6. 同一用户的原句、paraphrase、counterfactual 和多个 action 永远属于同一个 group，
   数据增强不能虚增样本量；
7. 不通过资格的 head 不靠其他 head 补票：保留为 `unsupported/abstain`，对应组件在
   正式 PM 中默认关闭。

### 10.5 SupportNeedObservation 的监督从哪里来

需求层不需要先生成 16-action 回复。监督按可信度分层：

| 来源 | 可监督内容 | 使用方式 |
|---|---|---|
| 显式用户边界 | listen/advice/light-step/plan 的方向，问题负担限制 | 高可信 partial label；只约束相应字段 |
| 预注册最小反事实 | support mode/goal 分布应向哪个方向移动 | paired consistency loss；同 group，不算两个独立样本 |
| 少量盲化人评 | mode、goal、phase、边界与 abstain | 校准和高杠杆边界 anchor；不是全量 gold |
| ESConv problem/emotion/experience | topic/affect/temporal context | 分层和 weak context；不是 mode/action gold |
| ESConv turn strategy/trajectory | exploration/comforting/action 的弱线索 | 只作 noisy phase/move label function；保留原始 strategy |
| Bank V2 eligibility | 哪些 family/card 在当前 mode/phase/goal 下有候选 | resource-opportunity partial label；不证明使用后有益 |
| clean matched response pairs | MP/MS/ME/RS 的实际增量方向 | 唯一可以训练 component effect 的结果层监督 |

旧 synthetic `advice_readiness_target`、旧 action pseudo-oracle、旧绝对 judge composite 和
ESConv gold strategy 都不得作为 `SupportNeedObservation` 的硬 target。需求层在人评/结构
anchor 上资格通过后，才允许把它的 out-of-fold prediction 提供给 PM；训练本身使用
cross-fitting，不能给一个 state 输入由包含该 state 标签训练出的 in-sample need prediction。

### 10.6 2026-07-27 第一实现检查点

已新增独立 V3 代码路径 `v1_5_support_need.py`，没有原地修改冻结的旧
`PMV2State/Step0Observation`：

- `SupportNeedObservation`、显式边界、MP/MS/ME opportunity、Strategy family opportunity
  与 uncertainty/abstain 均已有严格 schema；
- current user、last assistant、recent history、summary 四视图分别编码，并确定性构造
  current-vs-context delta；只保存内容/向量哈希和 encoder binding，不读取 item；
- partial-label 行保留 source family、role、reliability、order stability 与 abstain；
  abstain 行禁止夹带隐藏 target，人工/LLM/结构来源不得声明 automatic gold；
- categorical soft heads 使用 group-held-out nested CV、fold 内标准化/PCA/ridge、
  inner-group OOF temperature calibration 与 one-SE 最简选择；维度上限由独立 train
  groups 决定，低可靠度 group 不会被归一化放大；
- group ESS 先把重复 action/judge/alias 行聚回 user/dialogue group 后再计算。第一版测试
  曾真实抓到“行级 ESS 把 3 个重复 group 算成 6”的实现错误，修复后保留为反向测试；
- 零 API deterministic canary 两次逐字节一致：60 groups 已知信号 accuracy=`1.0`，
  label permutation accuracy=`.05`，五类 recall 均为 `1.0`，重复行不增加 group ESS。

`22i_run_support_need_train_pilot_v1_5.py` 已作为真实 train-only 入口，但当前尚未构建
完整的 Need/Bank partial labels，也尚未用真实冻结 BAAI 执行该 pilot。因此这是“代码能
学已知信号”的实现资格，不是“真实用户需要可学”的科学 PASS，更不授权 formal fit、
calibration、internal 或 external。

第一份真实输入包已经零 API 构建并独立复现：

- 只从 eligible ESConv train 选择 75 个未被旧 52-seed auxiliary 使用的对话，每个对话
  一个可见 state；validation/test、EvoEmo overlap 与未来 supporter target 均不读取；
- problem type 只用于 round-robin coverage；emotion 与 early/middle/late 只用于盲评
  anchor 分层，绝不进入盲包或在线 PM；
- 75 states 覆盖 13 个 problem、8 个 emotion，early/middle/late 各 25；
- 24 条人工 anchor 以 emotion×position 轮转，三个 position 各 8 并覆盖全部 emotion；
- 人工定义、顺序、abstain 和非临床 urgency 边界由
  `PM_V1_5_SUPPORT_NEED_HUMAN_ANNOTATION_GUIDE_ZH.md` 固定；人评仍只是可靠性/边界
  anchor，不是自动 action gold；
- packet 的合同与六个输出哈希由
  `data/pm_v1_5_contracts/support_need_human_packet_v1.json` 持久绑定；
- lineage 审计显示这 75 个 dialogue 全部已在 raw Strategy Bank，涉及 1,094 张卡。
  这是预期被发现的数据污染风险，不是可接受的训练状态：Strategy Bank V2 必须删除这
  75 个 source dialogue，并以 source/card overlap=`0` 通过机器检查，才可构造
  resource-opportunity partial labels。

第一批人工锚点现已完成并逐字节保存：24/24 exact coverage、1 条 abstain、23 个独立
dialogue 可用于诊断。标注语义审查发现旧
`question_or_task_burden_limit` 混合了“用户明确边界”和“标注者建议低负担”，因此：

- 原始人评不改写；normalized V2 同时保留完整原判断；
- production explicit boundary 只作为确定性 observable input；
- `human_recommended_low_interaction_burden` 只作有噪声 partial label；
- 新包中 explicit boundary 必须附可见 user 文本精确引文，response burden 单独三分类。

冻结 BAAI 的真实 train-only grouped OOF 结果没有证明五类 mode 已可正式学习：

| head | OOF prior loss | 最佳 view / loss | 关键事实 |
|---|---:|---:|---|
| support mode（5 类） | 1.7473 | lexical 1.6176 | accuracy 仅 .2323；只召回 comfort，另外 4 类 recall=0 |
| interaction posture（3 类，诊断） | 1.1221 | lexical 1.1007 | explore recall=0，不能替代五类结论 |
| dialogue phase | 1.1842 | lexical 1.1380 | exploration recall=0 |
| urgency | 1.1250 | lexical 1.1272 | log-loss 未优于先验；acute/routine 不能稳定区分 |

BAAI current/multiview 没有胜过透明 lexical/structure；这不证明 BAAI 无用。为把“small
encoder 不够”与“小样本/target 设计不成立”拆开，已在完全相同的 23 个 group 和 OOF
协议下用本机缓存的 `BAAI/bge-m3` 做零 API 复核：

- 五类 current-view log-loss `1.6321→1.6272`，差值 `-0.0060` 的 paired bootstrap
  95% CI 为 `[-0.0354,+0.0236]`；没有可靠优势；
- 五类仍然只稳定预测多数 `comfort_reassure`，`listen/explore/structured_planning`
  recall 仍为 0；
- 三类 dialogue phase 的 current-view log-loss `1.1537→1.0978`、accuracy
  约 `.38→.56`，说明更强 encoder 对粗粒度阶段有信号，但 exploration recall 仍为 0；
- 每个外折只有 18–19 个训练 group，强正则选择的有效维度通常只有 2；因此 384→1024
  维本身不可能弥补独立监督不足。

所以问题不能靠“直接换大 BAAI”解决。当前优先顺序改为：

1. 先把 flat 五类拆成可观察二元/层级轴（是否推进、是否提问、是否给建议、允许的任务
   负担、阶段），五类只作派生解释，不再作为唯一主 head；
2. 先在现有 23 条上做 encoder bakeoff。当前 BGE-small 和 BGE-M3 已完成；下一候选应是
   instruction-aware 的本地 embedding，但只能作为候选，不能因公开 benchmark 更高就
   自动替换生产 encoder；
3. 用 outcome-blind 未标注 train 文本拟合无监督投影/覆盖结构，用显式高精度边界产生
   partial positives；字段缺失不得当 negative；
4. 只有当某个具体稀缺轴仍无法辨识时，才恢复 fresh anchor 人评，而且新增条目必须能
   改变预先写明的决策。

原 75 个 outcome-blind train states 中另选的 24 条（16 expansion-fit、8 untouched
confirmation）已经准备。它们在下节资格赛完成前保持 **PAUSED / 不开放标注**，避免在
同一批标签上反复调方案。

### 10.7 2026-07-28 因子化表示资格赛与 Qwen/NLI 结果

已按上节预注册方向新增独立 `v1_5_support_need_bakeoff.py`，并在同一 23 个非 abstain
train dialogue group 上逐字节复跑。两份 1.1MB 报告完全一致，文件 SHA-256 都为
`51d75895…7bc4ee`；报告内部 SHA-256 `3421c4ad…d75ac1` 覆盖完整 source lineage，
binding 为
`data/pm_v1_5_contracts/support_need_factorized_bakeoff_v1.json`。

目标不再只用互斥五类，而拆为以下可重叠轴：

- `need_to_be_heard`、`need_for_emotional_containment`、`need_for_exploration`；
- `advance_readiness`、`focused_question_readiness`、`advice_readiness`；
- 稀缺诊断轴 `planning_readiness` 与 partial-only `low_interaction_burden`；
- 原五类 mode、phase、urgency 仍保留为 legacy/secondary diagnostics。

同一 grouped nested-OOF、fold 内 PCA/ridge、inner-OOF calibration、one-SE 容量门下比较
transparent observable、word/char lexical、BGE-small、BGE-M3、任务 instruction 固定的
`Qwen/Qwen3-Embedding-0.6B`、`cross-encoder/nli-deberta-v3-base` 假设分数及预声明
Qwen+NLI+observable hybrid。所有模型均绑定 exact revision/tree hash、local-files-only，
不读 ESConv metadata、未来 supporter、strategy、survey、internal 或 external outcome。

主要结果如下：

| 因子 | 类别数（neg/pos） | 最佳 view | loss | recall（neg/pos） |
|---|---:|---|---:|---:|
| need to be heard | 12/11 | BGE-M3 current | .6529 | .667/.455 |
| emotional containment | 10/13 | BGE-M3 current | .6782 | .300/.846 |
| exploration | 16/7 | NLI current+full+observable | .5483 | 1.000/.286 |
| advance readiness | 12/11 | NLI current | .6626 | .667/.545 |
| focused-question readiness | 19/4 | BGE-M3 current | .4647 | 1.000/.000 |
| advice readiness | 16/7 | observable structure | .6090 | 1.000/.000 |

表内“最佳”只按 confidence-weighted OOF soft log-loss 排序，不能掩盖 minority collapse。
另做非塌缩筛选后，focused-question 的 NLI current+full 为 `.4855`、recall
`.947/.250`；advice 的 NLI full-visible 为 `.6370`、recall `1.000/.143`。它们都以更高
loss 换回仅一个正例，不构成已过门的可靠轴。`planning_readiness` 只有 2 个正例，所有
view 的正类 recall 都为 0。

这回答了“是否应直接替换 BAAI”：

1. **Qwen3 不晋级。** 它没有成为任何主因子的最佳 view；在 advice readiness 上还稳定
   差于 lexical。focused-question 的平均 loss 虽改善且 bootstrap CI 略低于 0，但正类
   recall 仍为 0，不能用置信度改善掩盖少数类塌缩。
2. **BGE-M3 不是全局替代。** 它对 heard/containment/focused-question 有较好连续信号，
   但 focused-question 正类仍完全漏掉。
3. **NLI 适合作为轴特征，不是 gold。** 经低容量 OOF head 后，exploration 与 advance
   有非零双向 recall；focused-question/advice 也只有以更高 loss 换回一个正例的弱信号。
   固定零样本假设本身多数只在 chance 附近，说明必须由本任务人评校准，不能直接阈值化
   部署。
4. **因子化有实质价值但尚未过门。** heard/containment、exploration/advance 显示不同
   表示器的轴特异信号，但 focused-question、advice 与 planning 仍受稀缺正类和 loss/
   recall 取舍限制。预声明单一 Qwen+NLI hybrid 没有通过全部主因子检查，formal fit 与
   representation promotion 均保持 `false`。

因此下一步不再下载更大 encoder，也不打开全部 24 条扩展包。只开放预先冻结的 16 条
`expansion_fit` 人评，专门增加会改变 exploration/question/advice 判断的独立锚点；8 条
`untouched_confirmation` 继续不向标注页暴露。fit-only packet 已独立复现并绑定：

- `outputs/pm_v1_5_support_need_factorized_fit_packet_v1_candidate/`
- `data/pm_v1_5_contracts/support_need_factorized_fit_packet_v1.json`

所以 Phase 1 当前是“目标分解与 encoder/NLI bakeoff 完成；Qwen 不晋级；BGE-M3/NLI
保留为 axis-specific candidates；16 条 fit 人评待完成，8 条 confirmation 仍封存”，
仍然不是 Need learnability PASS。

### 10.8 16 条双人评合议、历史人评分流与 39-group 复核

两位评审已分别完成全部 16 条 `expansion_fit`。两份原始 JSONL 都通过 exact ID
coverage、abstain coherence、schema 和 user-only exact quote 校验。原始一致性为：
support mode 12/16（κ=.6701）、phase 13/16（κ=.7037）、urgency 12/16
（κ=.5188）、response burden 15/16（κ=.8873），goals exact-set 6/16。

合议不是多数投票，也不把两位评审当两个独立样本：

- scalar/goals 只能选 A 或 B 已提交的值；
- explicit boundary quote 只能从 A/B 证据并集中删选，不能新造；
- 特定选项不可行或拒绝某一种方法不能泛化成全局 advice rejection；
- 原始 A/B、逐字段分歧、理由和删去的 9 项过宽证据全部写入 trace；
- 16 个状态只增加 16 个 group。

绑定见
`data/pm_v1_5_contracts/support_need_fit_adjudication_v1.json`。与第一批合并后是 40 行、
39 个 non-abstain 独立 dialogue group；8 条 confirmation 未读取。

项目历史人评审计同时把任务角色彻底分开：

- 可直接进入 SupportNeed fit：第一批 24 条与本批 16 条合议行；
- 只能用于 judge：12 条 low-budget preference、12 条 role-decomposed preference/risk；
- 只能完成后用于各自任务：32 条 component-effect、5 张 Bank card、generation A/B；
  这些目前都是空模板；
- candidate/repro/normalized copy 不增加人评数或 ESS；
- 未来 V2 保留项目中的 generation/bank reviewer 文件也全是空模板。

完整清单见
`data/pm_v1_5_contracts/human_annotation_asset_audit_v1.json` 与
`docs/PM_V1_5_HUMAN_ANNOTATION_ASSET_AUDIT_ZH.md`。两组 judge packet 的 24 个
visible states 与当前 40 个 need states 零精确重叠，可以剥离 candidates、selected/
authorized context 和旧 preference 后重新做 outcome-blind need 标注；但它们是
response-difference-enriched active-learning pool，不用于类别 prevalence。

39-group bakeoff 使用与 23-group 资格赛相同的 exact model revision、local-files-only、
grouped nested OOF、fold 内投影/正则/校准和 paired bootstrap，并完成逐字节复跑。
报告文件 SHA256 为 `6cd7de38…c9030c`，内部 SHA256 为
`d9524618…67b4e`，binding 为
`data/pm_v1_5_contracts/support_need_factorized_bakeoff_expanded_v1.json`。

| 因子 | neg/pos | 最佳 view | loss | recall（neg/pos） | 相对 lexical 95% CI |
|---|---:|---|---:|---:|---:|
| need to be heard | 23/16 | NLI current | .6115 | .826/.500 | [-.150,-.014] |
| emotional containment | 20/19 | NLI current | .6527 | .350/.579 | [-.096,-.006] |
| exploration | 27/12 | BGE-small multiview+observable | .5736 | 1.000/.000 | [-.111,.005] |
| advance readiness | 17/22 | NLI current | .6574 | .529/.727 | [-.069,.001] |
| focused-question readiness | 32/7 | BGE-small current | .4638 | 1.000/.000 | [-.059,.007] |
| advice readiness | 24/15 | observable structure | .6874 | .875/.200 | [-.047,.010] |

`planning_readiness` 仍只有 3 个正例且最佳 view 正类 recall=0；
`low_interaction_burden` 只有 3 个负例且最低-loss hybrid 负类 recall=0。flat 五类
support mode 最佳仍是 observable，listen/explore/structured-planning recall 全为 0。
Qwen current 在 secondary urgency 上最低 loss，但 acute recall=0，不能据此晋级。

结论必须收窄：

1. Qwen embedding 仍不晋级，没有任何主因子把它选为最低 loss；
2. NLI 只在 heard/containment 上获得相对 lexical 的稳定 paired-loss 证据；它是轴特征，
   不是零样本 gold 或统一表示器；
3. 六个主因子中四个最佳 view 随 16 条新锚点改变，显示当前选模仍对小样本敏感；
4. 预声明 Qwen+NLI+observable hybrid 未通过全部主轴检查；
5. representation promotion、factorized formal fit、flat-mode fit 和 confirmation opening
   全部保持 `false`。

下一批只补 outcome-blind 的 exploration、focused-question、advice、planning 和
non-low-burden 边界。可以先利用上述 24-state 历史 judge 对话池做新盲评，再补真正能
增加 planning/多步骤负担反事实的 fresh states；不得用旧 judge preference 转标签，也
不得打开 8 条 confirmation 调方案。

### 10.9 主张不变，但 RAG treatment 与旧证据必须重置

V3 的抽象 estimand 仍然是：

> 在同一冻结 supporter、prompt、Bank、query builder、retriever/filter、decoding 和
> accounting 下，pre-retrieval PM 能否相对同栈 fixed/rule 在质量不劣时减少不必要资源。

因此因子化 SupportNeed 与 Bank V2 不要求改写论文的核心问题；它们改变的是解释变量和
operational treatment。由 raw 11,590-card Bank 改成五张 technique-only V2 card 后，
旧 generation outcomes、component labels、checkpoint、threshold 和 external freeze
不能继续作为 V3 证据。它们只保留为历史失败/机制诊断。

2026-07-28 的机器审计得到：

- 旧 sweep 确实含完整 response-mechanism contract，并在当时锁定 raw Bank、lexical
  retriever、top-k/floor、prompt、generator treatment 和 EF；当前
  `retrieval.py/prompts.py/action_execution.py` 等仍匹配，`api.py` 已与旧哈希不同，
  因而当前代码也不能冒充旧 run 的逐字复现；
- Bank V2 修订版已在
  `outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2_repro_check/` 六文件逐字节复现；
  旧 `outputs/pm_v1_5_strategy_bank_v2_review_repro_check/` 是首版 6,501-lineage
  产物，不是 V2 的 reproduction，目录名不构成身份证据；
- V2 仍为 `eligible_for_formal_rs=false`，五卡人评与 LLM 弱审计未完成；
- V2 technique-only schema 还没有进入 action execution、filter、prompt compiler、
  sweep、fixed baseline、ESConv 和 EvoEmo consumer；
- 新 V3 必须为 clean-pair generation、train sweep、transparent rule、fixed
  comparators、internal、ESConv、EvoEmo 七个岗位绑定完全相同的
  `response_mechanism_contract`。不同 state 检出不同 item 是合法的；Bank/corpus/query/
  retriever/settings 不同不是。

审计绑定为
`data/pm_v1_5_contracts/v3_claim_same_stack_audit_v1.json`，候选/复跑报告逐字节一致。
状态为 `BLOCKED_BEFORE_V3_RAG_TREATMENT_GENERATION`。这不表示本地研究停工；当前合法
工作是五卡审计、稀缺 need 边界补标、technique-only eligible-subset runtime 与七岗位
identity dry-run。它明确禁止 paid clean R0/RS generation、formal fit、confirmation/
internal/external opening，以及把历史 raw-RAG 数字写成 V3 方法结果。

历史 judge 的 24 个零重叠 visible states 已按上述边界重新制成盲包：

- 只保留 current user、recent dialogue、session summary；
- 旧 candidates、selected/authorized context、preference/risk labels 全部未复制；
- 新 blind ID 打断旧任务联结；candidate/repro 五文件逐字节相同；
- 只作 response-difference-enriched active learning，不允许估计类别 prevalence。

绑定为
`data/pm_v1_5_contracts/support_need_historical_reannotation_packet_v1.json`。完成新标注后
仍先重跑 grouped OOF 并检查稀缺轴；不会自动打开 formal fit 或八条 confirmation。

## 11. Strategy Bank 的内容审计与根修复

### 11.1 当前 Bank 实际是什么

对冻结文件 `data/strategy/strategy_cards_v1_5.jsonl`
（SHA256 `9cd614ad87677442feb6fb65ccbb4cecbba575867871474aede7700b0562a01e`）
的零 API 全量审计得到：

| 项目 | 真实值 |
|---|---:|
| cards / source dialogues | 11,590 / 823 |
| Question | 2,374 |
| Others | 2,127 |
| Providing Suggestions | 1,866 |
| Affirmation and Reassurance | 1,846 |
| Self-disclosure | 1,069 |
| Reflection of feelings | 901 |
| Information | 753 |
| Restatement or Paraphrasing | 654 |
| `guidance_text` 独立文本 | 8（每个 family 一个通用模板） |
| normalized duplicate `example_response` 所在行 | 1,089 |
| 最大 normalized duplicate group | 78（`hello`） |

每张卡的 `retrieval_text` 是原 ESConv 上下文，`example_response` 是原 supporter 回合，
`guidance_text` 只是“使用某 family、自然改写、不要假设未披露信息”这一类 8 个通用句。
它既包含有价值的倾听/反映/建议实例，也包含：

- 问候、结束语、问卷/平台元话语；
- case-specific 自我披露；
- 未验证的医疗、治疗、财务、法律、睡眠与生活建议；
- 粗糙、重复、语法不稳或推断过度的原回复。

因此当前 Bank 是“ESConv 回合案例库”，不是已经完成质量控制和适用性标注的策略知识库。
打开 RS 后，词法检索 top-3 可能拿到同题但不合当前 readiness/phase 的 raw examples；这正好
解释了为什么“请求建议”不能直接变成 RS 正标签。

首版 technique-only Bank V2 虽删除了 75 个 SupportNeed packet source，但其 lineage
仍含 130 条在任何 seeker turn 之前发生的 supporter opening。V2 修订版在所有内容/类别
过滤前先强制 `prior visible seeker turn exists`：共标记排除 413 张 raw opening card，
最终 technique lineage 从 6,501 降至 6,391；五张 technique 的
support_move/when-to-use/when-not-to-use/mode/phase/goal/burden/risk 文本逐字段不变。
旧五卡人评只能按 technique 内容精确迁移，不能沿用旧 card ID 或 lineage hash。

### 11.2 ESConv 原生字段可以利用，但不能把“数据集标签”偷换成“动作 gold”

对本地权威输入 `data/external/ESConv.json` 的零 API 全量检查表明，1,300 个 dialogue
逐条都有：

```text
problem_type
emotion_type
experience_type
situation
dialog[*].annotation.strategy
survey_score
```

本地 expanded release 实际含 13 个原始 `problem_type` 字符串、11 个原始
`emotion_type` 字符串和 2 个 `experience_type`。这里的“13/11”只是 raw string
基数，不等于 13/11 个干净本体类别：大小写、父母问题的近义命名和极小频数类别都真实存在。
当前 11,590 张正式卡的 `source_dialogue_id` 可以 100% 回连这些原生字段，但现有卡 schema
没有保存它们。按 `problem × emotion × strategy` 切分后有 532 个非空 cell，其中 208 个
少于 5 张卡、66 个只有 1 张，因此不能把稀疏组合硬编码成普适 action 规则。

各字段的合法用途必须固定如下：

| ESConv 原生字段 | 可以做什么 | 绝对不能做什么 |
|---|---|---|
| `problem_type` | Bank provenance；train-only 分层抽样；soft applicability；检查不同问题域的 coverage/偏差 | 直接决定 RS、直接当 PM action target；把部署时不可见的真值类别输入 PM |
| `emotion_type` | affect-context 分层；卡片适用性弱标签；分组误差与校准审计 | 冒充临床诊断；直接决定支持模式或资源动作 |
| `experience_type` | 区分当前/过去经历的时态与连续性风险；分层抽样 | 直接当 memory-needed 标签 |
| `situation` | 卡片来源语境与检索文本的一部分；离线质量审计 | 冒充当前用户显式陈述；当 support-need gold；复制进 external query |
| turn-level `strategy` | 描述 supporter 实际用了哪类支持 move；帮助归纳卡片 family | 证明该策略最优、证明 RS 有益、证明当前用户需要该策略 |
| `survey_score` | conversation-level source-quality prior、数据质量/敏感性报告 | 进入在线 PM 特征；摊到每个 turn 作 utility；当某张卡或某个 action 的因果效果 |

其中 `survey_score` 还存在真实缺失：initial intensity 为 1,300/1,300，seeker post-chat
字段为 1,150/1,300，supporter relevance 为 1,115/1,300。它既是 post-treatment，又是
conversation-level，最多用于“这段来源对话是否值得进入候选归纳”的弱权重和敏感性分析。

官方 ESConv 论文把 support strategy 定义为八类 supporter move，并把情绪/问题/反馈作为
不同层次的 annotation；expanded release 的 README 也明确把新 problem topics 用于
topic transfer。V3 因而保留这些维度，但不把它们混成一个“用户需要 RS”的答案键。
运行时若需要 problem/emotion context，只能由可见对话输出带不确定性的分布；offline
原生标签只用于训练分层、Bank 构建和误差分析。

仓库另有 `strategy_cards_v13.jsonl`（156 张），其 metadata 已包含
`support_state/problem_type/emotion_type/advice_request/support_goal`，可作为 schema
设计参考；但其 provenance 指向的 `data/esconv_pm_v32/pm_train.jsonl` 和
`scripts/build_strategy_cards_v13.py` 已不在当前工作区，无法从原始输入重建。因此不得
直接把 V13 当作可信 Bank V2，也不得把项目派生的 `support_state` 误称为 ESConv 官方字段。

### 11.3 V3 不直接沿用 raw top-3 surface

保留 raw bank 作为 provenance 和对照，但建立 `Strategy Bank V2`：

```text
card_id
strategy_family
support_move                 # 要做的对话动作，不是原句照抄
when_to_use
when_not_to_use
compatible_support_modes
compatible_dialogue_phases
goal_types
native_problem_types         # raw+normalized；soft compatibility，不是 action gold
native_emotion_types         # raw+normalized；soft compatibility，不是临床标签
experience_types
directive_burden             # none / light / structured
content_scope                # technique_only / domain_information
risk_flags
source_provenance
source_quality_observations  # conversation-level；永不进入在线 PM
quality_status
```

构建原则：

1. 先过滤 greeting/closing/survey/meta、逐字/近重复和明显不安全的内容；
2. `technique_only` 与 `domain_information` 分库。第一篇正式 RS 默认只允许
   technique-only；具体医疗/法律/财务建议不能靠 raw ESConv card 进入 prompt；
3. raw example 只可用于离线归纳 `support_move/when_to_use/when_not_to_use`，正式 prompt
   不默认暴露原用户案例；
4. Self-disclosure 默认高门槛，只有 readiness/phase 明确适配且不抢占用户叙事时才候选；
5. 每张卡必须经过规则审计、LLM 弱审计和小型人评抽检；不合格卡不进入正式候选池；
6. family、goal、phase、burden 标签是部分/弱标签，要保留来源和不确定性；
7. `problem/emotion/experience` 先保留 raw value，再做显式 normalization；稀疏 cell
   只能参与 soft scoring/分层，不能作为 hard-exclusive filter；
8. phase/readiness 必须来自当前可见 dialogue trajectory，不得只用 turn index 或
   source card 的静态类别；
9. 每张卡都记录 `native_metadata_source=official_esconv` 或
   `derived_metadata_source=<rule/model/human>`，禁止把派生字段伪装成官方 annotation。

### 11.3.1 第一份零 API V2 candidate

确定性 builder 已在首批 75-state Need packet 上真实运行并独立复现：

- raw Bank 11,590 cards / 823 sources；
- 75 个 packet source 对应的 1,094 张卡全部排除；
- 234 个 normalized duplicate groups 涉及 1,089 行；冗余行、112 条纯问候、66 条纯
  结束语、47 条平台/问卷元话语与 43 条具体 domain claim 均在 lineage 审计中计数；
- 第一篇候选只保留五个 technique family，将 748 个不重合 source / 6,501 条
  rule-filtered lineage 聚成 5 张透明卡；
- raw `example_response` 不进入卡片或 generator surface；卡片只暴露人工编写且可逐项
  审查的 `support_move/when_to_use/when_not_to_use/mode/phase/goal/burden/risk`；
- `Information`、`Others`、`Self-disclosure` 和 domain-specific information 当前全部
  隔离，不用类别数量换覆盖表象；
- 5 张卡的 `eligible_for_formal_rs` 固定为 false，只有五卡人评全部通过且独立的小额
  LLM 弱审计通过后，才可创建 train-only pilot 专用 promotion；仍不等于 formal Bank。

受追踪 binding 为
`data/pm_v1_5_contracts/strategy_bank_v2_candidate_v1.json`。人评界面和模板位于 binding
记录的 output directory，验证器为
`22m_validate_strategy_bank_v2_human_annotations_v1_5.py`。

### 11.4 检索与 PM 的职责边界

V3 采用：

```text
pre-retrieval:
  SupportNeedObservation + family-level Bank opportunity
  -> PM chooses memory subset and R0/RS

if RS:
  lexical candidate generation within compatible family/phase subset
  -> deterministic eligibility filters
  -> optional train-only-qualified reranker
  -> zero acceptable card => realized R0 alias
```

PM 仍不选择具体卡；但 PM 看到的“RS opportunity”必须与下游实际可取的 eligible pool 同源。
不得继续让 PM 根据 8 个全库 centroid 预测有用，而实际 retriever 在未经清洗的 11,590 张卡
里全局 top-3。此前 Hybrid 在合法 calibration/ESConv validation 上未胜 lexical-only，
所以 V3 不把 generic Hybrid 重新包装成解决方案；任何 reranker 必须在新的 need-aware、
clean-bank retrieval task 上单独验证，lexical 仍是必要 baseline。

## 12. 统一 PM 的决策分解

ESConv 与 EvoEmo/ES-MemEval-derived 不是两个 PM，也不是先“学会 RS”再另学 memory。
对每个状态，统一决策为：

```text
1. 当前支持模式与目标是什么？
2. 哪些资源在结构上可用？
3. 对每个 MP/MS/ME/RS，当前是否存在真实 opportunity？
4. 加入该组件是否可能改善回复？
5. 会不会带来错误个性化、过度结构化、多源干扰或成本？
6. 证据不足时是否应 abstain？
```

组件头学习：

```text
P(Δsupport_MP > 0 | state, MP opportunity)
P(Δsupport_MS > 0 | state, MS opportunity)
P(Δsupport_ME > 0 | state, ME opportunity)
P(Δsupport_RS > 0 | state, strategy opportunity)
P(component-specific risk | state, component)
uncertainty(component effect)
```

ESConv 是 `MP/MS/ME unavailable` 的合法区域，只训练 RS effect；memory-available 数据同时
训练 source effect、RS effect 和必要的 source×RS interaction。外部仍分别报告，因为两个
数据集的 estimand 和独立单位不同；内部模型、feature builder、需求层和 selector 必须相同。

## 13. 新训练数据：先有需求结构，再有资源 effect

### 13.1 因子矩阵

新 state 不是按“最终正确 action”生成，而按以下可观察因子做平衡/部分因子设计：

| 因子 | 建议水平 |
|---|---|
| support mode | listen / explore / comfort-reassure / light guidance / structured planning |
| goal type | be heard / make sense / stabilize / decide / act |
| dialogue phase | exploration / comforting / action |
| nonclinical urgency | routine / elevated / acute（危机安全升级另行绕过） |
| memory opportunity | unavailable / helpful / placebo / stale-conflict |
| strategy opportunity | unavailable / matched / mismatched / unsafe-or-overburdening |
| explicitness | explicit request / implicit contextual cue / counterfactual negation |
| catalog shape | small/large count、短/长 age span、不同 token envelope |

不做全笛卡尔积。用覆盖数组/分层抽样保证每个单组件正、零、有害效应都有足够的
user/dialogue groups，同时避免模板直接编码 action。

### 13.2 三类标签不能混成一个

1. **support-need partial labels**：来自明确可观察边界、对话结构、少量人工和独立 LLM
   弱标；用于需求 heads，不是 action gold。
2. **resource opportunity labels**：来自清洁 Bank/memory 的结构与相关性审计；用于限定
   “有没有可用资源”，不代表使用后一定更好。
3. **realized component-effect labels**：来自同 state、同 prompt/model/seed 的 matched
   回复对和拆职裁判；这是 PM 组件增量头的核心监督。

只有第三类能决定“应不应该开”。“用户准备接受建议”与“Bank 有匹配卡”只构成必要条件或
特征，不能单独成为 RS positive。

### 13.3 防 shortcut 数据设计

- explicit、implicit、negated、context-dependent wording 都要覆盖；
- 同一个 support mode 使用多个 semantic family，同一 family 覆盖多个 support mode；
- helpful 与 placebo 可以同题，只在是否包含个人相关、可产生边际价值的信息上不同；
- `current_user_text` 不写 MP/MS/ME/RS、case/regime nonce；
- 长短 history、summary present/absent、inventory count/age 与 target 交叉平衡；
- paraphrase invariance 与最小反事实 sensitivity 成为生成后的零 API门。

## 14. BAAI 与需求层的资格验证

在生成大规模 response 前，先只用 train-only 状态做以下比较：

| 候选 | 作用 |
|---|---|
| nearest 5-anchor centroid | 当前 baseline |
| word/char + observable cues | 无 BAAI baseline |
| current-turn BAAI | 检查短句直接表达 |
| full-context BAAI | 检查现有方案 |
| separated-view BAAI + low-capacity heads | V3 主候选 |
| optional commonsense hypothesis features | 只作消融，绝不作 gold |

按 user/dialogue group CV 报告：

- support-mode/goal/phase 的 macro-F1、balanced accuracy；
- Brier/ECE 与 selective coverage-risk；
- 每类 recall，禁止靠多数 `listen` 或高 abstention 过门；
- explicit→paraphrase invariance；
- negation、assistant-question-conditioned short answer、rhetorical question的最小反事实方向；
- train→ESConv-shape/EvoEmo-shape 的 OOD，不读 external outcome；
- PM downstream component-effect CV 相对无 BAAI baseline 的增益。

V3 的继续条件不是一个随意的 top-1 数字，而是：

1. separated-view candidate 在 user-group CV 稳定优于 nearest-centroid 和 lexical baseline；
2. 五类均有非零、可复现辨识，且高不确定样本能 abstain；
3. 关键 negation/context-dependent challenge 不出现系统性方向反转；
4. BAAI 加入后不能形成 synthetic target shortcut；
5. downstream component-effect learnability 有增益；若没有，BAAI 只保留为诊断特征。

如果 `bge-small-en-v1.5` 不满足上述条件，应在 **新 response 生成前** 比较同系列
base/large、经过任务数据对比学习的轻量 adapter 或 clean-bank reranker；一旦正式 response
distribution 生成并冻结，不得看结果后更换 encoder。COSMIC/COMET-ATOMIC 能产生
speaker intent/effect/reaction 的 commonsense hypotheses，IntentionESC 也强调
emotion-state→intention→strategy，但这些生成式推理只可作附加不确定特征，不能替代本任务
人工/结构 anchor，更不能输出临床诊断。

## 15. 裁判、prompt 与双家族分歧

第 4–6 节的拆职规则继续生效，并增加以下要求：

### 15.1 quality 岗只比较“回复是否适合当前需要”

rubric 固定顺序：

```text
explicit boundary
> factual/dialogue continuity
> emotional attunement
> interaction burden
> goal/readiness fit
> optional helpful depth
> style
```

quality judge 不看需求标签、Bank card、action、regime 或 hidden context。它只能从 visible
dialogue 判断两条匿名回复谁更适合。这样避免裁判因为“标签说 structured plan”而给计划型
回复加分。

### 15.2 evidence/strategy 岗检查机制，不代替 quality

- evidence 岗逐 claim 给 exact response/evidence excerpts；
- strategy 岗判断推进程度与用户边界是否匹配；
- deterministic 程序检查引用、人物/时间/数字、实际 prompt surface 和 cost；
- omission 通过 matched counterfactual effect 判断，不从“没选择资源”推导；
- 所有岗位都有 `tie/abstain/insufficient/N/A`。

### 15.3 分歧是数据

双裁判差异不再通过不停换模型、补第三家或调门槛“消掉”。每个岗位保留：

```text
family-specific raw decision
AB/BA stability
human-anchor calibration
agreement/disagreement/abstention
estimated label uncertainty
```

只有岗位内、顺序稳定、在人评 anchor 上满足最低能力的 judge 才参与该岗位。两家冲突时：

- treatment 本身 uptake 弱：归因到数据/生成机制；
- rubric 应用点不同：修 rubric 后必须换新 packet，不在旧 packet 上调到过；
- 各自稳定但价值取舍不同：保留 soft target/uncertainty；
- 少数高影响冲突：人工 active adjudication。

第三家可以提高覆盖和帮助定位 family bias，但不自动拥有多数真理。人评量应集中在新鲜
边界样本、系统性分歧和高杠杆 action，而不是把全部语料改成人工标注。

## 16. 透明、可观察、可复算的指标体系

### 16.1 “好/坏”必须落到可观察事件，禁止用一个含混总分遮蔽机制

本研究不再把一个未经定义的 `overall quality`、若干 Likert 维度的平均值或
`quality - risk - cost` 当作论文主指标。任何结果表中的指标都必须在计算前给出完整字典：

```text
metric_id
decision_question           # 这个数字回答什么问题
unit                        # state / response / claim / retrieved item / dialogue / user
eligible_denominator        # 哪些样本可进入分母
numerator_or_formula
source_fields
direction                   # 越高越好 / 越低越好 / 仅诊断
N/A / tie / abstain rules
aggregation                 # 先 state，还是先 dialogue/user
uncertainty                 # dialogue/user-cluster CI
threshold_origin            # 人工 anchor / 设计约束 / noninferiority；不能看结果后补
failure_action
prohibited_interpretation
```

没有分母、N/A 规则和失败后的动作，就不算一个定义完整的指标。所有报告必须同时给出原始
计数、分母、比例和 cluster interval；不能只写 `PASS`，也不能用宏平均掩盖某个来源、
support mode、problem/emotion group 或 judge family 的失败。

内部 action selector 可以为了做选择而使用预先冻结的约束式目标，例如

```text
predicted support gain
- λ_r * predicted applicable risk
- λ_c * deterministic resource cost
- λ_u * uncertainty
```

但这只是选择器内部规则，不是 evaluation score。四项必须分别报告，`λ` 必须在
calibration/internal/external 前固定并做敏感性分析；不得把内部 scalar 包装成“总体回复
质量”。

### 16.2 需求理解层：测“是否读懂当前需要”，不测“是否猜中了动作”

| metric | 可观察定义 | 好/坏方向 |
|---|---|---|
| explicit-boundary violation rate | 用户明确说“先听/不要建议/给一个小步骤”等，need head 或最终回复违反该边界的 state 数 / 有明确边界的 state 数 | 越低越好；这是首要 guardrail |
| support-mode macro F1 / balanced accuracy | 在结构 anchor + 少量盲化人评上，对 listen/explore/reassure/light-guidance/structured-planning 的 group-held-out 预测 | 越高越好；逐类报告，不用 micro accuracy 掩盖小类 |
| goal/phase Brier score | 预测分布与 partial/weak target 的平方误差；missing target 不进分母 | 越低越好；不得把弱标签当绝对真值 |
| selective risk at coverage | 在模型不 abstain 的样本里预测错误率；同时报告 coverage 与 abstain rate | 同一 coverage 下越低越好；不能靠全 abstain 获胜 |
| counterfactual direction accuracy | 只改变“想被听/想建议/拒绝建议/上一轮已问过”等最小线索时，预测分布是否按预注册方向移动 | 越高越好；系统性反向直接停机 |
| paraphrase invariance | 意义相同、词面不同的输入是否保持同一需求分布容差 | 越高越好；不是强求向量逐字相同 |

BAAI 的 cosine、centroid margin 或 embedding distance 只属于 representation diagnostics，
不是“理解用户”的主指标。只有它提升上述 held-out observable metrics 和后续 component
effect 学习，才可进入 PM。

### 16.3 Strategy Bank 与检索层：测“有没有合适资源、取到的是否合适”

| metric | 可观察定义 | 好/坏方向 |
|---|---|---|
| eligible-card coverage | 至少有一张通过 support-mode/phase/goal/burden/risk eligibility 的卡的 state 数 / 所有 RS-eligible states | 越高越好；同时按 problem/emotion/experience/family 分层 |
| audited applicable precision@k | top-k 中经规则+盲化抽检判为适配当前边界/目标的卡数 / 被检索卡数 | 越高越好；零卡时记 N/A 并另报 zero-hit |
| harmful-or-irrelevant exposure@k | top-k 中 harmful/placebo/meta/domain-information 越权卡数 / 被检索卡数 | 越低越好；不是与 precision 合成一分 |
| exact/near-duplicate exposure | 被返回卡中属于重复组的卡 / 被返回卡；另报最大组和 unique surface | 越低越好 |
| cell support | 每个 problem×emotion×phase×family 的来源 dialogue 数、卡数和有效抽检数 | 仅诊断；小 cell 不能硬得出优劣 |
| requested→realized RS rate | requested RS 后实际至少一张 eligible 卡进入 prompt 的 state 数 / requested RS states | 描述机制；不能把低值自动说成 PM 差 |
| zero-hit / R0-alias rate | requested RS 但无合格卡、prompt 等价于 R0 的比例 | 越低通常越好，但必须与 abstention 安全性共同解释 |
| card uptake | 回复中可逐字定位、且能由所选卡支持的 strategy move / 选中卡；同时报告未使用与误用 | helpful 组希望有 uptake，placebo/harmful 组希望无 uptake |

`problem_type/emotion_type` 用于 coverage、偏差和 soft compatibility；不能因为某类别历史上常
伴随某 strategy，就把该 strategy 视为当前正确答案。官方 turn strategy 只证明“曾被使用”，
不证明“该卡在这个新 state 有效”。

### 16.4 回复与裁判层：优先判具体错误和盲化偏好

| metric | 可观察定义 | 好/坏方向 |
|---|---|---|
| blind support preference | matched A/B 的 win/loss/tie/abstain 原始计数；quality judge 只见 visible dialogue 与匿名回复 | treatment 的 win rate 提升且 CI 支持；tie/abstain 不强转 0/1 |
| continuity error rate | 回复重复已知问题、混淆人物/时间/事件或声称对话未提供事实的 response 数 / responses | 越低越好 |
| interaction burden | 每条回复的问题数、并列任务数、指令步骤数；与用户 readiness/boundary 联合分层 | 不是越低越好；只判“是否超过当前用户可接受负担” |
| personalized-claim support precision | 回复中所有可识别个性化/记忆依赖 claim 中，有逐字可定位 evidence 支持的数量 / claim 总数 | 越高越好；无 claim 单列 `no_personalized_claim` |
| unsupported/stale/exposure violations | 每个 action-applicable 风险事件的 count / eligible responses，并报告 0–3 severity 分布 | 越低越好；N/A 不编码为零 |
| source omission effect | 同 state 的 control 与 clean helpful-source treatment 的盲化质量差；不是“没选资源”这个行为本身 | 只有 treatment 有真实边际增益时才可称 omission |
| strategy over/under-structuring | 当前明确边界/phase 下，回复推进强度是否过高或过低的事件率 | 越低越好；必须引用回复中的具体 move |

judge 的质量也用可观察指标检查：schema completion、exact-excerpt validity、AB/BA consistency、
tie/abstain rate、与少量人评 anchor 的岗位内一致率、跨 family 的方向表。任何一项不通过都
只能降低权重、abstain 或停止该岗位，不能通过“第三家多数票”改写成 gold。

### 16.5 PM 学习与最终主张：看 held-out frontier，不看动作是否“丰富”

PM 的 primary decision metrics 固定为三组并列量，不合并：

1. **support noninferiority / gain**：在 user/dialogue-held-out matched contrasts 上，
   learned PM 相对 `M0+R0`、transparent rule 和同成本 fixed comparator 的 blind support
   preference 差与 cluster CI；
2. **risk guardrail**：同一批响应的 boundary、continuity、unsupported/stale/exposure、
   over-structuring 事件率差；任一关键风险恶化都不能被平均质量提升抵消；
3. **resource cost**：实际 retrieval attempts、进入 prompt 的 memory/strategy items、
   input/output tokens、USD 和 latency 分开报告。

component head 的 Brier/log loss、group-CV、calibration curve 和 effect-sign accuracy是模型
学习诊断；action distribution、RS 选择率、memory 选择率只是 collapse/coverage 诊断，不是
成功指标。PM 选择多样不代表学会；PM 大量选择 R0 也不自动代表失败，必须看它是否在有边际
价值的 held-out state 选择资源、在无价值/有害 state abstain，并在质量—风险—成本 frontier
上胜过预注册 baseline。

ESConv、EvoEmo 与 ES-MemEval-derived 各自报告同一套原子指标，但不能把不同 estimand
池化成一个总体分。ESConv 主要观测 memory unavailable 背景下的 RS 增益/负担；
EvoEmo/ES-MemEval-derived 观测 memory available 背景下的来源选择、连续性、干扰、风险和
成本。跨域主张只在相同指标方向均成立时写“迁移”；一个域的提升不能替另一个域补票。

### 16.6 在任何新付费数据前必须产出的 metric registry

Phase 0 必须把上述指标变成受追踪、机器可读的 registry，并为每个 metric 给一条手算
fixture 和一条反向 fixture。registry 的任务不是再造工程仪式，而是让研究者、代码和论文
对“分子、分母、N/A、方向、阈值、失败动作”说同一件事。任何 runner 只输出一个笼统
`quality_score`、`utility` 或 `PASS` 而不给这些原子字段，均不得进入训练 readiness 或论文
结果表。

## 17. 小样本弱监督怎样训练才有效

### 17.1 独立样本不是 action 行，而是用户/对话 group

训练与报告的最小独立单位固定为：

```text
domain → dialogue/user → state → component contrast → prompt alias
```

同一 state 的 16 个 actions、同一回复的 alias、AB/BA、paraphrase、counterfactual 和 judge
family 只增加重复测量，不增加独立样本数。所有表必须同时报告：

```text
raw rows
unique dialogue/user groups
unique states
unique physical responses
non-missing weighted observations
ESS = (Σw)^2 / Σ(w^2)
```

最低可拟合边界在 outcome 前固定：

- 每个真实 support mode 至少 10 个不同 dialogue/user group，且 weighted ESS 至少 8；
- MP/MS/ME/RS 每个 main-effect head 的 helpful 与 non-helpful/harmful 方向各至少 8 个
  独立 group；placebo 单独保留，不能并入 helpful；
- 任一 source×source 或 source×RS interaction 只有在相关 main effects 已过门、且该
  interaction cell 至少 12 个独立 group 时才允许打开；
- 不满足最低数目的类/组件不重采样到“看起来够数”，而是合并到合理父类、保持
  `unsupported` 或暂不进入正式 action space。

这些是“是否值得拟合”的最低边界，不是统计功效保证。paraphrase/生成增强可改善鲁棒性，
但在权重中仍只算原 group 的一份证据。

### 17.2 弱监督不先压成 pseudo-gold

每个 label function/judge family 输出自己的 distribution、abstain 和 provenance。合成遵循：

1. 显式边界和预注册最小反事实是高可信 partial constraints，但只约束对应字段；
2. 人工 anchor 只校准岗位/类别可靠性，不自动成为全量 gold；
3. judge 可靠性按 `family × role × stratum` 估计，并用 Beta prior 向 0.5 收缩；小样本不允许
   得出 0 或 1 的绝对可靠性；
4. 同一 judge AB/BA 不一致时，该 pair 对该岗位 abstain；
5. 同 provider、同 prompt family 的多个输出共享一个总权重上限，不能冒充独立多数；
6. 两家稳定同向时形成高权重 soft target；一家稳定、另一家 abstain 时降低权重；方向冲突
   保持接近 0.5/高 uncertainty 或 missing，不硬投票；
7. treatment 没有 uptake、引用不完整或 action 不适用时，结果层权重为零/N/A，而不是
   “无风险、无效果”的零分。

每个训练观察的权重拆开保存：

```text
w = source_reliability
  × order_stability
  × evidence_completeness
  × treatment_uptake_validity
  × group/alias weight
```

各因子和最终 ESS 分别报告。不得只保留一个不可解释的 `label_reliable=true/false`。

### 17.3 两层小样本，而不是一步训练完整 PM

**层 A：需求与 opportunity（不需要付费生成回复）**

- 先构建约 60–100 个 train-only、user/dialogue-disjoint states；
- 五个 support modes 每类约 12–20 个独立 group，并覆盖显式边界、paraphrase、negation、
  assistant-context-dependent 和 rhetorical-question cases；
- problem/emotion/experience 只用于分层；不把 ESConv strategy 或旧 regime 当 target；
- 用结构标签、少量人评和 Bank eligibility 训练
  `SupportNeedObservation` partial-label heads；
- 只产生 out-of-fold observation，禁止 in-sample need prediction 泄漏给 PM。

**层 B：clean component effects（需要小规模生成与拆职测量）**

- 从层 A 已覆盖且 Bank eligibility 明确的 states 中选择约 48–80 个做最小 pilot；
- 每次只操纵 MP、MS、ME 或 RS 中一个组件；helpful/placebo/harmful 分开；
- helpful treatment 必须先通过 selected-item composition 和 generator uptake；
- matched pair 才进入 quality/evidence/risk 岗位；
- 单组件 main effect 可学后，才按最小 ESS 追加交互，不一次生成 all-three 矩阵。

层 A 失败时不花生成/judge 费用；层 B uptake 失败时不继续换 judge；measurement 失败时不把
不可靠标签送入 PM。这把过去所有失败拆到能被单独定位的层。

### 17.4 模型容量由独立 group 数决定

正式主候选不再从 512+ hash、48 PCA、Step-0 和大量 interaction 中直接训练 HGB：

- need heads：固定 BAAI + `{4,8,12}` 维共享 train-only projection + 少量 deterministic
  cues + 强正则 linear/ordinal heads；
- component heads：MP/MS/ME/RS 共享低容量主干，每个组件只有 main effect residual；
- risk heads：只在 action-applicable 样本拟合，结构性 N/A 权重为零；
- uncertainty：group bootstrap/ensemble dispersion + predictive entropy + OOD；
- interaction：默认精确为零，只有第 17.1 节的 ESS 与 main-effect gate 同时满足才解冻；
- candidate 选择：nested/group CV 的 one-standard-error rule，性能相近时永远选更简单者。

任何 BAAI adapter、非线性深层模型、RL 或大规模 self-training 都不进入第一候选。只有固定
BAAI 的低容量 heads 在 train-held-out groups 明确不足、且已有足够新鲜 anchor 时，才可在
新 response 生成前比较轻量 adapter。

### 17.5 Train-only 可学习性证明

正式 fit 前必须用完全 out-of-fold prediction 完成：

0. **实现层 canary**：在不含真实 outcome 的可控小型 fixture 上，标签方向、权重、N/A、
   group split、cross-fitting 和 selector 均能恢复已知 MP/MS/ME/RS main effect；打乱标签
   后回到 chance，复制 action/judge/alias 行不会改变 ESS 或模型选择。该项失败说明代码
   不能学，不得拿真实数据继续试；
1. **需求层**：separated-view heads 相对 majority、lexical 和旧 nearest-centroid，在
   macro F1/Brier/selective-risk 上满足 one-SE 优势；无真实 mode collapse；关键最小
   反事实不系统反向；
2. **Bank/opportunity 层**：eligible coverage、audited precision@k、harm/placebo exposure
   和 zero-hit 分项成立；PM-visible opportunity 与真实 eligible pool 同源；
3. **生成层**：helpful/placebo/harmful 的 uptake 方向可区分，效果不只由长度解释；
4. **测量层**：岗位内 AB/BA、exact excerpts、anchor calibration 和 abstain 合格；未合格
   岗位保持 missing；
5. **effect 层**：component soft-effect heads 在 user/dialogue-group held-out loss 和
   effect direction 上优于 constant-zero、M0-only 与同观测 transparent rule；
6. **policy 层**：out-of-fold selector 在至少一个预注册 RS cell 和一个 memory-source cell
   对 clean positive challenge 选择相应组件，在 placebo/harmful cell abstain；不是靠放宽
   risk threshold 制造非 M0；
7. **稳健性**：label permutation 回到 chance；BAAI removal、Step-0 removal、need-head
   removal、Bank-eligibility removal 的影响可解释；external-shape severe OOD 过门。

如果只有部分组件通过，第一篇可以训练**受限 action mask 的同一个 PM**：未获支持的
source/interaction 永久保持 off，并把主张限定到已通过的条件。只有 MP/MS/ME/RS 四个
main effects 都通过时，才允许声称完整 source-level router；不能为了保住“16 actions”
让随机 head 进入部署。

### 17.6 防止再次出现 100% M0+R0

下一次正式训练前必须先通过一个独立的 out-of-fold viability report：

```text
high-confidence helpful challenge selected correctly
placebo/harmful challenge rejected
nonfallback coverage
action entropy（仅诊断）
largest-action share（仅诊断）
raw predicted effect by component
risk feasibility by applicable dimension
cost before/after selection
```

动作多样性本身不是成功标准；但如果 clean positive challenges 已证明有增量，而 PM 仍对
全部 held-out states 选择 `M0+R0`，就是学习失败，不能进入 calibration。相反，如果当前
样本确实没有可辨识增量，全部 M0 可以是诚实结果，但此时 learned-router 主张为
`NOT_SUPPORTED`，而不是“保守 PM 已成功”。

这套顺序不能数学保证 external 一定给出正结果。它能保证的是：formal training 只会发生在
需求可辨识、Bank 有合格资源、回复能吸收 treatment、裁判能测、低容量 component heads
已经在未见 train groups 学到信号之后；不会再把已知不可学的数据直接送进 calibration，
最后才发现 collapse。

### 17.7 本计划中“下一次必须能学”的可检验含义

“能学”不能再等价于“训练脚本退出码为零”，也不能等价于“动作看起来多样”。在首次
formal fit 前，下列事实必须已经由 train-only、out-of-fold 结果同时成立：

1. 每个准备打开的 need/component head 都有达到最低 group/ESS 的正、零/有害方向；
2. 人工/结构/LLM 弱监督的冲突率、abstain、来源权重和有效 ESS 已知，没有隐藏的伪 gold；
3. 受控 canary 能恢复已知效应，真实 clean contrasts 上也胜 constant-zero 与透明规则；
4. clean helpful state 上的预测效应为正，placebo/harmful state 上不会被同样推高；
5. selector 的非 M0 选择来自已验证的正效应，不是人为放宽 risk/OOD threshold；
6. 尚无数据支持的 source 或 interaction 不训练、不部署、不进入主张。

若这六项成立，代码与数据具备了“在现有小样本中可学习”的实证前提；若不成立，计划会在
Need、Bank、uptake、measurement 或 effect 的具体层提前停止并点名缺口。它不承诺未知
internal/external 一定有利，但消除了过去那种“先昂贵训练，最后才发现标签、机制或样本
根本不可学”的失败路径。

## 18. ESConv、EvoEmo 与 ES-MemEval-derived 的外部迁移

训练覆盖“能力与输入形状”，不覆盖外部答案：

| 外部区域 | 训练必须提供的能力 |
|---|---|
| ESConv | memory 全 unavailable；support mode、goal、phase、RS opportunity 与 R0/RS effect |
| EvoEmo | 大目录、长 age/span、多 source opportunity、source/RS interaction、干扰和 abstain |
| ES-MemEval-derived | memory relevance/temporal conflict/multi-session continuity 和检索成本 |

同一个 PM 接受 legal-action mask。不得输入 domain ID，不得为 ESConv 另训 strategy model。
外部 test 的 gold strategy、回复、judge outcome 不参与 Bank 清洗、need-head 选择或阈值。
outcome-free 的 count/age/token/history/summary envelope 可以用于输入支持，但必须披露为
external-shape-informed development。

## 19. 历史执行顺序

### Phase 0：零 API 根治（现在）

1. 冻结本 V3 方法对象和非临床 claim；
2. 产出 Strategy Bank V1 全量质量/重复/内容/适用性审计，并把 ESConv 原生
   problem/emotion/experience/situation/strategy/survey 字段登记为明确的数据角色；
3. 构建 Strategy Bank V2 schema、raw+normalized metadata registry、过滤器和小型受审候选；
4. 实现 separated-view `SupportNeedObservation`；
5. 建 train-only need challenge 与 BAAI/lexical probe；
6. 实现 quality/evidence/strategy 三岗位新 prompt 和离线 schema tests；
7. 构建 outcome-free ESConv/EvoEmo input-support grid；
8. 建立第 16 节 machine-readable metric registry 与手算/反向 fixtures；任何含混总分不得
   进入 readiness。

### Phase 1：最小实质 pilot

分成两段，不能混成一次付费 pilot：

1. **Need/Bank 零 API pilot**：约 60–100 个新鲜 train states、每个 state 来自独立或
   正确分组的 user/dialogue；覆盖五个 support modes、goal/phase、显式边界、反事实、
   problem/emotion/experience 分层和 memory unavailable/available。先得到 out-of-fold
   `SupportNeedObservation`。首批已冻结 75 个 fresh dialogue/75 states 与 24 条
   emotion×position 人工 anchor；当前 raw Bank 与其 75/75 source overlap、1,094 cards，
   必须先在 Bank V2 删除为 0；
   当前已完成第一批 24 与 expansion-fit 16 的合议及 39-group OOF，但稀缺边界仍未过门；
   下一增量优先是 24 个历史 judge visible-state 的全量重新盲标和专门的
   planning/non-low-burden 最小反事实，不打开 confirmation；
2. **Effect 小额 pilot**：只从 Need/Bank 过门的 states 中选约 48–80 个，生成 MP/MS/ME/RS
   clean helpful/placebo/harm 单组件 matched pairs；不生成未经支持的交互或 all-three。

先看 need representation 和 Bank eligibility，再看 generator uptake；任一不成立就停止，
不进入大批 judge。所有 raw rows、groups、physical responses、weights 和 ESS 按第 17 节
同时报告。

### Phase 2：测量资格

- 从新 response distribution 盲抽 12–24 个边界对做人评；
- 用新 packet 资格测试每个 judge 岗位；
- 分歧按第 15 节处理；
- 不合格岗位不能靠其他岗位或综合分补票。

### Phase 3：train-only 可学习性

- 编译 component-level soft labels；
- 用 cross-fitted `SupportNeedObservation`，不能使用 in-sample need prediction；
- user/dialogue-group nested CV 比较 PM、transparent rule、M0+R0、same-cost fixed；
- 模型维度、组件/类别 ESS 和岗位权重满足第 17 节；
- 检查 action 非坍缩不是目标本身；只有 held-out effect 与约束选择有效才继续；
- 多源/RS interaction 只在 main effect 成功且 interaction ESS 足够后追加小样本；
- 只让获支持组件进入正式 action mask；完整 MP/MS/ME/RS claim 要求四个 main effects
  分别过门。

### Phase 4：一次性正式路线

1. 只扩量已获支持的 cell；
2. calibration 只消费一次，用于预注册校准和 fixed frontier；
3. candidate、need heads、BAAI、Bank、prompt、judge、cost 全冻结；
4. 两域 internal 各一次性消费；
5. internal 支持后才执行 ESConv 与 EvoEmo/ES-MemEval-derived external；
6. 外部失败不回头改训练。

## 20. 哪些旧产物复用，哪些不复用

| 旧产物 | V3 处理 |
|---|---|
| V8.19.2 468 states 与 canonical repair | 可复用作状态/根因/shape 池，不直接复用旧 action labels |
| 7,488 old-treatment outcomes / 20,736 judging | 保留为诊断、成本和失败证据，不重跑、不作 V3 gold |
| oracle memory / decomposition pilots | 只证明容量和定位干扰，可指导 pilot cells，不作 deployable label |
| 719 ESConv auxiliary generations | 可用于旧机制诊断；V3 Bank/prompt 变化后不能冒充新 treatment |
| fixed-seeker V3 formal | 可复用，除非可见状态/generator treatment 被 V3 改变 |
| BAAI snapshot | 暂作候选；Phase 0 资格过门才正式复用 |
| Strategy Bank V1 | 保留 provenance/baseline；正式 RS surface 必须经过 V2 资格 |
| sealed internal/external | 保持未开，不用于 V3 设计 |

不需要重新跑 actual-468、旧 full sweep 或旧 full judging 来证明它们仍失败。真正需要新生成的是
“clean treatment + clean Bank + final need interface”下的小型 response distribution；只有
小样本全链过门后才扩量。

## 21. 成功主张与诚实边界

V3 成功后允许的主张仍是原研究目标的有限版本：

> 在冻结生成器、检索机制和可见观测下，一个共享的 pre-retrieval PM 能否利用经验证的
> 支持需求表征与资源 opportunity，从小样本、有显式不确定性的弱监督中学习 MP/MS/ME/RS
> 的条件增量，并在回复质量非劣和风险不增加的约束下减少不必要资源与成本。

不允许声称：

- 理解了用户真实心理或完成临床诊断；
- 找到人类最优策略或 item；
- BAAI cosine 就是用户需要；
- LLM 多数票等于人类 gold；
- ESConv strategy labels 是所有现实建议/帮助的完整知识库；
- 内部成功自动证明外部迁移。

“确保成功”在科学上不能指保证结果为正；本方案能确保的是：在大规模付费和 holdout 消费
之前，需求可辨识、Bank 可用、treatment 有 effect、judge 能测、PM 能在 train-held-out
groups 学到五个条件必须逐个成立。任何一个不成立都能定位到具体层并提前停止，而不是再次
得到一个无法解释的 100% `M0+R0`。

## 22. 研究依据与能力边界

- [ESConv: Towards Emotional Support Dialog Systems](https://aclanthology.org/2021.acl-long.269/)
  用 Helping Skills Theory 和 8 类支持策略组织 ESC；它支持“策略重要”，不证明每个原始
  supporter 回合都是可直接检索的高质量建议。
- [ESConv 官方 expanded release](https://github.com/thu-coai/Emotional-Support-Conversation)
  提供 1,300-dialogue 文件、problem-topic 扩展、八类 turn strategy 与 FailedESConv
  质量判定说明；V3 直接审计本地同源 `ESConv.json`，但仍把 topic/emotion/survey 与
  action effect 分层，不把 release metadata 当动作 gold。
- [MultiESC](https://aclanthology.org/2022.emnlp-main.195/) 把动态用户状态建模和多轮策略规划
  视为两个核心挑战，支持本方案把需求表征与资源选择分开。
- [IntentionESC](https://aclanthology.org/2025.findings-acl.1358/) 明确定义
  emotion-state→intention→strategy 的推理链；本研究借鉴其分解思想，但不把自动推理当 gold。
- [COSMIC](https://aclanthology.org/2020.findings-emnlp.224/) 使用 COMET/ATOMIC 的
  speaker/listener intent、effect、reaction 作为连续 commonsense features；它说明用户所说
  的 “COMIC/ATO” 更可能是 COSMIC/COMET/ATOMIC 路线，也说明这些是辅助表征而非直接的
  当前支持需求标签。
- [BAAI BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5) 明确其 retrieval/
  semantic-similarity 定位、相似度分布和 task-specific threshold 要求，支持本方案把 BGE
  限定为待验证表示器。

## 23. ESConv/EvoEmo 零 API 数据理解审计（2026-07-27）

可复现脚本为 `scripts/v1_5/22m_profile_esconv_evoemo_v1_5.py`，正式本地输出为
`outputs/pm_v1_5_esconv_evoemo_understanding_v1/profile.json`；profile SHA 为
`cc2fd61e0d4c3fc6c2ad7314a2892a3c5a317160a52d19405fa7dd8786ce138c`。两次独立运行逐字节
一致，零 API。ESConv 的策略/反馈分析只使用 train 且排除 EvoEmo 重叠对话；validation/test
策略与 survey outcome 不读取。EvoEmo 只分析 deployable input shape 与 evaluator schema，
不读取 answer/future-topic text 进入模型或标签。

### 23.1 ESConv 实际能提供什么

- expanded release 共 1,300 dialogues；冻结 split 为 train/validation/test =
  934/186/180。train 中再排除 59 个 EvoEmo 重叠对话后，剩余 875 dialogues。
- 875 dialogues 中有 12,403 个 supporter turns，但 489 个发生在首个 seeker turn 之前，
  不能作为“看到当前用户话语后选策略”的决策样本；可用于该问题的实际决策数为 11,914。
- 观察到的八类 strategy 分布有覆盖，但不是最优标签。`problem_type`、`emotion_type`、
  `experience_type` 与下一条 observed strategy 的归一化互信息分别只有
  `.00273/.00248/.00018`；上一策略和对话阶段也只有 `.04849/.05172`。因此 topic/emotion
  适合分层和 coverage，不足以直接决定 RS 或 support move。
- 阶段规律是真实但不等于 gold：early 中 Question 占 35.1%；middle 中 Providing
  Suggestions 占 24.0%；late 中 Others 占 31.7%。模型必须看到阶段/连续性，但不能只是复刻
  人类 supporter 的顺序习惯。
- 明确 advice-request cue 只有 280 个；其中下一 move 是 Providing Suggestions 的仅
  30.4%。明确 listen-first/just-talk cue 在合法 train decision 中只有 3 个，且后续 move
  分别是 Others/Question/Providing Suggestions。原始 ESConv 无法单独支持五类
  `SupportNeedObservation` 的均衡学习，尤其无法学出可靠的 listen-first 边界。
- seeker turn-level feedback 只覆盖 5,130/11,914 decisions；各 strategy 均值集中在
  4.14–4.41。conversation survey 也存在缺失和明显高分天花板。二者只能作 post-treatment
  置信度/敏感性证据，不能成为 turn-level causal utility 或 RS gold。
- 表面质量诊断发现 267 个 normalized duplicate groups、1,205 行位于重复组、最大组 87；
  186 行只是问候/结束，657 行不超过 3 个词。问号/指令词与 strategy annotation 也存在大量
  不一致。启发式不能证明具体 annotation 错误，但足以否定“所有原 supporter turn 都是干净
  策略示范”。

对 V3 的直接约束：

1. ESConv strategy 只作低权重 observed-move prior、分层和 technique-family coverage；
2. `problem_type/emotion_type/experience_type` 不作为动作标签，只用于采样、group audit 和
   soft opportunity prior；
3. 由人评补足 listen/explore/comfort/light-guidance/structured-planning 的 need anchors，
   并为 listen-first、advice-request、burden-limit 建 matched counterfactual；
4. Strategy Bank V2 的 source evidence 必须排除“没有 prior seeker turn”的 opening move，
   以及 greeting/meta/duplicate/危险领域建议；
5. BAAI 是否增加可辨识性，要在人评 need label 完成后做 dialogue-group out-of-fold probe，
   与 lexical/metadata/constant baseline 比较；无增益则不进 PM。

### 23.2 EvoEmo 实际能提供什么

- 18 users、401 sessions、9,368 dialogue turns、446 events、34 subsequent topics。每用户
  session 中位数 22，非负 session gap 中位数 15 天、最大 387 天。
- 84 个 session ID 为 `escN`，可确定性回连 ESConv；因此 EvoEmo 不是 pristine independent
  external corpus，只能称 ESConv-derived longitudinal transfer benchmark。
- canonical loader 对原始文件中 6 个用户、34 个位置的非时间顺序做
  `ascending_iso_date_then_original_position_for_ties` 规范化；规范化后 session/reference
  完整性检查通过。所有正式分析必须经 `load_evoemo()`，不能直接按原始 JSON 顺序建立 age。
- 当前 deployable builder 产生的每用户目录中位数为 MP=7、MS=22、ME=68，最大
  MP/MS/ME=7/33/109；在未来 query 时，MS/ME age 中位数为 12/11 sessions，最大均为 33，
  MP age 中位数 23、最大 34。该 envelope 与旧 development 小目录是实质分布差，不是一个
  domain-name 问题。
- canonical catalog 总计 MP/MS/ME=126/401/1,215 items；ME word count 中位数 75、最大
  103。训练必须覆盖 count、age、age span、token burden 和多源干扰，不能只补零或只改变
  OOD threshold。
- evaluator questions 覆盖 information extraction/user modeling/temporal reasoning/
  conflict detection/abstention，subsequent topic 每项关联 2–8 sessions。这些能力说明外部会
  考什么，但 answer、evidence、related_sessions 和 future-topic text 均不得进入训练或调参。
- 4,679 个历史 supporter turns 含短回复、重复和 413 对连续同角色 turns；它们提供
  longitudinal context，不是干净的 response-quality 训练语料。

对 V3 的直接约束：

1. EvoEmo 只提供 outcome-free catalog shape envelope、chronology/reference contract 和正式
   external memory-routing 评测；
2. 新生成数据必须独立覆盖 MP/MS/ME 的可用/无关/陈旧/冲突、1–33 session age、small-to-large
   catalog、单源与多源 interference，但不仿写 EvoEmo topic 或答案；
3. 同一个 PM 在 ESConv 的 all-memory-unavailable 边界学 RS，在 longitudinal 区域联合学
   MP/MS/ME/RS；domain ID 仍禁止入模；
4. external 前报告 item-level retrieval hit、source-level routing、generator uptake 三层，
   避免把 retrieval 失败误归 PM。

### 23.3 “先无监督找规律”可以做什么、不能做什么

无监督/描述性分析可以：

- 找 coverage 缺口、类别不平衡、重复、阶段转移、catalog shape、age span 和异常；
- 用 frozen BAAI 聚类或近邻检查 train 的语义支持与 OOD，但只作为表示覆盖诊断；
- 形成 outcome-blind sampling strata 和反事实生成 cell。

它不能：

- 从聚类编号直接得到 listen/RS/MP/MS/ME gold；
- 把 observed strategy 的高频模式解释成最优策略；
- 把 survey/feedback 相关性解释成某策略的因果收益；
- 使用 EvoEmo QA/related-session/future-topic outcome 反向设计训练答案。

因此“吃透数据”之后的顺序是：完成当前 need/Bank 人评 → 用本节 profile 设计 coverage-balanced
train-only need probe → 验证 BAAI 是否增加 out-of-fold 可辨识性 → 只为缺口生成 matched
counterfactual 与 clean component treatments → 再验证 uptake 和岗位裁判。不能先扩大旧
ESConv action label，也不能直接重新训练旧 16-action gold。
