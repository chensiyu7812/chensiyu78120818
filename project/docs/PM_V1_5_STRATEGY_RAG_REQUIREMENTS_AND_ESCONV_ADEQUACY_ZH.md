# PM V1.5：从研究问题反推 Strategy RAG，并评估 ESConv train 是否足够

状态：`SOURCE_VOLUME PASS / TWO HUMAN SETS COMPLETE / FINAL CODEBOOK PENDING`

两份 72 条归并已完成并审计，最终使用决定见
`docs/PM_V1_5_DUAL_STRATEGY_ANNOTATION_DECISION_ZH.md`；后续规范见
`docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`。

## 1. 先定义研究需要什么，不从 ESConv 标签反推

V1.5 的核心问题是：

> 一个检索前、默认关闭的小型 PM，能否在 held-out 对话/用户上学会：什么时候打开一个
> 固定的 Strategy 资源会让本回合回复产生实质质量收益，同时不明显增加
> interaction-and-grounding risk，并在无收益时省掉资源成本？

因此，RS 资源必须同时满足：

1. **有时有用、有时无用或有害**：否则 PM 只需 always-on 或 always-off。
2. **同一动作语义**：train、internal、ESConv、EvoEmo 使用同一 Bank、filter、
   retriever、Top-k、prompt 和 generator。
3. **动作前可判断机会**：PM 只能使用当前可见对话、显式边界和 source-level
   opportunity，不能提前读取 card 正文、gold、judge 或未来 turn。
4. **能改变 generator 行为**：卡片必须给出一个具体 support move，而不只是
   “be supportive”。
5. **风险和成本可控**：卡片不能携带私人对话事实或未经验证的 domain knowledge；
   每回合注入量固定且较小。

这决定了本研究需要的不是一般“知识 RAG”，而是：

> **Strategy Guidance RAG：检索一个当前可执行、带适用条件和禁止条件的支持动作。**

## 2. 三种资源的责任必须分开

| 资源 | 负责什么 | 不负责什么 |
|---|---|---|
| Strategy RAG / RS | 本回合采用何种支持动作及如何安全执行 | 用户个人历史、事实知识、诊断 |
| MP/MS/ME memory RAG | 用户过去明确提供的 profile/session/event 证据 | 决定当前应采用何种支持动作 |
| generator 参数知识 | 普通语言生成和一般常识 | 把未验证事实写成用户事实 |

ESConv 是 memory unavailable 的单会话边界，只比较 R0/RS。EvoEmo 是 memory
available 的纵向边界，RS 仍使用完全相同的 Strategy Bank；MP/MS/ME 另行提供记忆。
Strategy Bank 不需要覆盖 EvoEmo 中所有个人经历或长期事实。

## 3. RAG 需要覆盖的是行为区域，不是 topic

V1.5 的最小行为覆盖为：

1. **倾听与准确理解**：简洁复述、直接证据支持的情绪反映、允许纠正；
2. **低负担探索**：一个必要的澄清问题、目标/重点/准备度确认；
3. **安慰与稳定**：基于可见证据的 validation、努力确认、有限而非保证式希望；
4. **用户欢迎的轻量行动帮助**：一个可拒绝 micro-step、少量选项、第一步；
5. **转换与边界管理**：询问是否愿意进入建议、尊重 listen-only/one-step/
   low-burden 边界。

这些区域可以重叠，也不要求最终恰好五类。它们是研究要求，不是预设聚类标签。

V1.5 不要求 Strategy RAG：

- 覆盖考试、失恋、工作等每个 topic；
- 提供医疗、法律、财务或临床建议；
- 复现 ESConv 原生八类；
- 一次完成复杂多步骤 planning；
- 注入 raw supporter response。

## 4. 一张合格卡的最小结构

每张卡必须是一个 atomic support move，至少含：

- `support_move`：generator 要实际做什么；
- `when_to_use`：可从当前 visible state 观察的正条件；
- `when_not_to_use`：边界、负担、风险和反例；
- `compatible_phase / goal / burden`；
- `risk_flags`；
- `retrieval_text`：仅服务排序，不含 private example；
- `prompt_guidance`：简短、可执行、不会要求照抄来源；
- `source_support`：train-only 独立 dialogue 证据和人工核查结果。

不同 topic 的同一个动作不是不同卡。只有适用边界或 generator 行为真正改变，才拆卡。

## 5. 最小检索结构

1. **透明 eligibility**：先依据显式 listen-only、no-advice、one-step、负担和风险规则
   排除不允许的卡；这些规则不由 embedding 猜。
2. **冻结排序**：只在 eligible cards 中，用同一个 train/calibration 资格赛选出的
   lexical 或 BGE retriever 排序。
3. **默认 Top-1**：机制阶段每回合只注入一张 atomic move；Top-2/3 只有在同一冻结
   catalog、同一排序且 Top-1 是严格前缀时才做 calibration 对照。
4. **无安全相关命中则 R0**：低于 score floor 或没有 eligible card 时不注入。
5. **PM 与 item retrieval 分责**：PM 决定 RS on/off；RS on 后 retriever 才选择具体
   card。PM 最多看到 eligible count、最高匹配强度等 source-level opportunity，
   不看到 card 正文。

R0 与 RS clean pair 除是否注入这张卡之外，visible state、generator、prompt 基础模板、
decoding、seed 和 output limit 必须相同。

## 6. “足够能用”的预先合格门

卡片总数不是 KPI。V1.5 使用以下门：

### 6.1 单卡来源门

- 至少 20 个独立 ESConv train dialogues 出现该 atomic move；
- 至少 3 个 problem types 和 3 个 emotion types，或有明确理由说明该动作天然窄；
- 不能由一个 problem type 占全部有效支持的 60% 以上；
- 固定来源样例审核中，动作匹配、通用性和风险边界均通过；
- synthetic paraphrase 可以增强 retriever，但不计独立来源数。

### 6.2 Bank 行为覆盖门

- 上述五个行为区域各至少有 2 个真正不同的合格动作；
- 不要求每区卡数相同；
- 没有新动作能通过单卡来源门时停止扩卡，最终数量由数据决定。

### 6.3 检索门

在 bank-disjoint train/calibration query audit 上：

- 至少 40 个预先分层的正 opportunity states；
- Top-1 同时 relevant、safe、符合边界的比例至少 80%；
- 明确 listen-only/no-advice/high-burden controls 中，material boundary violation 为 0；
- wrong-family、no-eligible 和 score-floor 行为分别报告；
- 比较 lexical 与 BGE，只选同一审核集上更简单且达到门槛者。

该 80% 是 V1.5 的最低工程资格线，不是对真实用户帮助率的估计。未达到时先修 card
定义/eligibility；不得靠外部 test 调 threshold。

### 6.4 Treatment 与可学习门

- clean R0/RS pair 证明 generator 对 card 有可观察 uptake；
- 至少 8 个独立 group 为 `positive_open`；
- 至少 8 个独立 group 为合格 `nonpositive_off`；
- unknown、未执行和测不出的 tie 不进训练；
- component head 使用 dialogue/user-group OOF，并胜 constant prior；
- 最终仍分别报告 quality、四项 atomic risk 和 tokens/calls，不合成一个分数。

## 7. ESConv train-only 当前供给是否够

当前合法来源为：

- custom 70/15/15 中 875 个 non-overlap train dialogues；
- raw bank 已排除 52 个 development seed sources，剩 823；
- 再排除 75 个 SupportNeed packet sources；
- 客观清洗后为 9,148 条 supporter turns、748 个独立 dialogues；
- validation/test/external outcome 使用量为 0。

八个 native label 清洗后每类仍有：

- 562–1,632 条 turn；
- 371–654 个独立 dialogue。

因此，从**总体数量和高层行为来源**看，ESConv train 足以建立一个紧凑、通用、
每回合 Top-1 的 Strategy Guidance RAG。支撑 10 张最低行为覆盖卡需要至少约
200 个 dialogue-card incidences；支撑 20–30 张卡需要约 400–600 个。当前有 748 个
dialogue 和 9,148 个可编码 turns，数量不是主要瓶颈。

但它还不能证明：

- 具体有多少个稳定 atomic moves；
- 当前 50 张中的每一张都有 20 个独立 dialogue 支持；
- retriever 能达到 80% Top-1 relevance；
- cards 确实给 generator 带来 material benefit。

这些必须经过 open coding、全量回标、retrieval audit 和 clean-pair uptake 才能回答。
所以当前结论是：

> **ESConv train 对 V1.5 紧凑通用 RAG“数量上足够、语义资格待证”；不是数据不足，
> 也不是已经建成。**

## 8. ESConv 的真实限制

来源 topic 很不均衡：

- ongoing depression、breakup、job crisis、friends、academic pressure 五类占
  9,148 条中的 8,901 条，约 97.3%；
- 其余 Sleep、Procrastination、Appearance、Children、Alcohol 合计很少；
- anxiety/depression/sadness 三种 emotion 占约 76.8%。

因此 ESConv 足以支持 topic-agnostic support moves，不足以支持：

- 各 topic 的专门建议卡；
- 医疗/成瘾/儿童等稀少领域的事实型 guidance；
- “已覆盖所有情绪支持情境”的主张。

这正是卡片必须抽象为通用动作、不能按 topic 复制的原因。

## 9. 如果归纳后仍不够

按以下顺序处理，不能使用 validation/test/EvoEmo 补库：

1. **合并粒度过细的动作**：把只有措辞区别的 submoves 合并；
2. **收窄论文主张**：只保留有来源、有 uptake 的行为区域，其他 RS opportunity
   fail-closed；
3. **增加 train-only 人工开放编码**：从 9,148 条中继续做 outcome-blind 分层样本；
4. **使用独立公开方法来源补卡**：例如可引用、允许使用的通用支持技巧材料；必须单独
   标注 provenance、人工审核，并让同一新 Bank 重跑 train/internal/external；
5. **synthetic 只做检索鲁棒性**：可生成表达变体，但不能冒充独立真实来源或 card
   utility evidence。

禁止：

- 打开 ESConv validation/test 来补稀缺动作；
- 根据 EvoEmo 文本或结果定制卡；
- 为不同外部实验各换一套 Bank；
- 为凑卡数制造 topic-only variants。

## 10. 当前最短路线

1. 完成 160 条 label-blind open coding，归纳 atomic move codebook；
2. 用 codebook 对 9,148 条做弱回标，并人工核查所有边界分歧和固定样本；
3. 按单卡来源门得到自然数量的 candidate cards；
4. 与现有 50 张 top-down reference 对齐，只复用有数据支持的定义；
5. 在 bank-disjoint train/calibration 上完成 lexical-vs-BGE Top-1 audit；
6. 冻结 Bank/retriever/Top-1/prompt；
7. 生成 R0/RS clean pairs，检查 uptake、quality、risk、cost 和可学习性；
8. 最后原样运行 ESConv 与 EvoEmo；外部发现覆盖不足只能报告，不能回头补卡。

机器可读合同：
`data/pm_v1_5_contracts/strategy_rag_requirements_and_source_adequacy_v1.json`。

双模型开放编码与最小人工门的实际结果见：
`docs/PM_V1_5_STRATEGY_OPEN_CODING_TWO_PASS_RESULT_ZH.md`。
