# PM v1.5 到底要学什么：最小、默认关闭、可发表的学习契约

状态：`ACTIVE / TARGET CLEAR / TRAINING NOT YET AUTHORIZED`

> **2026-07-29 范围澄清：** 最终规范以
> `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md` 为准。下文“先测 ME/RS”只表示历史识别顺序，
> 不再表示最终训练范围；MP、MS、ME、RS 四个 heads 都必须尝试，16 个合法动作全部
> 保留，并在适用外部环境中评测。

机器可读契约：
`data/pm_v1_5_contracts/minimum_pm_learning_target_v1.json`

## 一句话结论

PM 不需要“完整诊断用户需求”，也不负责挑具体记忆、挑具体策略卡或写回复。
它只学一个前置的、默认关闭的组件门：

> 在当前可观察对话和 source-level opportunity 下，打开一个资源组件是否有足够大
> 概率让本回合回复质量实质变好，同时不突破预先定义的 interaction/grounding risk
> 和 cost 约束？

若答案不是可靠的“是”，就输出 `M0+R0`。

这不会改变 v1.5 的主张。主张仍然是：一个小型、可审计的 PM 能学到至少一个有意义
的资源开关，并比 always-on 少做无收益调用；不是证明 PM 能理解所有需求或全局最优。

## 1. “开不开都行”到底怎样处理

必须区分两类 tie：

1. **真的等价**：合格评测确认开与不开在回复质量上 materially equivalent。推理时关，
   训练时可作为 `nonpositive_off`。
2. **测不出来**：judge 顺序不稳定、证据不足或人评不确定。推理时同样关，但训练时是
   `unknown`，必须排除，不能伪装成负标签。

因此，49 对中的 26 个 LLM tie 不能直接变成 off 标签。它们来自未通过 AB/BA
资格门的评测工具，含义是“未知”，不是“已证明等价”。

## 2. 正式16动作保留；先缩小的是识别试验，不是动作空间

当前真实配置仍是：

- memory subset：`M0 / MP / MS / ME / MPMS / MPE / MSE / MPMSME`；
- strategy mode：`R0 / RS`；
- 两者组合共 16 个动作。

`M0+R0`、`ME+R0`、`M0+RS` 是历史第一阶段 clean-pair 中首先比较的三个条件，不是永久
替代16动作；最终计划还必须补齐 `MP+R0` 和 `MS+R0` 的单组件识别。PM 也不应重新做一个直接16分类器；更稳妥的是分别学习 MP、MS、ME、RS
四个组件的边际效应，再组合成合法动作。缺少独立证据的多组件 interaction 保持
mask/off，而不是假定多个单项有用就一定可以安全叠加。

“最低可发表只要求至少一个组件学会”也不等于删除其余动作：它表示未获得数据支持的
组件或组合可以在本次模型中保持关闭，同时诚实报告 unsupported。

历史上先测 ME 的理由不是宣称它已经最好，而是它的 provenance 最直接；既有 oracle evidence
中 event-needed 3/3 出现正 uptake，三组 multisource decomposition 里 ME 单源也
3/3 为正。但这些只是“生成器能吸收一条干净记忆”的机制证据，不是回复质量证据。

## 3. “情况”由什么构成

PM 只看实际部署时可见的、检索前信息：

- 当前用户文本、受限长度的当前对话、部署时本来就可见的 session summary；
- 逐字可查的请求和边界，例如 welcome advice、listen-only、one-step、low-burden；
- source 是否有数据、数量、age、预计 token；
- 与真实执行使用同一 catalog 和 filters 得出的 family-level opportunity；
- 不得看 gold action、oracle item ID、候选正文、两臂回复、judge 结果或未来 turn。

RS 的最小条件是“用户边界 × strategy family opportunity”。本次人评发现的
`advice_welcome × Providing Suggestions` 只是候选 cell，必须在独立、结果盲选的
样本上确认，不能把发现样本又当确认样本。

ME 的最小条件是：有一条特定过去事件/承诺/细节；当前可见上下文没有重复；时效和冲突
检查通过；并且 metadata-level relevance opportunity 存在。

## 4. “怎样调用和使用”由谁负责

这一点不能全塞给 PM：

1. PM 输出 MP、MS、ME、RS 的 source-level bit，并组合成现有16个合法动作之一。
2. 同一套 RAG catalog、filters 和 metadata 定义同时服务 opportunity preview 与真实执行。
3. 历史正式 retriever 按 source 固定 top-k：MP=2、MS=2、ME=3，旧 raw RS
   bank=3；无正分命中才 realization=`off`。当前最小 RS pilot 已换成经过人工审核的
   5 卡目录，每个策略族只有一张卡，因此在 family gate 后固定 Top-1。两者不是同一
   treatment，不能直接用结果挑 k。
4. prompt compiler 用固定、受限的 slot 注入检索结果：
   - ME 是可能不完整的证据，只能谨慎引用，不得扩写成用户事实或原因；
   - RS 是回复技巧约束，不是要照抄的内容。
5. generator、seed/decoding、visible context 在 paired arms 中固定。
6. 生成后才审 uptake、引用是否正确、是否陈旧/冲突、质量和风险；这些不能回流成
   当前样本的 PM 输入。

所以“正确调用记忆”不是单个分类标签，也不等于永久 Top-1，而是：
`PM 开门正确 × 检索 item 正确 × 过滤正确 × 注入正确 × generator 使用正确`。
任一环节未固定，就无法把失败归因给 PM。

### 4.1 Top-1 还是当前 Top-k 必须先校准

历史正式 Filter 关闭时，retriever 返回的内容会全部进入 generator：

- `MP+R0` 最多 2 条；
- `MS+R0` 最多 2 条；
- `ME+R0` 最多 3 条；
- 旧 raw bank 的 `M0+RS` 最多 3 张卡；
- `MPMSME+RS` 最多 7 条记忆 + 3 张卡，共 10 块内容。

当前 5 卡目录则是每个策略族恰好一张合格卡，RS 在透明 family gate 后最多注入一张。
它不是旧 raw bank 的 rank-1 前缀。若直接比较新目录 Top-1 与旧 bank Top-3，会同时
改变卡片内容、来源质量和数量，无法识别 cardinality effect。

Top-1 可能减少无关信息、冲突和 token，但 lexical-cosine 第一名不等于真正最相关，
所以 memory 也可能损失 recall。V1.5 不把 k 再扩成 PM 的新动作维度，而是在
train/calibration 固定以下对照：

- ME：`M0 / ME-top1 / ME-top3`。
- RS：当前论文候选固定为 5 卡目录的 family-gated Top-1；旧 raw bank 的
  `R0 / top1 / top3` 只允许作单独历史诊断，不能决定新目录的 k。

任何真正的 cardinality 对照都要求同一状态、catalog、排序、generator 和 prompt，
且 Top-1 是 Top-3 的严格 rank-1 前缀；分别报告 quality、逐 rank evidence
relevance/freshness、原子 risk、tokens 和实际 item 数。用 calibration 选定 memory k
后冻结，learned PM 和全部 matched baselines 都使用同一设置。不能看到 final main-table
结果后再挑 Top-1 或 Top-3。若未来要测试新策略目录的 Top-3，必须先在同一 eligible
family 内构造至少三张独立合格卡，这不阻塞 V1.5。

## 5. quality、risk、cost 的清楚定义

### Quality

名称固定为 **immediate support quality**。按以下顺序做决定，不做加权平均：

1. 当前请求和明确边界是否更匹配；
2. 情绪是否更贴合且没有越界推断；
3. 是否对本回合更有帮助；
4. 是否清楚自然。

人评必须允许 treatment better、control better、materially equivalent、uncertain。
“稍微喜欢 A”不应自动成为 hard positive；正标签要求可引用的实质差异。
最终 learned-policy 与 comparator 的预注册非劣 margin 仍为 NetWin `-0.05`。

### Risk

论文统一称 **interaction-and-grounding risk proxy**，不是笼统的 safety，更不是
临床风险。四类原子事件分别报告：

- explicit boundary violation；
- unsupported personal claim；
- stale or conflicting evidence use；
- excessive directiveness。

每类保留自己的适用分母，material event 使用冻结的 `severity >= 2`；不合成总风险分。
其中 stale/conflict 只有 memory 真正进入 prompt 才适用，因此 RS-only 实验无法验证它。
最终每类 learned-minus-comparator rate 的非劣 margin 是 `+0.05`；memory-on 独立用户不足
20 时只作描述性报告，不冒充通过了非劣检验。

### Cost

报告 generator input/total tokens 和 realized retrieval calls。Cost 是硬预算和合格
正动作之间的平局裁决，不允许用低成本补偿“没有质量收益”。
最终相对 high-resource comparator 的实用阈值是 input tokens 至少减少 10%。

## 6. PM 的标签和决策

`positive_open` 同时要求：

- 真实 opportunity 存在，且 treatment 确实执行；
- 合格 paired measurement 显示质量有实质提升；
- 没有 applicable material risk 超过容忍范围；
- cost 在冻结预算内。

`nonpositive_off` 包括：无机会、control 更好、合格评测确认等价、risk 变差或超预算。

`unknown` 包括：treatment 未执行、评测工具不合格、人/judge 不确定、lineage/applicability
不清楚、或该样本用于发现 stratum。Unknown 在推理时默认关，在训练时排除。

决策不是 `quality - λrisk - μcost`。顺序是：机会 → 质量增益概率 → risk 约束 →
cost 约束；通过后才开，多个通过动作才用低 cost 打破平局。

## 7. 这 12 对人评告诉了什么

人评数据结构完整：12/12 对、24/24 回复、96/96 risk cells，无 blind ID 重复或缺失。

- 总体：RS 8，R0 4，精确 sign test `p=0.388`；
- advice-welcome：RS 5，R0 1；
- listen-only：RS 3，R0 3；
- Providing Suggestions：RS 5，R0 1；
- 人评决定标准：request fit 7、emotional attunement 3、immediate helpfulness 2。

这说明“条件性 RS benefit”值得继续，而不是证明 RS 总体更好。只有一个未署名 annotation
set、没有 inter-rater reliability、12 次都选 A/B 而没有 tie/uncertain，且关键 advice
cell 只有 6 对，所以不能直接训练。

同 12 对上，LLM quality 只给 RS 1、R0 3、tie 8；与人评 exact agreement 3/12。
它的结论是“质量 judge 未合格”，不是“RS 没有效果”。

风险 sanity check 更好：

- excessive directiveness：24/24 与人评 material 判断一致；
- unsupported personal claim：24/24 一致；
- explicit boundary violation：18/20 一致，LLM 在一组 listen-only 回复上双臂误报；
- stale/conflict：24 条全部程序性 N/A。

因此 risk 可以保留原子规则并继续校准；quality 评测必须修，memory risk 仍需 memory-on
样本。

可复跑结果：
`outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_analysis/human_audit_analysis.json`

## 8. 最快的正确推进顺序

1. **不训练当前 49 对。** 26 个不可靠 tie 不能当 off；7/16 也不能当 hard effect。
2. **RS 只做一次小确认。** 已从未做人评的 37 对中按 metadata 固定另一组 12 对
   （6 advice、6 listen，12 个独立用户，与发现样本零重叠）；允许 true
   tie/uncertain。页面位于
   `outputs/pm_v1_5_minimum_rs_human_confirmation_v1_candidate/human_blind_review.html`。
   关键分歧项只补一个第二标注者，而不是全量双标。
   若该独立样本复现方向，第一批 12 对不再作 confirmatory evidence，但可以在仅补
   “material / slight”这一项后复用为 train-only 数据，无需重做 96 个风险单元。
3. **冻结质量测量。** 用 24 对人评 anchor 校准一个更简单的 pointwise/paired rubric；
   若仍过不了资格门，v1.5 就直接以小规模盲法人评作为主 quality outcome，不再让 LLM
   judge 生产 hard labels。
4. **先做 ME clean pair。** 构造明确 opportunity、placebo/重复、stale/conflict 三类。
   因果诊断可以临时固定为一条 oracle/Top-1 ME，以隔离“能否正确使用一条证据”；这不
   改写历史 RAG 的 MP=2、MS=2、ME=3、raw RS=3。随后 memory 还必须回到冻结的
   正式 cardinality 做端到端确认；当前新 RS 目录保持 family-gated Top-1。
5. **达到每个 head 至少 8 个独立 positive 与 8 个独立 nonpositive 后再训练。** 模型先用
   透明规则、正则 logistic regression 或浅树；BAAI/Qwen embedding 只有在 grouped OOF
   明确增益时才加入。
6. **最终只报三张并列表。** Quality win/tie/loss；每个原子 risk 的适用分母和事件率；
   tokens/calls。与 always-off、always-on-when-available、literal-rule gate 比较。

EvoEmo 不需要强行“对齐成 ESConv 的分布”。它只作为固定 external/OOD stress test，
按预注册的可观察 request/boundary/burden/opportunity strata 分层报告。若做 reweight，
权重只能由 train-only、outcome 前的可观察协变量产生，并同时报告未加权结果；不能用
回复、judge、PM action 或 risk outcome 调权，也不能把没有真实纵向 memory opportunity
的 EvoEmo state 当 ME 负样本。分布不一致是 transport limitation，不是靠 outcome-aware
对齐消掉的问题。

完成这六步后，PM 学什么、何时开、开后谁负责正确使用、标签怎样产生、哪些数据不能用，
才算从研究层面闭环。当前已经把“问题定义”闭环了；还没有闭环的是质量测量资格和 ME
端到端训练数据，这两项正是下一步，不应再扩展需求诊断模型；正式16动作保持不变。
