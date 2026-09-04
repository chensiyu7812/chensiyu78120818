# PM V1.5 论文可用发现、责任边界与表述建议

更新时间：2026-08-04  
用途：供论文方法、结果、讨论、局限性与未来工作撰写时统一引用。本文不是新的实验协议，不修改任何已冻结模型、阈值、资源、生成器或外部评测计划。

## 1. 最小且可守住的论文主张

PM V1.5 是一个低容量、可审计的预注入资源路由器。在固定资源库、检索器和生成栈下，它学习保守地控制 MP、MS、ME、RS 四类资源是否进入回复生成；目标不是最大化单一回复分数，而是在即时支持质量、interaction-and-grounding risk 与上下文成本之间形成有意义的折中。

当前证据支持的最小表述是：

> 在内容独立的内部确认中，learned PM 相对透明人工规则表现出更高质量、较低风险和较低 prompt token；相对 always-off 明显提高质量；相对 fixed-high 降低风险和成本，但完整系统的预冻结质量非劣门与 critical-grounding 门未全部通过。因此 V1.5 提供了有意义的保守资源路由学习证据，而不是四组件均已稳定学会或完整系统已经确认成功。

不得写成：PM 已理解复杂隐含需求、四组件均已外部泛化、记忆总能提高质量、或系统已通过全部确认门。

## 2. V5.2 内容独立确认的核心结果

完整确认使用同一冻结的四个 logistic heads、0.5 阈值、16 动作接口、V5.2 backend-locked executor 和内容独立确认集；结果后未修改方法。

### 2.1 系统级 Pareto 信号

- 相对 always-off：质量均值差 `+0.2734`，cluster bootstrap 95% CI `[+0.1602,+0.3867]`；说明 learned PM 不是退化成恒关，也确实能带来质量收益。
- 相对 transparent rule：质量 `+0.1719`、material-risk rate `-0.0430`、prompt token `-12.4844`；三个点估计方向均有利。
- 相对 component fixed-high：质量点估计 `+0.0313`、material-risk rate `-0.0625`、prompt token 降低 `7.18%`。
- 但相对 fixed-high 的质量差 95% CI 下界为 `-0.1094`，低于预冻结非劣界 `-0.05`；critical grounding 为 4 个 seed-level event、2 个独立 state/group，原门按 event 计数未通过。

因此完整系统结论必须同时保留“有意义的 Pareto 学习信号”与“正式完整确认门失败”，不能只择一报告。

### 2.2 四组件不是轮流随机失败，而是不同层的问题

| 组件 | 内容独立确认观察 | 正确归责 |
|---|---|---|
| MP | learned ON 约 50%；ON 相对 OFF 质量仅 `+0.0313`；风险未低于全开 | 候选子类型、可区分增量与 Step1 表示共同偏弱，不宜声称已稳定学会 |
| MS | ON 相对 OFF 质量约 `+0.5313`，风险较低；learned PM 存在漏开 | 资源有效，主要问题在 Step1 跨语义族泛化/保守性 |
| ME | ON 相对 OFF 质量约 `+0.5313`，风险低 | 当前最稳定的记忆收益组件；但自然 EvoEmo 表示覆盖不足 |
| RS | 平均质量效应接近 0；learned routing 明显降低 fixed-high 风险，但“可信联系人”卡产生 4 次 Sarah 虚构回忆 | Step2/card-executor 与候选执行风险为主，不能只归罪 Step1 |

这些数字是组件诊断，不是四个独立确认门；系统主结论仍以策略级 quality-risk-cost 比较为准。

## 3. Step1、检索与 Step2 的责任必须分开

论文应把端到端链路拆成三种可审计问题：

1. **检索/候选层**：是否找到正确 owner、严格过去、与当前目标相关且有增量的 Rank-1 候选。
2. **PM Step1**：给定实际候选，是否应该允许该组件注入；它不负责选择具体 item，也不负责写回复。
3. **Step2/executor**：已允许的资源是否被忠实、可归因、非表面地转化为回复，且不虚构、不把过去升级成当前事实、不违反边界。

历史上多次把“候选不好”“PM 漏开”“generator 没做功”和“回复质量主观分歧”都简称为“组件失败”，导致最新小包覆盖旧结论。V5.2 已通过 backend-locked typed clauses 与双臂 risk/quality 分离，使责任比旧 V1.0/V1.5 清楚，但 Step2 的自由生成残余仍是主要限制之一。

## 4. Risk 构念的可守表述

本研究的 risk 不是临床安全总风险，而是 `interaction-and-grounding risk proxy`，主要覆盖：

- 虚构用户既往陈述或人物；
- 把旧记录、单次事件或过去结果升级成当前事实、稳定规律或成因；
- unsupported personal claim；
- 明确边界违反、过度指令或任务堆叠；
- 内部资源标签/控制指令泄漏。

质量、资源做功和 risk 必须独立评：资源可以做功但有风险，也可以没做功但没有风险；盲质量页看不到授权历史，因此“可见对话中无依据”不能自动等于真实 grounding risk。

## 5. 人评与自动评测的合理分工

- 人评不是用十几条样本训练 PM；小包只用于开发期机制资格/故障定位，正式训练来自冻结的成对结果集。
- 质量使用匿名 A/B material preference；轻微风格差异为 tie。
- Grounding risk 必须展示授权证据并独立裁决；不可用词法 guard 代替语义风险判断。
- 资源是否真正做功是 Step2 机制指标，不能直接作为 Step1 gold。
- LLM judge 仅作外部稳健性/敏感性，不替代主人工构念；cost、coverage、routing 与 QA objective metrics 可全量自动计算。

## 6. RAG/RS 的论文意义

RS 的研究问题不是“RAG 总是有用”，而是“资源有实质边际收益且风险/负担允许时开启，否则关闭”。同状态 A/B 结果已经表明 RS 有赢、有输、有等价，因而开关问题非退化；但策略卡质量、Rank-1 适配和 atomic executor 必须共同成立。

同一冻结 Bank/检索器必须用于训练和考试；不能内部一套 RAG、外部再换成针对考题定制的 RAG。外部测试超出 Bank 能力时应报告覆盖/局限，而不能用外测结果反向扩 Bank。

## 7. ES-MemEval 与本研究的关系

ES-MemEval 按 information extraction、temporal reasoning、conflict detection、abstention、user modeling 五种历史推理能力组织 QA；本研究按资源来源/功能及边际注入决策组织 MP/MS/ME/RS。两者是交叉而非包含关系：

- 本研究在资源类型、可控注入、边际收益、grounding risk 与 cost 上更细；
- ES-MemEval 在跨会话时间、冲突、拒答和用户建模推理上更系统；
- QA 不能替代 response QRC，response 人评也不能证明检索/答案正确。

冻结的 418 题 QA 中，typed learned 实际为 MS-only：MS ON `392/418`，其余 26 题全关；typed fixed-high 仅 MP+MS。RS 对事实 QA 结构性 N/A，ME 因 V5.2 原子表示覆盖为 0。因此 QA 只能称 MS routing 的跨任务压力测试和 MP+MS comparator，不能称四组件外部成功。

## 8. MP 在 ES-MemEval QA 中全关的根因审计

此前“205 题有 MP 候选但 learned 全关”的说法必须精确化：205 只是 `structurally executable`，不是 MP opportunity gold。

### 8.1 候选实际内容

- 184/205（89.8%）只是用户名；
- 18/205 是职业；
- 3/205 是地点；
- 21 道 user-modeling 且 MP 可执行的题，其 Rank-1 MP 全部只是用户名。

问题文本通常已包含用户名，因此姓名是“词面最相关、信息量最低”的候选。当前外部适配器仍把多数此类候选记为 `candidate_incremental_information=1`，说明结构可执行门不能代表真实增量。

### 8.2 Step1 特征与训练目标错位

200/205 个 QA MP 候选具有完全相同的六维特征，learned probability 约 `0.2718`；全部205题概率范围仅 `[0.2066,0.2986]`。内部 MP head 的主要正系数来自 `candidate_preference_scope_fit`，但 QA 候选是姓名/职业/地点，preference fit 全为 0。

内部 MP 训练只有 11 positive、53 nonpositive；虽然 balanced accuracy 约 `.711`，但未击败 prevalence Brier，正式状态为 `NOT_LEARNED_ON_FROZEN_FIT`。内容独立确认中 MP 的质量净增益也接近零。因此 QA 全关不是单纯阈值偶然，而是：

1. 支持回复边际效用与事实 QA answerability 的 estimand 不同；
2. EvoEmo 外部 MP 表示主要是浅层身份字段；
3. Rank-1 检索受用户名词面重合支配；
4. MP_PROFILE 与 MP_PREFERENCE/CONSTRAINT 混在同一 head，而训练正信号主要来自后者；
5. 当前特征无法区分“属于正确用户”与“真正提供答案/回复增量”。

不能靠把阈值降到 `.25` 修复：大量同特征向量会一起开启，仍不能识别有用 MP。未来版本应区分 `MP_PROFILE` 与 `MP_PREFERENCE/CONSTRAINT`，并在 QA 使用 answerability-aware profile retrieval；这不是 V1.5 外测后允许回改的事项。

### 8.3 论文解释

建议表述：

> 冻结 PM 的 MS routing 能进入 ES-MemEval QA，而 MP routing 未迁移。审计发现，外部可执行 MP 候选主要为低信息量且常已在问题中出现的姓名，训练信号则主要来自支持回复中的偏好/负担匹配。因此该结果反映资源子类型、候选检索与任务效用的跨域不对齐，不应解释为 ES-MemEval 缺少用户建模能力。

E4 中 fixed-high 与 learned/no-memory 的比较只能回答“当前这些 MP Rank-1 是否产生增量”，不能评价一个未来更完整的 MP Bank。

## 9. ES-MemEval 检索分母与公开数据限制

官方 session Top-4 在 341 道所有非空 evidence 都可严格映射到同用户 session 的题上，Recall@4=`.6963`、nDCG@4=`.5997`。

其余 77 题分为：

- 69 道 abstention 且 official evidence 为空；
- 1 道 conflict detection、answer=`Unknown`、evidence 为空；
- 5 道含公开对象无对应字段的 `timeline:n`；
- 2 道含 `p16_evnet_17` 拼写不一致。

前70题没有检索正例，只进入 QA 答案/拒答指标；后7题属于公开 artifact 标识问题。主检索分母保持341；确定性修复7题后的 n=348 敏感性为 Recall@4=`.6930`、nDCG@4=`.5941`。不得静默修 gold，也不能把70道无正例题记成检索器失败。

## 10. 内部、ESConv、EvoEmo/ES-MemEval 的“学习—考试”关系

- 内部受控环境负责训练与识别四组件的因果边际效应，并补测外部数据无法覆盖的 preference、constraint、ME reusable outcome、RS atomic move。
- ESConv response 主要检验 topic-agnostic 支持策略与即时支持回复，不适合证明长期个体记忆。
- EvoEmo response 使用每个外部用户自己的历史与同一 compiler/runtime，检验长期连续性、记忆注入、风险和成本；已暴露分区只能称 repaired/post-hoc replication。
- ES-MemEval QA 检验检索、时间、冲突、拒答、用户建模与答案正确性，是互补诊断，不是 response QRC 主表。

外部考题必须位于训练定义和运行机制的可声明边界内；超出表示覆盖的部分报告 coverage/abstain 和局限，不能用外测答案反向重建训练集或资源库。

## 11. 主要局限与未来工作

1. V5.2 完整系统确认门未全部通过；外部运行不能改变这一内部确认结论。
2. MP 未稳定学会，且外部 profile 表示浅、子类型混合。
3. ME 的内部原子表示没有覆盖自然 EvoEmo 事件表述；QA 中 ME 为 0。
4. MS 有真实质量收益但 Step1 可能漏开；跨语义族与跨任务校准仍有限。
5. RS 某些卡与自由 executor 组合可产生重复虚构；需要新版本的 card-specific execution constraints。
6. 当前 PM 不声称理解复杂隐含用户需求；更强 NLI/LLM semantic encoder 仅可作为未来 challenger。
7. 内部确认是合成受控环境；ESConv/EvoEmo 已有历史暴露，外部结果是 repaired replication，不是 pristine lockbox。
8. 人评存在主观性与来源披露导致的臂可识别问题；通过构念分离、位置随机、独立重叠和一次分歧裁决降低但不能消除。

## 12. 推荐论文结构

1. **问题定义**：固定资源栈下的 quality-risk-cost policy，而非万能情感支持模型。
2. **架构**：候选/检索、Step1 四个 component heads、16动作组合、typed Step2 executor。
3. **训练**：同状态单组件反事实、grouped OOF、低容量 logistic、固定阈值。
4. **内部结果**：策略级 Pareto 与四组件责任诊断。
5. **外部 response replication**：ESConv/EvoEmo 的质量、risk、cost、coverage。
6. **ES-MemEval QA diagnostic**：官方 objective metrics、session retrieval、MS-only typed transfer 与 MP/ME覆盖边界。
7. **消融/基线**：always-off、fixed-high、transparent rule、learned PM、current cost-matched；EvoEmo 次表含 Raw Top-4/All Raw，Legacy V1.0仅历史次表。
8. **讨论**：资源控制细粒度与记忆推理能力互补；MP/ME/RS局限与未来工作。

## 13. 主要证据入口

- V5.2 内容独立确认：`outputs/pm_v1_5_v5_2_confirmation_final_report_v1/report.html`
- V5.2 FIT 最终报告：`outputs/pm_v1_5_v5_2_fit_final_report_v1/`
- 外部验证计划：`docs/PM_V1_5_EXTERNAL_VALIDATION_AND_ESMEMEVAL_PLAN_20260804_ZH.md`
- ES-MemEval 适配与检索分母审计：`outputs/pm_v1_5_es_memeval_fit_audit_v1/report.html`
- 全局失败与纠正账本：`docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`

