# PM V1.5：RS 卡片质量与启用资格

## 结论

当前对象不是“11,590 张卡中做 Top-k 检索”的大 RAG bank，而是 5 张冻结的
`technique-only` 卡组成的小目录：Question、Restatement、Reflection、
Affirmation、Providing Suggestions 各一张。透明边界和策略族门控完成后，每次
最多注入一张。因此当前 RS 是“family-gated Top-1”，不是把 Top-3 一起塞给
generator。

截至当前证据，这 5 张卡可以用于训练集内的 Top-1 RS 机制 pilot，但不能声称整个
RAG 卡库“质量非常好”，也不再作为正式 Strategy-RAG 最终候选。正式候选将按照
`PM_V1_5_STRATEGY_BANK_V3_EXPANSION_ZH.md` 扩展为 outcome-blind 的中型卡库。

## 怎样判断卡片是否真的好

卡片质量必须分成四层，不能用一个 judge 总分代替：

1. **构造与数据边界**：卡片不来自当前支持需求样本的源对话；不把 ESConv
   原始回复示例注入 generator；只保留一般策略技术。
2. **卡片内在质量**：人工逐项检查动作定义、适用/禁用条件、mode/phase/goal、
   负担、风险和一般安全性。
3. **选卡是否正确**：只有当前边界允许且策略族适配时才选卡；无合格机会就
   R0。卡写得好不等于每个状态都该用。
4. **注入后的因果效果**：同一状态、同一 generator、同一 prompt，只改变
   R0/RS；盲评是否产生实质质量收益，是否新增实质风险，成本是否可接受。

只有四层都通过，才允许 PM 把相应条件学成“开 RS”。轻微偏好或实质等价都
记为 R0，这正是“资源没有明显收益就不开”的研究主张。

## 当前证据

- 11,590 张原始卡经过规则筛选、去重和来源排除，最终形成 5 张技术卡。
- 支持需求 packet 的源对话重叠为 0；原始回复示例没有暴露给 generator。
- 5/5 卡通过人工逐字段审核，置信度均为 4，并批准 train-only pilot。
- 49 个 pilot 状态实际使用 4 张卡：Suggestions 27、Affirmation 13、
  Reflection 5、Restatement 4；Question 尚无响应级证据。
- 第一批 12 对盲评中，RS 为 8 胜、R0 为 4 胜；advice-welcome 子组为
  RS 5、R0 1。这个结果只是发现性信号，不是显著优越性结论。
- 第一批 RS 响应的适用风险单元中，没有人工确认的实质风险事件。样本仍小，
  所以只能称“初步未发现新增实质风险”。
- 第二批 12 对独立确认尚待完成人评，因此正式响应效果资格仍未通过。

ESConv 原对话的 empathy、relevance 和情绪变化统计只用于描述卡族来源，不能证明某张
卡在某个 turn 上有效：这些分数是 conversation-level、post-treatment 观察量，并非
卡片级随机对照结果。论文中不得把它们写成卡片质量标签。

机器可读汇总位于
`outputs/pm_v1_5_strategy_card_quality_audit_v1/card_quality_audit.json`。

## 第二批 12 对是否需要继续

从测量角度它原本需要，但在决定扩展 V3 卡库后先暂停。它验证的是旧五卡 treatment；
若卡库和 prompt guidance 改变，就不能把结果直接用于新卡库晋级。

若论文最终退回五卡机制版本，这批可以继续。若采用 V3 Strategy-RAG，则等新库冻结后
重新生成最小确认包，避免把人评花在即将变化的 treatment 上。

原确认包的设计本身仍然正确：它防止把第一批偶然的 5:1 advice 信号当成规律。
第二批与第一批用户和 pair 零重叠，选择时不看回复和 judge 结果。页面要求
只有实质差异才选 A/B，轻微偏好必须选 tie，正好修复第一批被迫二选一的问题。

完成后只做一次预先固定的判定：

- advice-welcome 的 RS 实质收益方向若复现，且 RS 未新增实质风险，则该条件
  可晋级为 PM 的候选正标签；
- listen-only 若以 tie 或方向不稳定为主，则保持 R0；
- 任一条件没有收益、风险不合格或成本超限，都保持 R0；
- 不用第二批结果反复修改卡片、seed 或样本定义。

第二批页面：
`outputs/pm_v1_5_minimum_rs_human_confirmation_v1_candidate/human_blind_review.html`

## Top-1 与 Top-k

当前 5 卡目录每个策略族只有一张合格卡，因而 RS 的 Top-3 没有干净定义。
直接拿它与旧 11,590 卡 Top-3 比较，会同时改变卡片内容、来源质量和数量，
无法知道差异来自 k 还是卡库。

V1.5 的最快可靠方案是冻结 RS 为“一族一张、最多一张注入”。Top-1/Top-3
校准优先用于 memory。若未来确实要研究策略 Top-k，需先在同一策略族内准备
至少三张彼此不同、都独立通过审核的卡，再保持排序器、generator 和 prompt
不变比较 Top-1 与 Top-3。这属于后续研究，不应阻塞当前论文。

## 已知但不阻断当前 pilot 的修正项

- Affirmation 可在后续版本考虑增加 action phase 和 `minimization` 风险旗。
- Providing Suggestions 需在后续版本澄清 structured planning 是“每回合
  推进一步”，并由上游安全流程屏蔽 acute/risk 状态。

这些意见在原人评中明确为非阻断项。现在直接改卡会改变 card id 和既有 49 对
响应的 treatment，因此当前确认实验继续使用冻结原卡；修正在确认结束后进入
新版本，不能悄悄覆盖当前实验。
