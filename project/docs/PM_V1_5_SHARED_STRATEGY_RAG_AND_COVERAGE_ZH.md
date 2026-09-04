# PM V1.5：统一 Strategy RAG 与跨域覆盖边界

## 结论

主实验必须在 development、internal、ESConv 和 EvoEmo 中使用同一个冻结的
Strategy RAG treatment。这里的“同一个”不只是使用同名目录，而是同时固定：

- 同一份卡片 catalog 及卡片内容；
- 同一 eligibility / safety / burden 过滤规则；
- 同一 query builder、retriever、score 与排序；
- 同一注入条数、prompt compiler、generator 和 cost accounting。

不同状态检索到不同卡是正常的。ESConv 中 memory 不可用、EvoEmo 中 memory 可用也是
合法的环境差异。按数据集更换卡库、检索器、Top-k 或 prompt，则会改变 `RS` 这个动作
本身的含义，不能再把结果解释为同一个 PM 的外部泛化。

## 为什么旧 V1.0 的“两套 RAG”直觉不荒谬，但不回答当前问题

把 RAG 当作外部资源，并为每个领域准备更贴合的资源库，工程上可能提高效果。它回答的
是另一个问题：

> 给定一个经过领域适配的资源库，router 能否在该领域中有效使用资源？

这属于 domain-specific resource adaptation。若卡库利用了外部域的文本、标签或结果，
还可能是 development-informed 或 transductive evaluation。它不能证明：

> 同一个 PM 是否学会了一个固定资源组件何时值得开启，并迁移到新的状态分布。

当前 V1.5 研究的是后一个问题。因此，两套 RAG 可以保留为将来的明确标注的
`domain-adapted bank` 次要消融，但不能进入当前论文主结果，也不能与统一 RAG 的训练
结果混写。

“RAG 是外部资源”不等于“每个 test 可以换一套资源”。外部只表示它不在 generator
参数中；它仍可像同一本工具书一样，在训练和考试中保持固定。

## 当前 50 张卡到底覆盖了什么

> 重要状态修正：下面只描述 50 张 top-down candidate 的 metadata 结构覆盖，不再
> 代表数据驱动覆盖。该 taxonomy 不是聚类所得；其正式资格等待 train-only 全八类
> open coding 与 blind alignment。见
> `PM_V1_5_ESCONV_INDUCTIVE_STRATEGY_AUDIT_ZH.md`。

当前 core candidate 有五个安全 support-move family，每族 10 张：

- Question；
- Restatement or Paraphrasing；
- Reflection of feelings；
- Affirmation and Reassurance；
- Providing Suggestions。

按卡片 metadata 计数，当前结构覆盖为：

| 轴 | 覆盖 |
|---|---|
| support mode | listen 29；comfort/reassure 16；explore 28；light guidance 20；structured planning 4 |
| dialogue phase | exploration 27；comforting 29；action 29 |
| goal | be heard 18；make sense 23；stabilize 17；decide 14；act 21 |
| directive burden | none 30；light 20 |

这说明 50 张卡已经是一个覆盖较广的**通用安全技术核心候选库**，不是原来五张卡的
简单复制，也没有按考试、工作、失恋等 topic 堆模板。它很可能足以支持 V1.5 的
Strategy-RAG 机制实验，不需要为了接近 80–100 而继续凑卡。

但当前仍不能写成“已经充分覆盖 ESConv 和 EvoEmo”，原因有三点：

1. 50 张定义的人审尚未完成，卡片仍是 candidate；
2. BGE 对 ESConv 来源的细粒度映射已证明会发生功能错配，不能拿 `31/50` 的弱映射
   当覆盖 gold；
3. 尚未在冻结检索栈上完成 outcome-blind 的机会覆盖与 retrieval sanity audit。

`structured_planning` 只有 4 张、负担只包含 `none/light`，是当前最明显的窄区。对
V1.5 可以把 RS 明确定义为“每回合最多提供一个安全 support move”，多步骤计划由多轮
逐步推进，而不是一次注入多张计划卡。若论文要主张一次完成复杂 planning，则当前卡库
不够；该主张不属于最小可发表 V1.5。

Self-disclosure、Information 和 Others 没进入核心库是有意的安全边界。因此本研究不
主张复现 ESConv 八种原始策略标签的完整分类器。普通事实问答也不是当前 RS 的职责；
没有合格 technique opportunity 时，PM 应关闭 RS。

## “覆盖 ESConv 和 EvoEmo”的准确含义

50 张卡只需要覆盖跨域可复用的 support techniques，不需要覆盖两个语料中的所有 topic。

- ESConv 是 memory 不可用的单会话边界：主要检验 `R0` 与 `RS` 的选择及即时回复。
- EvoEmo 是 memory 可用的多会话边界：同一组 RS 卡继续工作，但 MP/MS/ME 的内容由
  独立 memory catalog 提供。50 张策略卡不承担“覆盖所有长期记忆知识”的职责。

因此，Strategy Bank 的 topic 分布不必模仿 EvoEmo；真正需要对齐的是 PM 可观察的决策
条件，例如用户边界、建议准备度、phase、goal、负担、是否存在合格卡、memory
availability、干扰和 cost。外部文本分布可以变化，动作语义不能变化。

## PM 训练数据真正需要覆盖的“知识点”

PM 不是背卡片，也不在 action 前读取具体卡片文本。它要学习的是资源机会与后果的边界：

1. 没有真实合格机会时关闭；
2. 用户明确只想被倾听、拒绝建议或负担很高时关闭 directive RS；
3. 有明确 advice / exploration / stabilization opportunity 时，区分
   `material benefit`、`tie` 和 `harm`；
4. tie 默认关闭，因为没有实质收益就不承担额外 cost；
5. risk gate 或 cost budget 不通过时关闭；
6. 只有预期有实质质量收益且 risk、cost 均通过时开启。

训练数据因而必须按这些决策轴和最小反事实配对，而不是按 topic 均匀抽样。至少应包含：

- advice welcomed ↔ listen-only；
- relevant safe card ↔ irrelevant / boundary-mismatched card；
- low burden ↔ high burden；
- clear opportunity ↔ no opportunity；
- benefit ↔ tie ↔ harm；
- memory unavailable ↔ available，以及有无 source interference。

同一情境只改变一个关键边界，比收集更多相似 topic 的普通对话更有训练价值。数据按
dialogue/user 分组切分，test/external 的回复、judge outcome、gold strategy 和具体
措辞不得反向用于造卡、选 threshold 或补训练分布。

## 冻结前的最小通过标准

50 张卡是否“够用”由下面四项决定，不由卡片总数决定：

1. **定义通过**：全部卡片完成内容审核；未修正的安全/边界问题为零。
2. **结构覆盖通过**：五个 mode、三个 phase、五个 goal 均有可用卡；V1.5 的
   structured planning 明确限定为单回合一个 move。
3. **检索通过**：只在 train/calibration 的 outcome-blind 状态上检查 eligible set、
   Top-1/Top-k 相关性、错族和安全边界；不得使用 external outcome 修检索器。
4. **机制通过**：在同状态、同 generator 的 R0/RS clean pairs 上确认部分明确机会能
   产生实质收益，同时 tie/harm 足以支持“默认关闭”的训练边界。

冻结后，ESConv/EvoEmo 只能报告 outcome-free coverage diagnostics 和最终结果；即使
发现某域零命中较多，也不能为该域临时补卡或改检索。若确需修改，必须建立新的 study
版本并重做统一冻结。

对应机器可读合同：
`data/pm_v1_5_contracts/shared_strategy_rag_cross_domain_v1.json`。
