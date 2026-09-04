# PM-v1.5 历史人评资产审计与使用边界

> V1.5 最低可发表路线只保留五卡审核为必做人评；历史 24-state SupportNeed
> reannotation 与 8 条 confirmation 已降为 V2.0 可选资产，不再阻塞当前训练。

日期：2026-07-28  
协议：`pm-v1.5-historical-human-annotation-asset-audit-v1`

## 1. 结论

保留项目中名称涉及 human/reviewer/annotation/adjudication/spot-check 的 101 个文件，
只有 46 个不同文件哈希。大量文件是同一盲包、空模板或 candidate/repro 副本，不能
按文件数计算人评量或有效样本量。

当前真正完成的人评有 64 条逻辑记录：

- 40 条 SupportNeed：第一批 24 条（含 1 条 abstain）和第二批 16 条双人评合议结果；
- 12 条 low-budget judge 候选回复偏好；
- 12 条 role-decomposed judge 回复偏好与错误维度审查。

只有前两组 SupportNeed 资产同构、outcome-blind，允许合并进入训练内
factorized/grouped-OOF 诊断。合并后是 40 行、39 个非 abstain 独立 dialogue group。
第二批的两份原始人评共 32 行只表示 16 个被重复评审的状态，不能把 ESS 加倍。

## 2. 资产分流

| 逻辑资产 | 完成情况 | 可用于什么 | 禁止用途 |
|---|---:|---|---|
| SupportNeed 第一批 | 24/24；23 non-abstain | factorized need heads、grouped OOF、rubric error analysis | action gold、独立 test、临床标签 |
| SupportNeed 第二批双人评 | A/B 各 16；合议 16 | 合议行进入 fit；A/B 分歧用于可靠性与边界分析 | 把 A/B 当 32 个 group；从分歧中挑有利模型的答案 |
| low-budget judge 人评 | 12/12 | judge qualification、人机一致性、回复质量误差分析 | SupportNeed target；由胜出回复反推用户需求 |
| role-decomposed judge 人评 | 12/12 | judge semantic shape、risk taxonomy、回复偏好分析 | SupportNeed target；自动 response gold |
| strict-judge normalized copy | 12 行派生副本 | 原 judge bakeoff 聚合 | 当作新增 12 条人评 |
| factorized component-effect 人评包 | 0/32，空模板 | 完成后校准 component-effect labeling functions | 当前训练；SupportNeed target |
| Strategy Bank V2 人评包 | 0/5，空模板 | 完成后审核 technique、适用阶段和负担风险 | 当前训练；用户需求或 PM effect gold |
| generation semantic review | A/B 均 0/9，空表 | 完成后作为 synthetic surface gate | SupportNeed、judge 或 PM 训练标签 |
| SupportNeed confirmation | 0/8，仍封存 | 模型族和阈值冻结后的一次确认 | 当前选模、调参或错误分析 |

未来 V2 保留项目中另有四版 generation reviewer A/B 表和一组 40-pair bank
mechanism reviewer A/B 表；逐项检查均为未填写模板，完成行数为 0。它们不构成历史
人工证据。

## 3. 第二批双人评如何使用

两位评审在 16 条上的原始一致性为：

- support mode：12/16，Cohen κ = 0.6701；
- dialogue phase：13/16，κ = 0.7037；
- nonclinical urgency：12/16，κ = 0.5188；
- response burden：15/16，κ = 0.8873；
- goals 完全一致：6/16。

分歧不是可以丢弃的噪声。原始 A/B 文件逐字节保留，合议 trace 同时保存两份原判断、
逐字段分歧、合议理由、证据引文的并集与删除项。最终 scalar/goals 必须来自 A 或 B，
显式边界引文只能从两位评审已经提交的 user quote 中删选，不能在合议时新造证据。

若具体选项不可行、用户拒绝某一种做法或表达现实约束，不能自动提升为
`advice_rejected=true`；它只在用户明确拒绝建议这一交互类型时成立。这是本轮删去 9
项过宽边界证据的主要原因。

## 4. 为什么 judge 人评不能“顺手转成”需求标签

两组 judge 人评都向标注者展示候选回复，并要求比较哪一个回应更好。此时“回复内容、
使用了什么记忆/策略、产生了什么问题”已经是 treatment/post-treatment 信息。
从“B 比 A 好”反推“用户原本需要 explore/light guidance”会把候选质量、资源是否匹配和
当前需求混为一体，也会泄漏 SupportNeed 模块在真实推断时看不到的信息。

这些 24 条 judge 状态仍然很有价值：可以只抽取其**对话状态**，隐藏候选回复、评审结论
和所有资源结果，重新用当前 factorized SupportNeed rubric 做一轮新标注。届时产生的是
新的 need annotations；旧偏好本身仍不转标签，也不与新标注算作两个独立状态。

逐个 canonical visible-state 比较后，这 24 个状态与当前 40 个 SupportNeed 状态的精确
重叠为 0，low-budget 12 条与 role-decomposed 12 条之间的重叠也为 0。它们因此可以
组成一个 24-state 的新 reannotation 候选池，但必须标记为
`response-difference-enriched active-learning pool`：原抽样曾服务于候选回复/judge
诊断，不是自然对话的无偏随机样本，不能用于估计 support-mode prevalence。

该候选池现已按此规则实际制成新盲包，并完成 candidate/repro 五文件逐字节复现：

- 页面：
  `outputs/pm_v1_5_support_need_historical_reannotation_v1_candidate/human_blind_review.html`
- binding：
  `data/pm_v1_5_contracts/support_need_historical_reannotation_packet_v1.json`

新 blind packet 只有三个 visible-state 字段，旧 candidates、selected/authorized
context、旧人评与偏好/risk label 全部未复制；24 个新 blind ID 也不沿用旧 judge ID。
完成新标注后，每个 state 只增加一个 group，仍不得用于 prevalence 或自动 action gold。

## 5. 可复现证据

机器可读审计：

- `outputs/pm_v1_5_human_annotation_asset_audit_v1_candidate/report.json`
- 文件 SHA256：
  `4d19710b4f1dcd5628fd71d322b9a8914acb77bcd27118d058a1850fe85a22fe`
- 内容报告 SHA256：
  `61651068d5de6d097af9e1fa8ac95b33185d38b6beaee53be53982d6d0c69cce`
- candidate 与 repro-check 逐字节一致。

审计仅检查当前磁盘保留资产；已删除的早期 archives 不从日志或记忆中伪造重建。没有
API 调用，没有打开 internal/external outcome，也没有打开 8 条 confirmation。
