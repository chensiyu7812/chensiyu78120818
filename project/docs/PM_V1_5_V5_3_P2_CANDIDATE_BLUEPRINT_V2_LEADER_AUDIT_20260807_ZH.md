# V5.3 W7R / P2 候选蓝图 V2 Leader 复审（2026-08-07）

## 结论

Worker commit `b21edc4` 完成了上轮点名的 schema、lineage 和字段隔离修复；这些不是假结果。但 V2 仍不能进入正式训练或 paired generation，因为几项“计数归零”并没有修复对应学习构念。

状态：

`METHOD_AND_BLUEPRINT_REPAIR_REQUIRED_BEFORE_P2_READY`

机器报告：`outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2_leader_audit_v1/report.json`

复跑脚本：`scripts/v1_5/76l_audit_v5_3_p2_candidate_blueprint_v2_v1_5.py`

## 已确认修好的部分

- MP `field_type/field_value/version` 已完整；
- MS 不再是单候选，且保存了真实 BGE semantic score；
- ME 构造者的 intended-target 答案已从 Step1 特征删除；
- RS subtype 与 selection mode 已分开；
- 单组件 surface/version hash 已补齐；
- interaction 的基本 lineage/schema 已补齐；
- external exact overlap 仍为 0，唯一 8-gram 是上一轮已裁决的普通话语片段，不是内容同源。

这些修复可以保留，不需要推倒重来。

## 仍不能训练的原因

### 1. MP 是“绕开冗余计数”，不是产生隐藏增量

16/16 preference positive 的 current turn 仍完整包含存储 preference 的两个实质词。例如存储内容是 `prefers reflection and question`，当前消息也明确要求 `reflection and question`。

之所以 `current_redundant=False`，只是候选被缩短到两个词，低于旧启发式“共享至少三个词”的触发线。因此 14/16→0 是测量规避，不是构念修复。

正确办法不是继续换词，而是改候选层：MP preference/profile 作为同用户结构化候选存在，不依赖当前消息复述 field value 才能被发现；Step1 判断它是否会改变当前回复。

### 2. MS 正例仍把应检索的答案写在当前消息中

16/16 continuity positive 都被真实特征判为 `current_redundant=True`。用户消息直接写出“上次目标是 X”，此时 MS 只是重复答案。更严重的是，BGE 只有 5/16 选中真正 target；11/16 选中的是刻意复制 target 内容的 same-family distractor。

候选池虽然从 1 增至 3，但真实 EvoEmo 候选池为 13–33。V2 仍没有测到外部的检索规模和干扰结构。

正确状态应只说“我想接回上次未完成的那件事/提醒我那条具体观察”，答案只存在于严格过去的同用户候选池中。

### 3. ME 删除答案时把合法题干也删了

40/40 ME row 的 `step1_features={}`。应删除的是 `rank1_matches_intended_target`，不是以下运行时可观察特征：

- candidate 是否含 past action + result；
- 当前 action readiness 是 invite / decline / unknown；
- 当前是否已经说过相同结果；
- relevance、margin、age、token cost。

没有这些信息，ME head 无法学习何时 ON/OFF。

### 4. 独立组数被 state-level ID 放大

V2 报告 MP=48、MS=32、ME=40、RS=13 个 group，但按同一用户/内容 family 聚类后分别只有：

- MP：16；
- MS：16；
- ME：16 个内容 family；
- RS：7 个动作 family。

MP 的同一用户三个变体和 MS 的同一用户两个变体被分成不同 group，未来可能跨 split，产生近重复泄漏。必须按用户/内容 family 重新聚类，而不是把每个 state 叫一个独立 group。

### 5. ME 和 interaction 仍不完整

- 8 个新 ME family 只有 positive；negative/decline/redundant 只存在旧 family，condition 与 family 混杂；
- RS 13/13 缺 `card_precondition_met` 与 `card_already_executed_last_turn` 等合法运行时 slot；
- 5 个 interaction 全部没有 MP，因此还没有真实四资源 interaction state。

## 通俗解释

现在处于“正式出卷前最后校对题目”的阶段：

- 还没正式训练 PM；
- 还没花钱生成 paired replies；
- 还没做人评；
- 因此这不是 PM 训练失败，也不是研究结论失败。

W7R 把表格列、身份哈希和候选分数补齐了，相当于试卷装订正确；但部分题目仍然把答案写进题干，ME 题则把必要条件也删掉了。所以现在停下是节省后续费用和人评，不是回到旧循环。

## 下一步：Leader 接管一次方法级 P2R，不再让 Worker 调表面词句

本轮不再让 Worker 继续围绕审计计数修改短语。Leader 直接完成一次方法级修订：

1. MP：实现结构化同用户候选 surface，使 candidate presence 不依赖当前复述 preference/profile；保留 preference/profile 两子域。
2. MS：构造答案只在历史池中的自然 continuity query，并使用 13–33 条严格过去同用户候选，保存 BGE 完整 lineage。
3. ME：接回 `me_contribution_slots()` 的合法运行时特征，construction target 继续 audit-only。
4. RS：接回 `rs_contribution_slots()` 的 card precondition、已执行排重和负担特征。
5. Group/split：同用户、同 family、近反事实变体必须同组；禁止 state-level group 膨胀。
6. Interaction：构造少量但真实的 MP+MS、MP+ME、MP+RS 及四组件状态，所有 policy 共享 exact candidate identity；不要求 16 动作等频。
7. ME 新旧 family 的 opportunity condition 做内容解耦，避免 positive-only 新族。

完成后只做一次零 API 静态复审。通过即冻结实际 N/split/release、实现 formal runner，然后进入一次 P3 paired generation；不再针对静态表面继续迭代。
