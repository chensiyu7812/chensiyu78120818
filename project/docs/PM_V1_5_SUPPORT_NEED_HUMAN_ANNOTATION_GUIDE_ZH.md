# PM-v1.5 SupportNeedObservation 人工标注指南

> **2026-07-28 V1.5 快速路线说明：** 已完成的 40 条/39 non-abstain group 继续作为
> train-only 特征诊断；历史 24 条 reannotation packet 与封存的 8 条 confirmation
> 不再是 V1.5 发布前置任务。除非转入 V2.0 的完整 SupportNeed 研究，不需要继续标注。

## 1. 这份标注在判断什么

标注对象是用户**此刻需要怎样被支持**，不是给 PM 直接指定动作，也不是判断应否
开启 RS、MP、MS 或 ME。资源是否存在、资源是否匹配以及使用后是否改善回复，会由后续
独立的 opportunity/effect 数据判断。

每条记录只显示当前用户话语、最近对话和当前会话摘要。不得根据题目来源、ESConv
metadata、未来 supporter 回复、strategy annotation、survey、internal-test 或 external
outcome 推测答案。

## 2. 固定判定顺序

1. **明确边界。** 用户是否要求只倾听、拒绝建议、明确请求建议、要求一个小步骤，或
   暗示此时只能承受一个问题/任务。
2. **当前目标。** 用户主要想被听见、理解发生了什么、先稳定下来、作决定，还是采取
   行动。
3. **对话阶段。** 当前仍在探索、需要安慰稳定，还是已经准备行动。
4. **非临床紧迫度。** 只判断本轮支持的紧迫程度；不得作疾病、诊断或危机分类。危机
   safety policy 独立于小样本 PM。
5. **不确定就 abstain。** 信息不足或两个 mode 无法可靠区分时，不要勉强制造标签。

## 3. support mode 的可观察定义

| mode | 选择条件 | 不应选择的情况 |
|---|---|---|
| `listen` | 用户主要想说出来、明确要求先听、当前不宜推进解决方案 | 用户明确请求澄清问题或实际步骤 |
| `explore` | 一个低负担问题能澄清感受、事实或目标 | 连续追问、多问题盘问，或用户明确要求不追问 |
| `comfort_reassure` | 当前先需要承接、安慰、正常化感受或降低情绪负荷 | 用空泛赞美替代对具体感受的回应 |
| `light_guidance` | 用户愿意接受一个可拒绝、低门槛的小建议 | 用户只要倾听，或问题需要多步规划 |
| `structured_planning` | 用户明确准备行动，且需要共同拆解步骤、权衡或计划 | 尚未搞清目标，或当前负担不允许多步骤 |
| `abstain` | 可见信息不足、冲突或过于含混 | 仅仅因为标注者犹豫；能由明确边界解决时不应 abstain |

“用户请求建议”只是 `light_guidance/structured_planning` 的必要线索之一，不是 RS
标签。即使允许建议，Strategy Bank 也可能没有匹配资源；即使未请求建议，reflection 或
comfort 也可能适用。

## 4. goals、phase 与 urgency

`goals` 可以多选：

- `be_heard`：被听见、被承接；
- `make_sense`：理解感受、事实、关系或冲突；
- `stabilize`：先降低当下负荷；
- `decide`：作选择或明确方向；
- `act`：执行一个步骤或计划。

非 `abstain` 条目至少选择一个 goal。

`dialogue_phase`：

- `exploration`：信息/感受/目标仍待澄清；
- `comforting`：当前重点是承接和稳定；
- `action`：已准备讨论选择或执行。

`nonclinical_urgency`：

- `routine`：普通支持节奏；
- `elevated`：当下明显承压，需要更快、更聚焦地承接；
- `acute`：本轮应优先稳定并降低负担，但这仍不是临床诊断或危机标签。

## 5. 五个显式边界字段

字段可填 `true`、`false` 或 `unknown`。只有文字明确支持时才填 `true/false`；信息不足
填 `unknown`。

- `advice_rejected`：用户明确拒绝建议或只要求倾听；
- `advice_requested`：用户明确请求建议、帮助决定或规划；
- `one_small_step_requested`：用户明确只要一个小而可做的步骤；
- `listen_first_requested`：用户明确要求先听/先理解；
- `question_or_task_burden_limit`：此时最多适合一个简短问题或一个小任务。

## 6. 质量要求

- `confidence` 为 1–5；3 以下必须在 `notes` 说明歧义。
- 不使用身份、problem/emotion 标签或未来答案作推断。
- 不把情绪强度等同于建议需求。
- 不把“有可用记忆/策略”当作已知；本包不展示这些资源。
- 不为追求类别平衡而改变单条判断。
- 人工锚点是弱监督可靠性与 sanity 参照，不是自动 gold，也不单独决定 PM 动作。

## 7. 第一批标注后的语义修正

第一批 24 条标注显示，旧字段 `question_or_task_burden_limit` 实际主要被用来表达
“标注者认为下一条回复应保持低负担”，而不是“用户逐字明确说只能承受一个问题/任务”。
二者不能共用一个 target：

- **用户明确边界**是可观察输入，必须能给出可见 user 文本中的精确引文；
- **建议回复负担**是上下文相关的人类 partial label，可以是
  `minimal_presence`、`one_focus` 或 `multi_step_ok`；
- 没有找到明确引文表示“没有已核实的显式证据”，不得自动编码成用户明确说了 false。

第一批原始标注保持逐字节不变，并通过 normalized V2 binding 保留。扩展包采用上述新
schema，不要求返工第一批；两批都不是 PM action gold。

## 8. 第二批因子化补充标注

2026-07-28 的同一 23-dialogue-group 表示资格赛显示：

- “是否推进”和“是否需要探索”已有可复现但仍弱的双向 NLI 信号；
- “是否适合一个聚焦问题”和“是否可给建议”的最低-loss view 仍漏掉全部正类；NLI
  只能以更高 loss 各换回一个正例，planning 正类仍全部漏掉；
- Qwen3 instruction-aware embedding 没有整体胜过 BGE/NLI，不能靠继续换大模型代替
  新鲜独立锚点。

因此第二批只开放预先冻结的 16 条 `expansion_fit`，8 条
`untouched_confirmation` 不出现在标注页面。标注者不需要知道每条的角色，也不得刻意
制造平衡；仍按第 2 节顺序逐条判断。特别留意以下区别：

1. 想“理解发生了什么”不自动等于当前适合追问；用户也可能先需要被听见或安慰；
2. `explore` 只在一个聚焦、低负担问题确有必要时选择；
3. `light_guidance` 表示一个可拒绝的小建议；`structured_planning` 要求用户已能承受
   多步骤共同拆解；
4. `recommended_response_burden` 独立填写，不由 support mode 机械推出；
5. 无逐字明确边界时不填引文，不能把 assistant 的问题当作 user request。

当前 fit-only 标注页为
`outputs/pm_v1_5_support_need_factorized_fit_packet_v1_candidate/human_blind_review.html`。
导出的 JSONL 仍须通过 exact coverage、abstain coherence 与 user-only quote 校验；完成
后只是训练/校准 partial anchors，不自动授权 representation promotion、formal fit 或
PM action。

## 9. 第二批完成后的合议与历史资产边界

两位评审已各自完成全部 16 条 fit-only 记录。原始表均通过 schema、exact coverage 和
user-only quote 校验；support mode 一致 12/16，phase 一致 13/16，urgency 一致
12/16，burden 一致 15/16，goals 完全一致 6/16。

训练时不能把两份表展开成 32 个 dialogue group。正式输入是逐条保留 A/B 原判断并经过
受约束合议得到的 16 行：

- scalar 与 goals 只能在 A/B 已提交的答案中选择；
- 显式边界证据只能从两份原始证据中删选，不能补造；
- 特定方案不可行或拒绝某一个选项，不等于全局 `advice_rejected`；
- 所有分歧和删除证据保留在 adjudication trace；
- 8 条 `untouched_confirmation` 继续封存。

第一批 24 行与本批 16 行合并后共 40 行，其中 39 行 non-abstain；每个状态仍只贡献
一个独立 group。low-budget judge、role-decomposed judge、Strategy Bank review、
generation semantic review 和 component-effect review 的任务对象均不同，不得并入
SupportNeed fit。完整清单和允许用途见
`docs/PM_V1_5_HUMAN_ANNOTATION_ASSET_AUDIT_ZH.md`。

## 10. 历史 judge 可见状态重新盲评

两组历史 judge anchor 各有 12 个 visible state，与当前 40 条 SupportNeed state 零精确
重叠。它们原本展示过两个候选回复及 memory/strategy，因此旧偏好、风险判断和“哪个回复
更好”绝不能变成 need 标签。新包已执行以下隔离：

- 只复制 `current_user_text`、`recent_dialogue` 与 `session_summary`；
- 不复制 candidate response、selected context、authorized context、action 或旧标注；
- 为每条生成新的 `need_reann_*` blind ID；
- private lineage 只保存来源任务与“不复制哪些字段”的审计事实；
- 24 条只是 response-difference-enriched active-learning pool，不代表真实 mode 比例。

标注仍完全使用本指南第 2–6 节与第二批相同的 V2 schema。尤其不要猜“旧系统为什么选中
这条”，也不要为了补 planning 正类而把尚未准备行动的用户标成 structured planning。

当前页面：
`outputs/pm_v1_5_support_need_historical_reannotation_v1_candidate/human_blind_review.html`。
导出文件必须重新通过 exact coverage、abstain coherence 和 user-only exact quote 校验。
绑定：
`data/pm_v1_5_contracts/support_need_historical_reannotation_packet_v1.json`。
完成后只增加 24 个新 state group；不复用旧 judge label、不估计 prevalence、不自动授权
formal fit，也不打开 8 条 confirmation。
