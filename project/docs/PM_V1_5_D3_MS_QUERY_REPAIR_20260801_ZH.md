# PM V1.5 D3 MS 检索修复与替换报告（2026-08-01）

## 结论

本轮不改变论文最终的 quality-risk-cost 目标，但修正了 Step 1 的直接监督构念。原 D3 的 RS、MP、ME 各 40 对继续保留为系统效果证据；原 MS 40 对降为诊断证据，不能进入正式路由训练。原因不是人评结果“不够好”，而是生成前的客观检索审计发现：原 MS query 同时读取 seeker 和 supporter 文本，supporter 侧用于构造低 nonredundancy 的“你已经说过 prior summary”提示反过来污染了摘要排序，导致 15/32 个 MS state 的候选包含无关通用周记。

已建立有界替换：32 个全新、内容不重叠的内部用户，32 个 MS primary 对和结果前冻结的 8 个 repeat 对，共 40 对、80 次生成。80/80 已完成，唯一 40 对盲评页已经生成。

## 2026-08-01 构念纠正：不再用单次回复胜负定义 PM 是否调用正确

此前本文把两个不同问题合并成了一个：

1. PM 在回复生成前，是否正确识别到一个合格、非冗余、当前可用的 MS 机会；
2. 一次随机生成的 MS-on 回复是否在人类 A/B 偏好中实质胜出。

第二项受 generator 是否真正利用证据、回复随机性和评审偏好影响，不能反向替代第一项。
但第一项通过也不自动等于系统有收益。因此现在固定为五层独立测量：

| 层 | 回答的问题 | 当前角色 |
|---|---|---|
| PM routing | 此时是否存在允许开启 MS 的资源机会 | Step 1 学习目标 |
| retrieval | 是否找到了正确用户、严格过去、相关且有增量信息的摘要 | 独立检索资格 |
| execution | action bit、selected ID 与 prompt 注入是否一致 | 机器确定性审计 |
| grounded use | generator 是否正确使用，或在不必要时合理忽略证据 | Step 2 利用审计 |
| system outcome | 整体 quality-risk-cost 是否优于基线 | 最终验证，不作单条 PM gold |

机器可读合同为
`data/pm_v1_5_contracts/resource_routing_measurement_layers_v1.json`。

新 80 次生成完成后的机器审计已经证明：32 个独立状态、40 对、80 个调用全部按冻结
动作执行；treatment 只增加 MS，control 未注入 MS，所有 selected MS 均进入 treatment
prompt，80 个输出完整。该结论只证明检索与执行正确，不能冒充 PM 已学会或 generator
已利用。

进一步检查其他已开启组件后发现，原 16 个正机会中有 3 个已被 ME 注入的内容逐字或
包含式覆盖，因此对完整 action 并非真正增量机会。最终 routing target 为 13 on / 19 off。
随后训练了一个不读取回复、judge 或数据集身份的 L2 logistic router。模型输入不包含等同答案的
`candidate_incremental_alignment_score`，只包含候选匹配、是否重复、是否带既往结果、
当前是否能利用连续性、相对已开启资源是否重复和其他组件背景。八个内容主题逐一留出时，
五个预定 seed 的 balanced accuracy 均为 `.9211`、positive recall=`1.00`、on/off 各16，
Brier=`.0508`，优于 prevalence prior 的 `.2412`。

这个结果只能证明**受控基础开关任务可学**：当前正负构造清楚，且16个预测开启中仍有
3个相对于ME重复的false-on；确定性跨资源去重门可在运行时关闭它们，但不能把门后的
100%冒充纯模型准确率。下一步不是继续制造主观单次胜负标签，而是把冻结 router
带到真实内部 catalog 和 EvoEmo catalog，分别审核 routing、retrieval 与 grounded use，
最后才报告 aggregate quality-risk-cost。

## 修复了什么

共享 query 合同改为按来源构造：

- MP：保留完整可见上下文；
- MS：只使用 seeker 写出的当前对话和当前文本；
- ME：保留完整可见上下文。

内部替换数据与 EvoEmo 正式运行路径现在调用同一个 `source_specific_memory_queries` 和同一个逐来源 candidate discovery。PM 仍在候选发现后、注入和生成前决策。检索器仍是冻结的 lexical/cosine-style token overlap 实现，不把本次修复冒充 BGE 或新语义模型收益。

## 生成前客观资格结果

粒度为 32 个 state、每个 state 检索 MS Top-2：

- Top-2 非空：32/32；
- Top-2 两条均与当前主题直接相关：32/32；
- 被选 item 精度：64/64 = 1.00；
- 当前用户归属正确：32/32；
- 全部严格早于当前 session：32/32；
- 任意替换 supporter 文本后，MS query 不变：32/32；
- 任意替换 supporter 文本后，MS Top-2 不变：32/32；
- 与 ESConv 精确规范化文本重叠：0；
- 与 EvoEmo 精确规范化文本重叠：0。

这些是构造期可客观核验的 retrieval qualification，不需要心理支持质量人评，也不读取任何生成结果。

## 实验设计仍然回答什么

检索资格只回答“调用 MS 时拿到的是不是当前用户、严格过去、与当前状态相关的 session summary”。它不回答“把这两条合格摘要注入后，冻结 generator 的当次回复是否实质更好”。后一个问题仍需同状态 on/off pair 作为系统验证；否则只能证明检索与路由按定义做对，不能证明整条资源路径改善了最终质量—风险—成本平衡。

因此训练目标现已纠正为：

`P(存在合格、非冗余且当前可用的 MS 资源机会 | 当前可见状态、候选 descriptor、MP/ME/RS 背景)`。

配对生成另行估计系统条件效应；它不再为每个 state 直接提供 PM hard gold。

32 个 primary state 覆盖其余三组件的 8 种背景，每种 4 个用户。当前 router 使用 8 个预生成特征：候选匹配、三个原子增量条件、跨资源非重复和三个显式背景 bit；不看用户 ID、主题标签、回复或人评结果。运行时仍保留完整 16 动作，MS 只是四个独立 head 中被局部修复的一个。

## 为什么这符合“平时学习与考卷”

内部和外部不会共享私人记忆内容，但必须共享能力接口。当前共同链路为：

`MS_SESSION ontology → strictly-prior supplied-summary compiler → seeker-only query → same lexical Top-2 retriever → same descriptor → post-candidate PM gate → same injection prompt → same generator`。

内部训练用内容独立的类纵向用户覆盖这一接口；EvoEmo 测试时只换成其各自用户的真实私有历史，不换 compiler/query/retriever/决策时点。这样 PM 学的是“面对什么状态和实际候选时允许注入”，而不是记住内部用户或某批固定摘要。外部仍需报告真实候选上的 routing 覆盖率、检索正确性与 OOD/abstain，不能用内部受控 `.9211` 代替外部证据。

## 当前冻结资产

- 客观资格报告：`outputs/pm_v1_5_d3_ms_replacement_step0_v1/preflight_report.json`
- 逐 state 检索证据：`outputs/pm_v1_5_d3_ms_replacement_step0_v1/retrieval_qualification_audit.jsonl`
- 32 个 primary：`outputs/pm_v1_5_d3_ms_replacement_step0_v1/d3_ms_replacement_primary_pairs.jsonl`
- 8 个 repeat：`outputs/pm_v1_5_d3_ms_replacement_step0_v1/d3_ms_replacement_repeat_pairs.jsonl`
- 80-call 冻结计划：`outputs/pm_v1_5_d3_ms_replacement_generation_v1/generation_plan_report.json`
- 研究合同：`data/pm_v1_5_contracts/d3_ms_retrieval_replacement_v1.json`
- 五层测量合同：`data/pm_v1_5_contracts/resource_routing_measurement_layers_v1.json`
- 新80-call检索/执行审计：`outputs/pm_v1_5_ms_routing_retrieval_execution_audit_v1/report_evidence.json`
- MS受控路由器结果：`outputs/pm_v1_5_ms_opportunity_router_fit_v1/fit_report.json`
- MS受控路由器模型：`outputs/pm_v1_5_ms_opportunity_router_fit_v1/ms_opportunity_router.joblib`

冻结计划逐 pair 使用同一 seed，control/treatment 只增加 MS bit，消息 hash 不同；预计费用上界约 0.019183 美元。runner 在载入项目 secrets env 后完成 80/80，call ID 唯一、两臂完整、finish reason 全部 complete。40 对中有 2 对两臂输出逐字相同；这是合法的零效应观测，未重生成、未删除，盲评时应判 tie。

唯一新增盲评页面：

`outputs/pm_v1_5_d3_ms_replacement_blind_v1/human_blind_review.html`

## 旧 paired-effect 停止规则（历史记录，已由第9节优先修订）

1. 只执行这 80 次，不扩 RS/MP/ME，也不新增卡片或换 embedding。
2. 完成后生成一个 40 对盲评页；这是本轮唯一新增质量判断。
3. 只对 MS-on 实质胜例做最小 interaction-and-grounding risk 复核。
4. 合并数据时删除原 D3 MS 40 对，只保留原 RS/MP/ME 120 对与新 MS 40 对。
5. 四个 head 按 user-grouped OOF 和冻结门一次训练；不因结果不好搜索 BAAI/Qwen、C、seed 或阈值。

## 质量盲评后的测量修正

40 条 visible-only 质量标注本身完整且一致：解盲为 MS-on 6胜、MS-off 1胜、tie 33；8个repeat严格一致7/8。它不能直接全部进入训练，因为5个以 `visible_context_fidelity` 决胜的条目没有向评审者展示已授权的私人历史。逐项核验发现，部分被写作“无依据”的长期关系、分手后重建日常、新城市等内容实际由同一用户的 MP/profile 证据直接支持。问题属于评价工具缺少必要证据，不属于标注员错误。

修复范围固定为全部且仅这5条，不按 treatment 胜负选择：

- 原35条非 fidelity 决策冻结不动；
- 5条原决策先视为 `unknown_pending_grounded_adjudication`；
- 补充页展示两臂所用 memory 的同用户、严格过去证据并重新随机化 A/B；
- 原决策、原 blind ID、组件、arm、memory ID均隐藏；
- 评审不得奖励“提到记忆”，只需确认事实是否有据并比较整体实质质量；
- 新裁决替换原5条，不叠加样本数。

补充页面：`outputs/pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1/human_grounded_review.html`。

两份补充意见对证据真实性结论一致，但对是否达到“实质更好”有分歧。正式采用具备完整逐行JSONL且严格执行 material threshold 的5条全tie裁决；另一份A3/tie2只作敏感性说明。还发现5条中 treatment 恰好全部位于A，因此A3结果存在无法与位置偏好分离的额外风险；全tie裁决不产生臂方向偏差。正式5换5聚合后，40对为MS-on/MS-off/tie=`1/1/38`，8个repeat严格一致7/8；按32个state等权合并为31个target=0和1个target=.5。只剩1个MS-on质量胜例进入最小risk审核，正式训练仍未授权。

## 最终风险审核与训练结论

唯一 MS-on 质量胜例的 interaction-and-grounding risk 审核为 `no material risk`。逐字与哈希血缘检查确认，该候选就是原盲评 pair 的 treatment 输出；它来自冻结的35条非-fidelity项目，而不是后来补裁的5条，所以与补裁页面中的另一条回复措辞相似但不逐字相同，不是串接错误。

这个 `0/1` 只能解释为“质量胜出的 MS-on 回复中，审核到的 material risk 为0”。它不是 MS 所有开启回复的风险率，也不是 MS-on 与 MS-off 的风险差；未胜出的输出没有进入此项最小风险审核，分母不能外推。

风险门通过后，最终40对和32个state的数值不变：31个state target=`0`，1个state target=`0.5`，没有target=`1.0`的稳定正例。因此本轮已经证明：共享的 seeker-only Top-2 检索能找到正确、相关、严格过去的同用户摘要；但没有证明“注入这些摘要通常改善回复”，也没有形成足以学习条件开关的正例支持。旧 treatment 下不训练一个伪装成 learned policy 的 MS 判别head；其确定性关闭只是修复完成前的证据保护回退，不是最终设计。完整16动作合同仍保留。

后续零API根因审计进一步确认，问题不是MS必然无用，而是本批把“同主题相关”错当“对当前回复有增量价值”：64/64摘要都相关，40对仍有38个tie；旧 grounding 两档各自都是19个tie加1个非tie，它主要受supporter元话语影响，不能表达摘要是否提供了当前文本之外的既往结果、区别或有效做法。已冻结一次有界的incremental-value repair：32个新用户，16个有用非冗余prior-outcome候选、16个相关但无增量候选，四类当前请求与8种背景完全交叉，仍为32 primary+8 repeat、80 calls。新透明特征只读seeker当前文本与实际候选，supporter改写不改变分数；prompt继续允许忽略无用记忆，不强迫提及。至少形成8个post-risk正例和8个非正例，才训练MS head。

最终资产：

- 风险标注：`outputs/pm_v1_5_d3_ms_on_win_risk_review_v1_candidate/risk_annotations_formal.jsonl`
- 最终pair标签：`outputs/pm_v1_5_d3_ms_final_effect_labels_v1/final_pair_effect_labels.jsonl`
- 最终state soft targets：`outputs/pm_v1_5_d3_ms_final_effect_labels_v1/final_state_soft_targets.jsonl`
- 封账报告：`outputs/pm_v1_5_d3_ms_final_effect_labels_v1/finalization_report.json`
- 非使用根因审计：`outputs/pm_v1_5_d3_ms_nonuse_root_cause_v1/report_evidence.json`
- 增量价值修复合同：`data/pm_v1_5_contracts/ms_incremental_value_repair_v1.json`
