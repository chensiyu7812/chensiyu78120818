# PM V1.5 V5.3：由三项外部考卷反推系统与训练数据（2026-08-07）

## 结论

直接读取 ESConv、EvoEmo/ES-MemEval 原始数据、冻结 split/surface 和 W7R 蓝图后，结论是：

1. 外部考卷需要的 PM、检索和执行链路已经足够清楚，不再缺一个抽象研究定义；
2. “旧内部训练数据缺少纵向深度”是真的，但不是唯一问题；更准确地说，缺的是**纵向历史深度、候选竞争密度、跨会话演化和自然交互轨迹**；
3. W7R 仍不合格。除深度不足外，还存在题干泄露答案、合法运行时特征缺失、condition 与 family 混杂、group 膨胀和多组件交互缺失；
4. 外部数据只能规定部署支持范围和考试结构，不能直接提供 `worth_opening` 标准答案。Step1 的监督结果仍必须来自冻结 Step2 下同状态 ON/OFF 的质量—风险—成本差异；
5. 下一步只做一次方法级 P2R，静态复审通过即冻结并进入正式 paired generation，不再继续围绕表面短语做小包循环。

机器可复跑报告：

`outputs/pm_v1_5_v5_3_external_exam_backward_training_spec_v1/report.json`

复跑脚本：

`scripts/v1_5/77l_profile_external_exam_backward_training_spec_v1_5.py`

本审计 API=0，未读取生成回复、质量、risk、judge 或人评 outcome。

## 1. 三张外部考卷的真实结构

### 1.1 ESConv：考自然即时互动和 RS，不考长期记忆

原始数据有 1,300 个 dialogue，单段 16–120 turns，中位数 27；共有 18,376 个 supporter turns，包含 Question、Providing Suggestions、Reflection、Restatement、Affirmation 等八类原始策略标注。当前 V5.3 自定义 split 中有 169 个排除 EvoEmo 同源后的 eligible test dialogues。

这意味着 ESConv 的核心不是识别一句显式的“请问我一个问题”，而是根据多轮轨迹判断：

- 上一轮助手刚做过什么；
- 用户现在是在继续表达、要求聚焦、拒绝建议，还是已经给出明确问题；
- 某张 RS 卡的动作是否重复、越界或增加负担；
- 没有必要动作时能否保持 `M0+R0`。

训练数据要求：自然多轮话语、previous assistant move、边界和重复负例。把 current turn 写成“Please ask exactly one focused question”只能教会模型读机器指令，不能运输到 ESConv。

ESConv 没有可因果归属的跨会话私有记忆，因此 MP/MS/ME 必须结构性 mask。它只验证 RS 子域。

### 1.2 EvoEmo response：考同用户长历史中的选择与执行

原始 artifact 有 18 个合成长历史用户：

- 401 个历史 sessions，每用户 13–33；
- 446 个 events，每用户 13–37；
- 150 条 social relationships，每用户 3–15；
- 126 个 basic profile facts，固定为 name/age/gender/job/education/nationality/location；
- session summary 为 12–37 words，中位数 25；
- 没有 response-preference 型 MP 项目。

因此它要求的不是“候选池里放三条几乎独立的句子”，而是：

- 只读同一用户、严格过去的历史；
- 在 13–33 条规模中面对同主题但不同事件、不同时间、已解决/未解决、不同人物的干扰项；
- MP_PROFILE 只在确实改变建议范围、时机或负担时才开；
- MS 能找回具体旧观察、目标、区分或未完成线程，而不是只找同主题摘要；
- ME 只有在 exact Rank-1 含过去动作及结果/机制、当前允许行动且不冗余时才可能有价值；
- Step2 必须真正吸收并自然使用证据，而不是追加记忆说明。

EvoEmo 没有 MP_PREFERENCE，因此该子构念必须由内部受控实验验证，不能把外部缺席解释为 PM 失败。

### 1.3 ES-MemEval QA：考多证据历史推理，不是同一张 response 考卷

公开 v1.0.0 artifact 有 1,427 题：

- information extraction：309；
- temporal reasoning：284；
- conflict detection：267；
- user modeling：306；
- abstention：261。

每题 evidence 数量均值 2.34、中位数 2、最大 20；902 题至少需要两条不同 evidence，553 题至少三条，151 题至少五条。258/261 abstention 题没有 evidence。

因此 QA 链路必须支持：Top-k 多证据整合、时间与冲突推理、用户建模和无证据拒答。单一 exact Rank-1 response candidate 无法独立承担这项任务。

正确分责是：

- shared private store、owner/time isolation 和 retriever 与 response 系统共用；
- QA adapter 负责 Top-k 上下文组装和短答案；
- response PM 只作有限跨任务路由压力测试；
- QA answer/evidence 不进入 response Step1 的 paired-effect 训练标签。

## 2. 外部结构反推出的训练数据最低规格

### 2.1 所有组件共享

每个正式 counterfactual group 必须具备：

1. 自然 current utterance，而不是对生成器下命令；
2. same-user、strictly-past、无未来/跨用户资源；
3. candidate pool 具有真实密度和同主题 hard negatives；
4. 当前可见消息不包含应检索的历史答案；
5. Step1 只看运行时可观察特征，construction target 留在 audit-only；
6. 每个内容 family 内都交叉出现 positive、decline、redundant、goal-mismatch，而不是新 family 只有 positive；
7. 同一 user/family 的近反事实变体绑定同一 split group；
8. label 由冻结 Step2 的 paired ON/OFF effect 产生，不由人直接写“应该开”。

### 2.2 MP

- `field_type/value/owner/time/subtype` 结构化保存；
- 同一 profile fact 与 relevant、irrelevant、already-restated、stale/changed current goal 交叉；
- candidate presence 不依赖当前消息复述 field value；
- MP_PROFILE 对齐 EvoEmo；MP_PREFERENCE 保留为内部专门能力。

### 2.3 MS

- 每用户 13–33 条严格过去 catalog，而不是恒定三候选；
- 同主题不同事件、人物、状态、时间和结局的干扰项；
- current query 只表达“接回上次目标/提醒具体观察”，不写出答案；
- 保存 BGE Rank-1、Top-k、margin、age、resolved/conflict 和完整 lineage。

### 2.4 ME

- exact Rank-1 必须可编译为 past action + result/mechanism，否则该状态 unavailable；
- invite/decline/unknown readiness、redundancy、goal fit、age/margin/cost 均为合法运行时特征；
- intended target ID 与 `rank1_matches_intended_target` 只用于审计；
- 同主题但不同动作/结果的合法竞争项必须存在；
- 不要求在 EvoEmo 中强行制造高覆盖，候选缺席本身也是部署分布的一部分。

### 2.5 RS

- 使用冻结 6-card Bank；
- natural multi-turn dialogue、previous move、card precondition、already executed、burden 和 explicit stop；
- `M0+R0` 是合法对照，不是失败 fallback。

### 2.6 多组件

必须有真实 MP+MS、MP+ME、MP+RS、MS+ME+RS 和四组件候选共同存在的状态，且各 baseline 使用完全相同的 candidate identity。这里不要求 16 动作等频，只要求联合 projection 和冲突处理有真实训练支持。

## 3. 为什么 W7R 仍不合格

W7R 的 schema/lineage 修复可以保留，但对上述外部支持范围仍有六个阻塞：

1. MP preference 正例 16/16 已在 current turn 说出全部行为增量，只因候选缩成两个词而绕过三词冗余启发式；
2. MS continuity 正例 16/16 已经给出历史答案，候选池恒为 3，且只有 5/16 Rank-1 为构造 target；
3. ME 40/40 的合法 Step1 运行时特征被删空；
4. 8 个新增 ME family 只有 positive，condition 与 family 混杂；
5. MP/MS 按 state 拆 group，48/32 个表面 group 实际仅各 16 个用户簇；
6. RS 缺 contribution slots，5 个 interaction 全部没有 MP。

所以“纵向复杂度不够”是正确诊断的一部分，但只扩大候选池仍不足以修好训练数据。必须同时修复可观察特征、反事实条件、group 和 interaction。

## 4. Leader/Worker 接下来的唯一顺序

### 并行阶段，零 API

Worker 只构造一个独立的、外部文本零复制的 longitudinal catalog asset：

- 13–33 条严格过去 session/episode；
- recurring-topic hard negatives、resolved/conflict/owner/time 变化；
- 结构化 MP_PROFILE、MS observation、ME action-result；
- 不创建 current state、不计算 Step1 特征、不生成 label、不修改 runbook/机器合同/失败账本。

Leader 同时负责：

- P2R schema 和 state materializer；
- 把 MP/MS/ME/RS contribution slots 接回；
- natural current state 与 catalog 解耦；
- counterfactual grouping/split；
- 多组件 interaction 和正式 shortcut/leakage audit。

### 汇合阶段

1. Leader 导入 Worker catalog，生成完整 P2R blueprint；
2. Worker 只读独立复核候选身份、严格过去、池规模、内容泄漏和条件覆盖；
3. Leader 运行一次最终零 API 静态审计；
4. 通过即冻结 N/split/release 和正式 runner；不通过则只允许修机械身份错误，构念仍不成立时缩小主张，不再调表面短语；
5. 用户另行授权后进入一次 P3 paired generation、一次训练和一次 fresh confirmation。

