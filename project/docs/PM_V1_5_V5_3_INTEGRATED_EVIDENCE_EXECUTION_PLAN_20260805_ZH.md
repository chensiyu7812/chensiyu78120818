# PM V1.5 V5.3：证据整合执行器、效用路由与一次性再确认方案

状态：`PROPOSED / NEW METHOD VERSION / P1 IN PROGRESS / LAST ALIGNED 2026-08-06`

机器合同：`data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json`

> **版本对齐记录**：本文、机器合同
> `data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json` 与
> `PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md` 是三份权威事实源，分别承载人类计划、机器约束和
> 失败/修复历史。任何会话新增的已验证进展、问题或假设推翻，必须同步回写相应事实源；开始新一轮
> 工作前先核对三者状态，不得凭对话记忆判断进度。`PM_V1_5_V5_3_OPEN_PROBLEMS_AND_SEQUENCE_`
> `20260806_ZH.md` 从本次对齐起只作历史快照，不再作为第四份活跃清单。

## 0. 直接结论

V5.2 的主要外部失败不能靠重新解释旧人评解决。其 `primary_response` 由 Llama 只看当前对话生成，
MP/MS/ME 的资源内容随后由后端逐字追加。这个设计保证资源不会被 LLM 改写，却也保证了 LLM 无法
吸收、选择、桥接或围绕资源组织回复。EvoEmo 中 learned PM 的 48 个核心状态有 46 个包含 MS，
而 `MS+R0/MS+RS` 几乎全部输给 always-off；这是 Step1 过度开启与 Step2 机械追加的联合失败。

V5.3 不改变研究问题：仍检验一个低容量、可审计的四资源 PM，能否在不明显降低即时回复质量、
不增加明确 interaction-and-grounding risk 的条件下，减少不必要的资源注入和生成成本。四个 bit、
16 个合法动作和 `M0+R0` 全部保留。

不能预先保证新结果一定通过统计门；能够保证的是：

1. 不再把资源机械粘到一段与资源无关的开场白后；
2. Step1、retrieval、Step2、generator 和 evaluator 的责任可逐层定位；
3. 所有主 baseline 使用同一候选、同一执行器、同一 generator 和同一评测栈；
4. 训练、确认和外部复现一次性版本化，结果不再回流形成“小包—修 prompt—再评”的循环；
5. 即使没有通过，也会得到可发表、可复现的明确失败边界，而不是新的模糊状态。

## 1. V5.2 证据的最终身份

### 1.1 继续有效

- V5.2 的候选检索、owner isolation、Bank、四组件 ontology、16 动作 compiler、外部数据版本与成本账本；
- V5.2 内部和外部结果作为“backend-locked 机械追加架构”的有效诊断；
- ES-MemEval 的官方 QA、F1/BERTScore、341 道可映射题的 R@4/nDCG；
- 旧人评对质量、功能和 grounding-risk 定义中已经稳定的 codebook；
- 所有失败账本、版本哈希和暴露记录。

### 1.2 不得继续充当 V5.3 证据

- V5.2 的 `worth_opening` 标签和四个 value head：executor 改变后，ITT 环境已经改变；
- V5.2 的内部 confirmation 与 external E7：只能描述旧系统，不能确认新系统；
- 在 p7–p18 上得到的任何结果：这些分区已经暴露，只能作 repaired replication；
- 旧 `locked_clauses` 的功能率、风险率或质量差，不能直接外推到证据整合执行器。

旧 V5.2 不重做完整人评。只完成已经抽出的独立 risk overlap，用严格口径关闭测量账本；该结果不参与
V5.3 设计选择、训练或晋级。

### 1.3 V5.2 外部失败的校正归因与可直接支持的证据

V5.2 中 `An earlier session recorded ... I am keeping that as past context ...`、
`One current practical constraint on record is ...` 等文本，不是 generator 读到内部指令后复述或幻觉出来的。
`base_generation_messages` 明确不向 generator 提供 MS/ME 正文；这些句子由后端 `_ms_clause`、`_me_clause`
和 `_mp_profile_clause` 编译，并由 `compose_locked_response` 在生成后确定性追加。`locked_response_guard_errors`
还要求 clause 逐字存在。因此准确责任是 **Step2 deterministic renderer 把审计/时态控制模板设计成了用户可见文本**，
而不是 generator 泄漏了其未曾看到的 prompt。

EvoEmo 外部账本可直接支持：

- risk overlap 的25条 EvoEmo 回复中11条出现固定 memory wrapper；主 risk packet 的124条 EvoEmo回复中
  53条出现该 wrapper。它们覆盖 fixed-high、transparent-rule 和 learned，不是 learned 独有；
- wrapper 的出现率不是 LLM 的随机泄漏率。任何含 locked clause 的输出都由代码保证出现 wrapper；上述比例只表示
  评审样本中有多少回复实际启用了相应记忆资源；
- EvoEmo 138个 response state 中旧 learned head 请求 MS 135次（p7–p12为57/60，p13–p18为78/78）；
- 对 always-off 的48个主质量比较中，46个 learned 动作含 MS；在当时的 `Primary_ChatGPT` model panel 下，
  45负、1胜。该结果是强诊断信号，但不是实际临床人员或多人共识 gold，论文必须如实标注 annotator 身份。

上述证据证明机械追加、treatment 暴露和 MS 极高开启率真实存在，但不能把每一个质量负例都只归因于 renderer。
Rank-1 当前目标适配、PM价值判断、开场白质量、资源真实边际价值和评审噪声仍可能共同作用，必须由下述分层责任账本拆开。

## 2. 新的系统边界

固定链路：

`state → Top-k candidate discovery → exact Rank-1 execution candidate → hard eligibility →`
`four Step1 value heads → joint action → typed response program → evidence-aware generator →`
`deterministic trace/grounding guard → final response or M0 fallback → quality/risk/cost`

### 2.1 Retrieval：Top-k 发现，Top-1 执行

- MP/MS/ME/RS 的候选发现和 descriptor 可以保留 Top-k，用于 margin、覆盖、OOD 和成本上界；
- 每个被开启的组件只允许 exact Rank-1 进入本轮执行，避免把 MS 的两条、ME 的三条全塞给 generator；
- 保存 `candidate_ids_topk` 与 `execution_candidate_id`，二者不得混写；
- execution candidate 必须同 user、严格过去、版本有效，并能回溯 literal evidence；
- 检索排序不读回复、judge、gold action 或外部 outcome。

> 实现状态（2026-08-06）：`58_retriever_scale_stress_test_v2_v1_5.py` 已在**构造性合同压力测试**
> 中验证：给定编译器合格、话题对齐的信号候选时，MP/MS/ME 不会仅因真实目录规模而机械塌缩。
> 这只排除了“规模必然导致崩溃”，不等于生产 Rank-1 在自然状态上已经可靠，也不等于 BGE-MS
> 已正式晋级；MS 的真实用户级比较仍不显著且 BGE 目前仅为 opt-in challenger。V1 因构造错误所得
> 的塌缩结论已撤回，详见账本 V15-EVAL-23。

### 2.2 Eligibility：只挡机器可证明的错误

硬 OFF 仅包括：candidate absent、wrong owner、版本失效、已明确冲突、明确当前边界禁止、资源定义不完整。
topic 相似、候选看似有用或“以前讨论过”均不等于值得打开；它们只进入 Step1 因子。

### 2.3 Step1：学“当前执行器下的请求价值”

Step1 不挑具体记忆，不写回复，也不预测复杂心理需求。每个组件分别估计：在候选已存在且没有硬错误时，
由 V5.3 执行器请求该组件，相比不请求它是否具有正的期望边际价值。

新增的关键输入不是更大的通用语义 embedding，而是可审计的 `contribution slot`：

| 组件 | 值得开启所需的可观察槽位 | 典型 OFF |
|---|---|---|
| MP preference | 当前回复动作确实可被该格式/负担偏好改变 | 当前已逐字提出同一偏好；回复没有可改变的形式 |
| MP profile/constraint | 当前目标需要建议或安排，约束会改变可行内容 | 只需倾听/命名情绪；约束与当前目标无关 |
| MS | 用户明确要连续性、旧观察、未完成线程或一个具体区分能回答当前问题 | 仅同主题；泛泛旧摘要；当前已经说完同一内容 |
| ME | 当前欢迎一个行动选项，且候选含过去动作及结果/机制 | 当前不要行动；只有背景事件；结果不可迁移 |
| RS | 卡片的 atomic move 与当前请求/边界匹配且未在上一轮执行 | 动作已执行、重复、负担超限或 `when_to_use` 不成立 |

> 实现状态（2026-08-06）：上表槽位已实现为候选级**原型代码**
> `src/metacom_pm/v1_5_v5_3_contribution_slot_features.py`（commit `611e75e`），最大程度
> 复用已验证组件（`describe_memory_candidate`/`observable_flags`/`compile_atomic_reusable_
> outcome`/`compile_atomic_session_observation`），字段区分 `rank1_*`（仅用 exact Rank-1
> 候选计算，代表真正会被注入执行的内容）与 `topk_*`（Top-k 诊断量，不代表实际注入内容，
> 二者混用曾导致一次需要更正的域间差距虚高，见账本 V15-MEAS-51）。ME 的8个确定性模板 case
> 在把措辞改为正则可识别形式后达到8/8，只证明 schema/规则管线一致，不证明自然语义泛化，
> 也不产生 Effect FIT gold。**尚未做**：四组件 contribution-slot 的独立未见改写资格、MP/MS/RS
> 的跨域支持审计，以及真实 paired outcome 标签。

Primary 仍是四个 source-specific、低容量 L2 logistic value heads。输入包含预注册主效应和少量明确交互：

- `goal_function_fit × contribution_slot_available`；
- `specific_increment × current_redundancy`；
- `past_action_result × current_action_readiness`（ME，三态：邀请行动/拒绝行动/不明确；
  `UNKNOWN` 不得折叠成负例）；
- `continuity_request × specific_prior_observation`（MS）；
- `preference_applies_to_response_act`（MP）；
- `card_precondition × nonredundancy × burden_fit`（RS）；
- candidate age、retrieval margin、预计新增 token。

训练损失使用按 counterfactual group 等权的、未作 class balancing 的二元交叉熵加 L2；不再用
`class_weight=balanced` 后又要求概率击败自然 prevalence Brier。FIT 内只允许预注册的 `C` 小网格，
按 grouped nested CV 的 Brier 与 balanced accuracy 联合选择一次；阈值在 FIT 内按预注册 QRC 约束选择，
随后冻结。BGE 只允许一个冻结 challenger，不替代 primary。

### 2.4 Step2：typed response program，而不是 clause append

Step2 先为动作编译一个机器可审计的响应程序，再让 generator 实现整个回复。程序至少包含：

- `current_goal` 与允许的回复动作；
- 每个 execution candidate 的 `owner/time/source/evidence_id/literal_evidence`；
- `required_contribution`：该资源必须怎样改变本回复；
- `epistemic_mode`：当前事实、过去事实、待核实连续性或可拒绝类比；
- `forbidden_inferences`：当前原因、稳定人格、未出现人物、诊断、效果保证；
- `atomic_move_budget` 和总负担上限；
- `visible_style_constraints`；
- 无法满足时的 `cannot_integrate` 原因。

generator 必须看到上述程序与证据，并一次性输出自然回复和内部 trace。最终用户回复不得再由
`primary_response + locked_clauses` 组成。内部 trace 至少含所用 evidence ID、落实的 response act 和
每个资源的 contribution slot；trace 不展示给用户。

组件执行语义固定为：

- MP preference：只改变行为和表面，不向用户宣读“你的 profile 说……”；
- MP constraint：只有它实际改变建议/安排的范围、时机或负担才算可执行；没有作用对象时不注入；
- MS：以自然的过去来源承接具体旧目标/观察，并保留“现在是否仍适用”的可证伪出口；
- ME：把过去的动作—结果/机制作为一个可拒绝选项，不把一次结果升级为规律或保证；
- RS：只完成一项原子动作，不在后面追加第二个问题、建议或任务。

### 2.5 Guard：只检查机器事实，不假装理解全部语义

机器硬检查：

- requested、candidate、execution ID、owner、version 和 action 完整一致；
- generator 确实收到且 trace 只引用获准 evidence ID；
- 用户可见文本没有 MP/MS/ME/RS、resource ID、`current active practical constraint` 等内部标识；
- 未授权的专名、数字和具体过去事件不得出现；
- RS 原子动作数、列表长度和明确一点式边界可确定时必须满足；
- `M0+R0` 是合法动作，不得误记为 fallback。

语义上是否陈旧、冲突、过度概括或强加意义仍由独立 grounding-risk 评测判断。自然词 `me`、
`before`、`earlier` 不得再作为组件或虚构记忆的硬判据。

若结构/绑定检查失败，返回确定性、无历史声明的 M0 回复并保留为 ITT 结果；不进行第二次自由 LLM 调用。

### 2.6 逐状态责任账本：不再只从最终回复倒推谁出错

P1 必须实现并在任何新 API 生成前冻结 `stagewise_accountability_ledger`。每个 state 至少保存：

```text
row_id, state_id, user_id, semantic_family, counterfactual_group_id,
policy_condition, seed_label, lifecycle_stage,
candidate_ids_topk, exact_rank1_id, candidate_owner_id, candidate_version,
retrieval_fit_label, eligibility_owner_time, eligibility_goal_function,
eligibility_boundary_burden, eligibility_specific_increment,
pm_probability, pm_threshold, pm_decision, pm_correct, requested_action,
realized_action, generator_received_evidence_ids, used_evidence_ids,
generator_reported_used_evidence_ids, normalized_used_evidence_ids,
required_evidence_use, functional_contribution, grounding_fidelity,
atomic_move_compliance, execution_valid, scaffold_exposure,
fallback_reason, quality_outcome, risk_outcome, prompt_tokens, total_tokens,
primary_failure_owner, secondary_failure_owner
```

同一 `state_id` 会有多个 policy/action/seed realization，因此唯一键冻结为
`state_id × policy_condition × seed_label`；不得只按 `state_id` 写一行后相互覆盖。候选、资格、
PM 与功能/grounding 字段均按组件保存，不能把多组件动作压成一个无法追责的总布尔值。

五层构念和主责固定如下，不得用最终质量一项替代：

| 层级 | 判定问题 | 预冻结指标 | 主责 |
|---|---|---|---|
| Retrieval | exact Rank-1 是否属于正确用户/时间/对象并服务当前目标 | `retrieval_top1_fit_rate`、owner/time正确率、Recall@k或better-rank rate、abstention | retriever |
| Eligibility | 候选是否真实、边界允许且提供具体增量 | owner/time、goal/function、boundary/burden、specific increment四门 | hard eligibility |
| PM Step1 | 候选可用时，本轮请求它是否有正的冻结执行器边际价值 | BA、recall、specificity、Brier、LOFO、ON/OFF比例、相对matched-random选择提升 | PM |
| Step2 | PM请求的 exact evidence 是否被忠实、自然且按动作合同使用 | requested/realized exact match、required-evidence use、functional contribution、grounding fidelity、atomic compliance、fallback | executor/generator |
| End-to-end | 完整策略是否达到质量—风险—成本平衡 | NetWin、material risk、critical events、token/cost | full system |

责任规则固定为：retrieval错而PM关为PM成功拦截；retrieval错且PM开时retriever为首责、PM资格门为次责；
retrieval正确且PM正确请求但证据未用、被扭曲或动作未落实时Step2负责；前三层均有效而ON仍不优于OFF时，
该状态的资源边际效应为nonpositive或存在结果噪声，不能自动归罪generator。generator自报trace仅为遥测，不能作为
functional contribution或grounding fidelity的gold；必须结合运行时绑定、结构检查和独立语义审核。

预冻结聚合必须同时报告：

- `retrieval_top1_fit_rate` 与候选缺失/abstention；
- `pm_hard_gate_violation_rate`、`pm_value_selection_balanced_accuracy`；
- `requested_realized_exact_match_rate`、`required_evidence_use_rate`；
- `functional_contribution_rate`、`grounding_fidelity_rate`、`scaffold_exposure_rate`、`fallback_rate`；
- quality/risk/cost 按 `retrieval_fit × pm_correct × execution_valid` 分层，禁止只报全局平均掩盖责任。

### 2.7 P1现有工程进展（2026-08-06，见账本第20节）

P1（实现 V5.3 executor，零新 outcome）已有以下真实、经代码验证的进展，供跨会话对齐当前状态：

**已完成**：
- 候选级 `contribution_slot` 原型代码（2.3节，commit `611e75e`），并完成 Rank-1/Top-k 字段拆分；
- MP/ME/MS 构造性规模压力测试已排除“目录规模必然导致检索崩溃”（不等于自然状态资格）；
- ME 编译器召回率修复（0.25%→1.89%，账本 V15-MEM-56）与检索排序断链修复
  （真实138-state面板0/138→108/138，账本 V15-ARCH-33）；
- `PMV2FeatureBuilder` 确认为已弃用的 pre-retrieval 表示，不作为 Step1 输入（账本 V15-ARCH-34）；
- MP/RS 从 `REQUIRED_EVIDENCE_NOT_USED` 及其变体豁免（账本 V15-PM-43）；
- judge 化名消解，修复 EvoEmo 第三方化名误判（账本 V15-GEN-36）。
- 当前 V3/V5.2 已消费构造的 ME exact Rank-1 支持复核：128/128 个 effect state 均通过
  compiler 且与构造目标精确绑定；这只证明当前构造可承载 ME，不是 V5.3 fresh 资格或效应证据。
- 合法 `M0+R0` 的空证据trace已改为运行时确定性规范化：模型若自报`user_message_1`等不存在
  ID，只作为原始telemetry保留，realized `used_evidence_ids`固定为空，不再丢弃正常无资源回复；
  有授权证据的动作继续严格检查ID（账本V15-ARCH-35）。
- `stagewise_accountability_ledger` 已实现为严格类型schema，唯一键为
  `state_id × policy_condition × seed_label`，并补齐原合同遗漏的`pm_correct`、
  `required_evidence_use`、`execution_valid`与`scaffold_exposure`等可计算字段；零API schema freeze
  已通过（40个合同字段0缺失，账本V15-MEAS-54）。正式P2/E4 runner仍须逐行写入并做完整性断言。

**已发现、尚未解决、P2冻结数据前必须处理**：
- ME 试点目前只验证了 eligibility/特征管线，不是本文2.3/4.3节要求的真实 Effect FIT 标签
  （账本 V15-EVAL-24）；MP/MS/RS 尚未各自跑过同等方法论；
- 旧 `explicit_advice_welcome` 在当前已消费 effect 构造上识别为0/128；V5.3三态
  `current_action_readiness` 原型在同一开发回放上识别128/128，但尚未经过内容独立资格，
  不能当成已泛化；
- MP/MS/RS 的候选级特征尚未各自做过训练域 vs EvoEmo域的正式对比（目前只对ME完成）。

特别说明：EvoEmo 已暴露，只能用于 transport diagnosis，不能把其 `78.3%` subtype prevalence
当作内部训练目标。训练域只需覆盖部署所需的特征范围和正/非正机制，不需要复制外部开启率。

**尚未开始**：4.2节正式规模FIT数据生成、P2冻结、P3训练、P4/P5确认与外部证据——本节之前的
全部阶段。

## 3. Generator 选择与冻结

V5.3 可以更换 generator，因为这是新方法版本；所有 baseline 必须使用同一个新 generator。不能默认
Llama-3.1-8B 一定能稳定完成证据整合，也不能在新 confirmation 上挑模型。

只允许一次 pre-outcome compatibility bakeoff：

1. 使用已经消费、不会再进入 FIT/confirmation 的历史 Step2 失败案例；
2. 候选最多两种：当前 Llama-3.1-8B 与一个更强、可负担的固定 instruction-following generator；
3. 只看 schema、evidence-ID、required contribution、禁止专名补全、原子动作和完成率，不看新质量结果；
4. 选择规则和费用上限在调用前冻结；
5. 两者均未通过则停止 V5.3 付费生成，不能继续轮换模型。

最低兼容门：结构完成率 100%，evidence-ID 绑定 100%，内部标签泄漏 0，未授权专名/数字 0，
RS 原子动作遵从≥95%，资源 required-contribution 自动可核验率≥90%。这些是工程兼容门，不是论文效果门。

## 4. 从外部考卷反推 V5.3 训练数据

### 4.1 训练 superdomain

新的内部训练状态必须覆盖而不复制外部文本：

- ESConv：结束/寒暄/倾听/复述/聚焦问题/明确建议邀请/上一轮已执行等 RS 格；
- EvoEmo：同用户大历史目录、具体旧目标、旧观察、未完成线程、过去动作—结果、同主题干扰项、
  current echo、wrong owner、age/冲突和大候选池；
- ES-MemEval：事实回忆、时间、冲突、拒答和 profile/session 交叉，但不把 QA 任务等同四资源分类；
- 内部专有：相反 MP preference、联合动作、显式边界和完整16动作。

每个 semantic family 内同时含 ON/OFF；topic、长度、前缀、候选数、资源 subtype 和标签解耦。
同一模板换话题不算独立样本。split 按 synthetic user、semantic family、counterfactual group 全部隔离。

> **P2前置合同（2026-08-06校正，见账本 V15-DATA-79）**：旧 468-card PMV2 backend 的
> ME actual Rank-1 `past_action_result=0%`，因此不得复用；但当前 V3/V5.2 已消费构造的128个
> effect state 已全部具有 compiler-valid 且精确绑定的 ME Rank-1。两件事不矛盾：前者淘汰旧
> backend，后者证明新构造方法可用，但二者都不是未见 V5.3 效应证据。P2 不回写或重标旧语料，
> 也不复制已消费行，而是新建内容独立的 V5.3 superdomain，并要求：
> （1）设计为正例的 ME 原始历史经同一 compiler 后确有 action+result；（2）每个 intended-positive
> state 的同 topic Rank-1 绑定通过；（3）负例覆盖 context-only、不可迁移、冗余、拒绝行动等机制；
> （4）topic relevance 仍优先，不能为了追平 EvoEmo 的78.3%比例而让 subtype 覆盖目标匹配。
> （5）行动准备度用三态记录，`UNKNOWN` 不作 OFF gold。正式数据必须在内容独立的用户/语义族/表述上
> 覆盖 invitation、decline、redundant、goal-mismatch 和 unavailable 等可观察机制，但不得在真实paired
> outcome产生前把构造条件叫作40个“正/非正gold”。

### 4.2 样本规模由真实独立组、可识别性和可获得精度决定

旧V5.2 outcome可用于解释历史CI为何很宽，但不能再投影出一个必须靠复制模板达到的V5.3最低N。正式生成前
先完成外部考卷反推的support matrix和候选蓝图，然后：

- 使用全部通过内容、身份、严格过去、shortcut和split审计的唯一user/family/counterfactual group；
- 特征矩阵须有变化且可识别，四个head的每个关键运行时slot都同时存在支持与反例；
- invitation/decline/redundant/goal-mismatch等是构造strata，不在outcome前假称正/非正label；
- FIT/confirmation/sealed按user和family完整隔离，同一模板换话题或同用户多个state不增加独立N；
- 在生成前按实际独立group、参数量、预期可报告CI宽度和预算冻结真实N；若不能支持窄非劣声明，就报告
  实际估计和宽CI或缩小主张，不能复制近重复样本补到128/64/40。

外部结构依据及各组件训练支持范围见
`docs/PM_V1_5_V5_3_EXTERNAL_EXAM_BACKWARD_TRAINING_SPEC_20260807_ZH.md`。

每个 state 使用同一 seed 的 ON/OFF matched pair；如要估计生成波动，只在预冻结25%子集加第二 seed，
不能把同一状态的多个 seed 当独立样本。

### 4.3 标签和训练目标

质量只比较同 state、同 candidate、同 seed、仅目标组件不同的 A/B。grounding risk 单独审核，cost
从真实 usage 读取。按 ITT 保留 generator nonuse、fallback、tie、lose 和 misuse。

`worth_opening=1`：ON 在状态聚合后实质优于 OFF、没有 critical grounding error，且执行账本有效。
其他有效结果为 nonpositive；只有 assignment、wrong resource binding、缺输出/usage、identity 漂移等机械
实验错误才作 invalid。功能是否真正做功单列为机制分析，不再作为删行许可证。

## 5. 一次性阶段与停止规则

### P0：冻结旧证据和暴露账本

- V5.2 结果只作旧架构诊断；
- 完成剩余独立 risk overlap，修正 risk codebook但不重评旧质量；
- 重新计算所有内部、ESConv、EvoEmo state/user/question 的时点化暴露；
- 输出 fresh-holdout availability。没有真正未见外部数据时，不许把旧分区改名为 lockbox。

### P1：实现 V5.3 executor（零新 outcome）

- 新模块、新 contract、新 output directory；不得原地修改 V5.2 实现或结果；
- 实现 Top-k discovery/Top-1 execution、typed response program、evidence-aware generator、guard 和 fallback；
- 用历史已消费案例完成单元测试和最多两模型 compatibility bakeoff；
- 冻结 generator、prompt、schema、execution plan、guard、cost accounting 和实现哈希。

### P2：冻结数据、baseline 和评测

- 冻结 FIT/confirmation/sealed 的 user/family/group split 和 power-derived N；
- 冻结所有 Step1 feature、model class、超参数网格、threshold selection 与 BGE challenger；
- 冻结主 baseline、随机种子、alias 去重、quality/risk/cost 指标、critical event 统计单位；
- 生成任何新回复前完成 leakage、shortcut、nuisance 和 candidate coverage 审计。

### P3：一次 FIT、一次训练

- 整批生成全部 matched pairs；
- 一次完整质量与risk结果面板，不再出10/16/32条修复包；
- 20%预冻结独立 overlap；分歧集中裁决一次；
- 一次训练四head和一次BGE challenger；FIT之后不改特征、generator、阈值或样本。

### P4：一次内部 confirmation 与 sealed

- confirmation 只消费一次；失败不建立第二份；
- confirmation 通过后才打开 sealed；
- 运行完整16动作、单bit消融、joint projection 和所有同栈 baseline；
- 得到内部QRC主结论。

### P5：外部证据

- ESConv：同栈RS corrected replication；
- EvoEmo：同栈纵向记忆 repaired replication，使用每用户私库和该用户自己的历史；
- ES-MemEval：继续作为QA/检索/回答诊断，不冒充response QRC；
- 如需“全新外部纵向确认”主张，必须另有未触碰用户/数据集。当前 EvoEmo p1–p18均有不同程度
  开发或结果暴露，不能通过重新切分恢复纯净性。

EvoEmo 用户数很少，user-cluster CI天然较宽。它适合证明机制运输、方向和失败模式；若论文硬要求
窄 `-0.05` 外部非劣界，需要新增规模更大的独立纵向支持数据或独立模拟器协议，不能靠重复18个用户
或把多个turn当独立用户获得虚假精度。

## 6. Baseline 与消融

所有主 baseline 共享完全相同的 retrieval、exact candidate、hard eligibility、V5.3 Step2、generator、
seed 和评分。不同的只能是资源开关策略。

### 6.1 主表

1. `always-off`：合法 `M0+R0`；
2. `fixed-high-eligible`：打开所有存在且通过硬资格的组件；
3. `transparent-rule`：预冻结人工规则；
4. `learned-PM-full`：四个 V5.3 value head 与16动作投影；
5. `cost-matched-fixed`：仅用 FIT/qualification 的 input-token 分布选择一次固定动作；
6. `cost/on-rate-matched-random`：按组件和成本分层、预冻结seed随机开关，用于检验 learned 的收益是否
   只是“少开”而不是“会选”。

### 6.2 次表和消融

- `learned-full minus MP/MS/ME/RS` 单bit消融；
- 单组件 fixed arms，用于解释各组件，不代替主系统比较；
- EvoEmo 的 Raw Session Top-4 与 All Raw 仅作 memory representation 次表；
- Legacy V1.0 只有能在当前栈精确 replay 才作同栈 baseline，否则只列历史结果；
- oracle candidate/effect 只作不可部署上界。

cost-matched 和 random-matched 必须在测试 outcome 前冻结；与其他臂完全alias时只计一次物理调用。

## 7. 评测设计修正

### 7.1 质量页

质量评审者看到：当前对话、同一用户的“已验证过去背景”中性面板、匿名A/B回复。两臂展示完全相同的
背景面板，不显示哪个资源被开启。这样既不会把真实过去记忆误当幻觉，也不会奖励“只要提到过去”。

### 7.2 Risk页

显示当前对话、该回复获准使用的证据与边界、候选回复。严格区分：

- 旧信息未在当前消息重述，不自动等于 stale；
- 真实但无关的过去信息主要是function/quality失败，只有产生具体伤害时才是material risk；
- 自然来源说明不自动等于内部标签泄漏；必须实际泄露内部ID/字段/系统规则或造成明显非人化破坏；
- fabricated recall、wrong owner、把过去升级为当前原因、明确边界违反仍是critical。

### 7.3 评审来源

必须保存 `annotator_type=human/llm` 与真实模型或人员编号。由ChatGPT/Claude完成的标注不得在论文写成
“human review”。实际人评可作primary，LLM judge只作敏感性；若没有实际人类，必须准确称为model-panel
evaluation并降低结论强度。

### 7.4 评审是一次性测量，不是调试回路

V5.3 只允许三次预冻结的正式语义评测波次：P3整批FIT、P4一次fresh confirmation、P5一次合并外部复制包。
每一波固定为一套primary、预冻结20% independent overlap和一次集中分歧裁决；不得再建立10/16/28/32条
“修复小包”。P1兼容性资格赛只使用已消费历史案例和结构/绑定/做功工程门，不读取新质量结果。

只有 assignment错位、wrong resource binding、owner/version串接、缺输出或usage、身份哈希漂移、盲评A/B串接
错误等预定义机械无效，才允许版本化修bug并重跑受影响单位。质量差、risk高、功能率低、PM未过门或输给baseline
均是科学结果，不得据此改prompt、阈值、样本、评审说明或再建confirmation。这样保留必要语义评测，同时阻断
“人评—修正—再人评”的开放循环。

## 8. 最终成功门

### 8.1 Step1机制门（内部fresh confirmation）

每个head：balanced accuracy≥`.65`、recall/specificity≥`.60`、Brier优于prevalence和transparent rule、
预测ON/OFF各≥15%、leave-family-out BA≥`.60`、五seed/重采样稳定性满足预冻结门。四head全过才称完整
four-component PM learned；否则按组件报告，但仍可测试系统级Pareto。

### 8.2 系统QRC门

主质量指标固定为 cluster-level NetWin=`P(win)-P(loss)`，非劣界`-.05`，95% cluster bootstrap CI。
adoption score仅作敏感性，不与NetWin混用。

learned-PM必须同时满足：

1. 对 always-off：质量非劣，预测ON strata的NetWin点估计>0；
2. 对 fixed-high：质量非劣，material-risk不恶化，generator input tokens至少下降10%；
3. 对 transparent-rule：质量与risk非劣，并在质量或cost至少一个维度严格改善；
4. 对 cost/on-rate-matched-random：质量点估计更好或risk更低，证明选择能力不只是稀疏化；
5. critical fabricated recall、wrong-owner personalization和明确边界违反按独立state/user计数，正式门为0；
6. 每个可测组件报告coverage、ON/OFF、requested/realized、functional-use和风险，不用总平均掩盖组件；
7. 任何material misuse不能被质量或成本抵消。

### 8.3 外部结论层级

- `A`：内部完整门通过，且至少一个真正未触碰外部纵向数据上的memory QRC通过；
- `B`：内部完整门通过，ESConv与EvoEmo在其可测子域方向一致，但EvoEmo因暴露/用户数只作复制；
- `C`：部分组件或learned不超过rule，但得到稳定的分层边界；
- `D`：新executor或learned policy仍未超过baselines。

## 9. 这次什么叫“成功”

工程成功可以在结果前保证：同栈、公平、资源真正进入generator、无机械append、绑定可追溯、无数据泄漏、
样本量由precision决定、结果一次性冻结。

科学成功不能预先保证。若在这些条件下仍未通过，结论将是“在当前资源、有限语义和固定generator下，
低容量PM未能稳定取得预注册QRC平衡”，而不是继续修改到通过。只有这种停止规则，最终的正结果才可信。
