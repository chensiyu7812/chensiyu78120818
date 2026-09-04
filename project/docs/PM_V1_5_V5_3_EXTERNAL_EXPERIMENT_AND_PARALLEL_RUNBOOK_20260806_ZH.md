# PM V1.5 V5.3 外部实验定义与双 Codex 并行运行手册

状态：`COORDINATION AUTHORITY / CAPABILITY-FIRST REVISION / PRE-FORMAL-FIT / 2026-08-06`

科学事实源仍为以下三份，本文件不建立第四套科学合同：

1. `docs/PM_V1_5_V5_3_INTEGRATED_EVIDENCE_EXECUTION_PLAN_20260805_ZH.md`
2. `data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json`
3. `docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`

本文件只承担两项职责：

- 把三个外部实验的可测能力、指标、判定口径和 baseline 集中展开；
- 规定两个 Codex 会话的任务边界、依赖顺序和唯一交接点，避免同时修改同一事实源或提前消费 outcome。

上述合同不是不可修订的教条。**在任何V5.3正式paired outcome生成前**，若真实数据、代码审计或链路目标
证明合同不合理，可以由leader记录理由并同步修改三份事实源；正式outcome生成后才禁止为追分修改同一版本。
如本文件与上述三份事实源冲突，先按研究需要判断哪一方应修正，再由leader做一次三源同步。正式环境固定为：

```text
/home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5
PYTHONNOUSERSITE=1
PYTHONPATH=src
```

任何付费 API 调用仍需独立的 stage、identity、精确费用上限和用户明确授权；本文不是付费授权。

---

## 1. 当前判断：已经确定了什么，尚未达到什么

### 1.1 已经由真实外部数据反推并冻结的方法边界

外部数据要求的不是一个“理解一切复杂心理语义”的大模型 PM，而是下列有限、可审计系统：

```text
dataset adapter / current visible state
  -> same-user strictly-past private resource store
  -> source-specific Top-k candidate discovery
  -> exact Rank-1 execution candidate
  -> machine-provable structural validity + narrow semantic opportunity features
  -> four source-specific low-capacity value heads (MP/MS/ME/RS)
  -> 16-action joint projection, including legal M0+R0
  -> typed response program
  -> one evidence-aware whole-response generation
  -> structural/binding guard and deterministic M0 fallback
  -> stagewise accountability ledger
  -> separate quality, risk, cost and mechanism evaluation
```

三个外部考卷反推出的最低能力分别是：

| 外部域 | 实际要求 PM/链路做到什么 | 不要求它证明什么 |
|---|---|---|
| ESConv | 在没有纵向私有记忆时自然 mask MP/MS/ME；RS 能识别当前支持动作、已执行动作、显式边界与重复；允许 M0+R0 | 不证明长期记忆或个性化 |
| EvoEmo response | 同用户、严格过去、大候选池；区分 MP profile/constraint、MS continuity/observation、ME action-result；拒绝 wrong-owner、current echo、冲突和只同主题的干扰项；Step2 真正吸收证据 | 不证明 MP preference、真实临床效果或所有16动作的自然覆盖 |
| ES-MemEval QA | 对历史进行检索、读取、时间/冲突/拒答推理并生成短答案；gold 对生成侧不可见 | 不等价于支持回复 QRC，也不能证明四个 response head 都学会 |

这部分方法规格已经足够明确，不应再因某个外部分数临时更改本体、组件定义或职责划分。

### 1.2 当前还不能写成“已经达到”的部分

截至本文件建立时，V5.3 达到的是“目标和解决路线已清楚、可以直接做出来”，但还不是“PM 已经学会并泛化”：

- 新 Step1 特征、typed Step2、M0 规范化和逐层账本已有实现或原型；
- V5.3 四个 value head 尚未获得正式 paired ON/OFF outcome 并训练；
- `stagewise_accountability` schema 已冻结，但正式 runner 仍未接线；
- ME 受控稀疏池的 Rank-1 绑定不能代表 EvoEmo 真实候选密度；
- 三态行动准备度在70条真实 EvoEmo 发言上零假阳性，但 INVITES_ACTION 只有2个正例且只命中1个，DECLINES_ACTION 没有自然正例，故只能称保守精度初步合格，不能称召回已完整资格；
- MP/MS 域对比正在由另一 Codex 实现，RS 域对比与全新 ME superdomain 尚待正式交付；
- V5.3 external plan、call plan 和 outcome 都尚未物化。

本文件建立期间，Worker随后提交的域审计又进一步确认：EvoEmo没有`MP_PREFERENCE`内容，故该子构念只能
内部验证；MS候选编译资格内外均有支持，但`continuity_request`槽位内外都未触发；RS旧训练构造使用直接向
generator下指令的语言，真实runtime在160条上0触发。这些不是“外测不公平”，而是P2必须重建训练
superdomain、并把外部不可测能力留在内部实验的直接证据。

因此当前准确状态是：

> 外测需要的 PM 和链路已经定义；主要 V5.2 架构错误已有对应修复；但“修改后的完整系统已达到外测要求”仍须经过 P2 冻结、P3 FIT、P4 fresh confirmation 后才能成立。

这里不存在理论上“做不出来”的阻碍。三态观察器的2个正例不是能力上限，而是随机自然抽样不适合测稀有类别
召回。正确工程解法是另建内容独立、措辞多样、类别平衡的资格集，必要时用窄语义分类器替代不断扩正则；
自然样本只估计真实频率和假阳性。MP/MS/ME/RS的缺口也都已经能落到数据构造、检索、Step1或Step2的
具体任务，不再是模糊地“继续做人评”。

---

## 2. 所有 response 实验共用的评测合同

### 2.1 公平性与同栈约束

所有 response baseline 必须共享：

- 相同 state、当前可见对话和同用户私有历史；
- 相同 memory compiler、Strategy Bank、query builder、候选池和 exact Rank-1；
- 相同的结构性无效检查，以及相同的语义机会特征计算；只有candidate absent、wrong owner、future/stale
  invalid、明确当前拒绝、明确重复/已执行等可机械证明的情况才能hard-off，稀有语义机会不得在Step1前被
  窄正则静默删除；
- 相同 typed Step2、generator、temperature、seed、token cap、guard 和 fallback；
- 相同评审页面、rubric、聚类单位和置信区间算法。

不同条件唯一允许变化的是资源开关策略。资源正文、检索排名、generator 或 evaluator 不得随 baseline 改变。

### 2.2 机制指标：先分责，再看最终回复

每个 domain、policy、component 都必须报告：

| 层 | 指标 |
|---|---|
| Retrieval | candidate-present、Top-1 fit、owner/time正确率、abstention、候选池规模、retrieval margin；可比时报告 Recall@k/nDCG@k |
| Eligibility | owner/time、goal/function、boundary/burden、specific increment 四门及 hard denial 率 |
| PM Step1 | ON/OFF率、概率与阈值、structural-invalid ON、BA/recall/specificity/Brier（只在有gold的内部域）、相对 matched-random 的选择增益 |
| Step2 | requested-realized exact match、generator received/used evidence、required-contribution、functional contribution、grounding fidelity、atomic-move compliance、scaffold exposure、fallback |
| End-to-end | quality NetWin、material-risk rate、critical events、prompt/total tokens、API cost；latency只作描述性诊断 |

最终 quality 差不能自动归罪 PM：必须按 `retrieval_fit × pm_correct × execution_valid` 分层。

### 2.3 Response quality

Primary 是同状态匿名 A/B 采用判断：

- 页面显示相同可见对话；若存在经过验证的过去背景，两臂显示同一份中性背景面板；
- 不显示 policy、component、action、cost 或资源开关；
- `A materially better / B materially better / tie / uncertain`；
- 只有足以改变实际采用决定的差异才选 A/B；轻微文风、温度或长度差异为 tie；
- decisive criteria 固定为：
  `grounded_context_fidelity`、`emotional_understanding`、
  `request_and_dialogue_fit`、`immediate_helpfulness`、
  `clarity_naturalness_not_overloaded`、`materially_equivalent`。

主统计：

```text
NetWin = P(learned better) - P(learned worse)
```

报告 win/loss/tie 原始数、cluster-level NetWin、95% cluster bootstrap CI。ESConv 按 dialogue/user 聚类，EvoEmo 按 user 聚类。同一状态不同 seed 不得当独立人。

实际人类标注才可称 human primary。ChatGPT、Claude、Gemini 或其他模型产生的标注必须记录为 `annotator_type=llm`，只能称 model panel / LLM sensitivity，不得改称人评。

### 2.4 Interaction-and-grounding material risk

Risk 独立于 quality。页面显示当前对话、该回复真实获准的证据/边界和单条回复，只判断严重到足以改变是否采用的风险，不评价帮助性或资源是否做功。

固定类别：

1. `stale_or_conflicting_use`
2. `unsupported_personal_claim`
3. `overgeneralized_pattern_or_cause`
4. `fabricated_recall`
5. `explicit_boundary_violation`
6. `excessive_directiveness`
7. `internal_resource_label_exposure`

另从证据 owner 绑定派生 `wrong_owner_personalization` critical flag；它不能被淹没在普通 unsupported claim 中。

主统计为每独立 cluster 的 `any_material_risk`、各类别事件率、learned-minus-comparator risk difference 和 95% cluster bootstrap CI。资源未做功可以是 `risk=no`；真实过去信息未在当前回合重述也不自动等于 stale。

### 2.5 Cost

Primary cost 是真实 generator input/prompt tokens。另报告 completion tokens、total tokens、API价格换算、retrieval/embedding本地计算开销和 observed latency。Latency 不是随机化性能结论。

主系统相对 `fixed-high-eligible` 必须至少降低10%平均 input tokens。cost-matched/random-matched 的动作和 seed 必须在任何测试 outcome 前冻结。

### 2.6 “能用、及格”的系统目标与外部证据层级

本研究要的不是所有指标满分，而是一个相对基本策略**能用、可比、确实学过选择**的PM。P4内部fresh
confirmation按下列目标判断；这些数值是透明的实质差异参考线，不是自然定律：

1. 对 always-off：质量非劣（NetWin 95% CI下界 `>= -0.05`），且预测 ON strata 的质量 NetWin 点估计 `> 0`；
2. 对 fixed-high：质量非劣；risk difference 的95% CI上界 `<= +0.05`；input tokens至少下降10%；
3. 对 transparent-rule：质量和risk非劣，并在quality或cost至少一个维度严格改善；
4. 对 cost/on-rate-matched-random：quality点估计更好或risk更低，证明不是单纯“少开”；
5. `fabricated_recall`、`wrong_owner_personalization`、`explicit_boundary_violation` 三类 critical event 必须逐条报告、
   不能被平均质量掩盖；研究原型出现事件不等于整项实验作废，但不能据此声称部署安全；
6. 任一 material misuse 不得被质量或成本抵消；
7. 每个可测组件必须单列 coverage、ON/OFF、functional use、risk 和 requested-realized，不能只报总平均。

如果learned PM相对fixed-high、transparent-rule和matched-random在质量上没有明显实质下降，同时risk或cost
至少一个稳定改善，就可以称“有限PM具有可用的Pareto选择能力”；若四head只有部分超过规则，则按组件报告，
不把整个系统宣布作废。

完整四资源主张的最低“及格”证据分两层，不能混为一个神秘总分：

1. **学会选择**：MP/MS/ME/RS每个head在内部user/family-held-out数据上都必须同时产生ON和OFF，balanced
   accuracy点估计高于0.5、优于恒开/恒关，并至少不劣于transparent-rule；同时在相同ON率或相同cost下优于
   matched-random。这里不要求0.8或0.9，也不因单个宽CI宣布算法不存在，但恒关不能算学会。
2. **选择有用**：联合16动作learned-PM相对fixed-high保持质量、降低risk或cost；相对transparent-rule和
   matched-random至少有一个不可由“单纯少开”解释的净改善。只有第一层而没有第二层，只能说分类器学到了
   标签；只有第二层而第一层失败，只能说一种保守启发式碰巧省资源。

外部域只测其真实覆盖的子集：ESConv测RS，EvoEmo测MP_PROFILE/MS/可获得的ME，ES-MemEval测历史检索与QA。
MP_PREFERENCE和外部缺失的拒绝建议能力由内部受控环境承担。外部不能测到某个子构念，不会抹掉内部证据，
但也不能伪称该子构念已完成外部验证。

P5的ESConv/EvoEmo使用同一estimand和参考线并报告完整CI，但不设置新的神秘“通过门”。ESConv正式test
有169个dialogue、EvoEmo有18个user cluster；把turn当独立人会虚增精度。P5直接回答方向、运输、机制、
失败边界，以及是否与“能用、及格”的系统目标一致。若将来要声称窄界限的external noninferiority，才需要
规模更大的独立外部数据。

### 2.7 评审波次

P5 外部 response 只进行一次合并语义评测波次：

- 一套 primary；
- 预冻结20%独立 overlap；
- 一次集中分歧裁决；
- 全量独立 LLM judge 只作敏感性；
- 禁止小包人评—修prompt—再人评。

---

## 3. 外部实验一：ESConv response replication

### 3.1 数据与单位

- 正式V5.3使用全部169个独立ESConv test dialogues，每段对话按固定协议选一个state；旧122只是排除47个
  格式兼容试点对话后的历史子集，不再把“曾用于修执行格式”当永久排除理由；
- 原始`ESConv.json`有1300段，但训练/开发部分可能参与Strategy Bank形成，不能把1300全部冒充test；
- 正式物化前重新验证hash、dialogue去重和无corpus-level`situation`泄漏；
- memory unavailable：MP/MS/ME 结构性 mask；
- 独立统计单位：dialogue/user，不是单条 supporter turn；
- 身份：corrected repaired replication，不用 ESConv outcome 训练或调 PM。

### 3.2 测什么

- RS exact Rank-1、动作前提、非冗余、上一轮已执行与显式边界；
- learned RS head 是否比固定全开/人工规则更会选择；
- M0+R0 是否在不需要策略时保留质量并节省成本；
- RS Step2 是否只执行一个原子动作、没有追加第二建议/问题。

不能由 ESConv 声称 MP/MS/ME 长期记忆泛化。

### 3.3 指标与标准

- 机制：RS candidate coverage、eligible rate、ON/OFF、Top-1 fit、requested-realized、atomic compliance、fallback；
- quality/risk/cost：第2节全部指标；同一QRC参考线用于解释，直接报告169个test dialogue上的效果和不确定性；
- RS 子域成功还要求：有实质 ON 和 OFF 覆盖，不能通过“恒关”得到 QRC；具体最低 ON/OFF 支持由 P2 power/sample freeze 在 outcome 前写入机器合同；
- 若 cost-matched-fixed 与 always-off 完全 alias，只生成一次物理回复并在逻辑表保留两个条件。

### 3.4 Baselines

主表：

1. `always_off`：M0+R0；
2. `fixed_high_eligible`：所有 eligible RS 开；
3. `transparent_rule`：冻结人工规则；
4. `learned_pm_full`：外部动作空间仍为16动作，但 MP/MS/ME被结构性mask，实际检验RS bit；
5. `cost_matched_fixed`：当前栈、outcome-blind冻结；
6. `cost_and_on_rate_matched_random`：按RS开启率/成本和预冻结seed随机。

次表：RS单bit oracle上界可作诊断；Legacy V1.0 只有在当前 V5.3 栈精确 replay 时才能进入次表，否则只列历史数字。

---

## 4. 外部实验二：EvoEmo longitudinal response replication

### 4.1 数据与单位

- 正式V5.3使用全部18个合成长历史用户、204个已固定response states（每个用户沿用既有固定track/turn）；
- 旧p7–p18的12用户/138状态只是qualification/lockbox治理切分，不再作为永久丢弃p1–p6的理由；
- 不按V5.2质量/risk重新挑state，直接使用完整预先存在的204-state panel；
- 每个 state 只能访问该 user 且严格早于 current session 的历史；未来会话、其他用户和全库混检为机械无效；
- 独立统计单位为user；204个state绝不能当204个独立用户；
- 所有 p1–p18 已有不同程度开发/结果暴露，不能称 pristine lockbox。

### 4.2 测什么

- MP_PROFILE/constraint 是否只在会改变建议范围、时机或负担时开启；
- MS 是否找回并使用具体旧目标、观察或未完成线程，而不是只找同主题摘要；
- ME 是否在当前允许行动时检索真实 action-result/mechanism，并作为可拒绝选项使用；
- 大候选池、同主题干扰、wrong owner、current echo、age/conflict 下的检索与路由；
- RS 在纵向回复中的联合动作表现；
- Step2 是否真正融合证据、保留过去/当前时态、归属和自然表达；
- quality—risk—cost 运输及各层失败责任。

不能由此证明 MP_PREFERENCE、完整16动作自然覆盖、真实临床效果或全新用户泛化。

### 4.3 指标与标准

- 第2节所有机制、quality、risk、cost指标；
- candidate池按组件报告中位数/分位数、Top-1 fit、wrong-owner/未来泄漏必须为0；
- user-cluster bootstrap为主；state-level结果只能作描述性；
- 第2.6节QRC参考线用于判断方向是否与内部一致；直接报告18用户上的估计和宽CI，不因样本较小停止实验；
- MP/MS/ME 每个组件均报告 available、eligible、ON、used、functional、risk 的完整漏斗；候选缺席或 UNKNOWN 不得从分母静默删除；
- Raw-session比较只回答“结构化表示与原始session上下文”的次问题，不替代policy主表。

### 4.4 Baselines

主表：

1. `always_off`；
2. `fixed_high_eligible`；
3. `transparent_rule`；
4. `learned_pm_full`；
5. `cost_matched_fixed`；
6. `cost_and_on_rate_matched_random`。

次表/消融：

- `learned_full_minus_MP/MS/ME/RS` 四个单bit消融；
- MP/MS/ME/RS单组件 fixed arms；
- `raw_session_top4_plus_same_strategy`；
- `all_raw_sessions_plus_same_strategy`；
- oracle candidate/effect只作不可部署上界；
- Legacy V1.0仅在当前栈精确replay时作次表，否则只列历史结果。

所有 raw-session 条件也必须同用户、严格过去，并使用同一个generator/seed/evaluator。

---

## 5. 外部实验三：ES-MemEval QA diagnostic

### 5.1 数据与单位

- 官方公开v1.0.0 artifact共1,427题，覆盖18个用户；V5.3主诊断使用全部1,427题，并明确这是公开
  benchmark evaluation，不是未见lockbox；
- 旧p13–p18的418题结果只作V5.2历史对照，不作为缩小V5.3主分母的理由；论文需说明公开artifact为
  1,427题，与论文正文报告的1,209题manifest不同；
- 五类：information extraction、temporal reasoning、conflict detection、user modeling、abstention；
- 生成只看到 question 与该condition允许的历史，不看到 answer、evidence、capability 或 group；
- 聚类单位为 user，另报告question-level bootstrap作敏感性。

### 5.2 测什么

- 记忆检索是否找对session；
- 历史事实、时间、冲突和拒答是否读对、答对；
- typed-memory与raw-session/full-history表示的准确率—成本关系；
- response-utility PM 在明确历史QA上的有限跨任务运输。

ES-MemEval不做情绪支持response quality，也不做第2.4节的 interaction-and-grounding risk 主张。QA中的幻答、错误归属和应拒不拒单列为 QA error/abstention diagnostic，不与response material-risk率合并。

### 5.3 指标

Objective primary：

1. 官方 normalization 后的 set-overlap Token F1；
2. `bert-score==0.3.13`、`bert-base-uncased`、不rescale的 BERTScore F1；
3. 按全部1,427题和五种capability分别macro average；
4. abstention accuracy、should-abstain false-answer rate；
5. conflict-detection单列；
6. 在全1,427题上重新物化official-evidence映射；只对可确定映射题报告session Recall@4/nDCG@4，
   无gold或标识不完整题单列，不伪造检索分；旧418题中的341题数字只作历史对照；
7. prompt/completion/total tokens、调用成本与latency诊断。

LLM-as-Judge 0–2只能作第三敏感性指标，必须独立冻结prompt/model并报告与objective指标的一致性；不得作为训练gold或替代Token F1/BERTScore。

### 5.4 判定标准

ES-MemEval是诊断，不承担V5.3 response QRC的二元通过门。必须报告每个condition的绝对分、与 `typed_memory_fixed_high` / `official_session_rag_top4` 的同题paired差、user-cluster 95% CI和token成本。

允许的结论只有：

- 检索/回答是否改善；
- learned PM是否表现出有限QA运输；
- 哪些capability或资源schema不在当前支持范围。

不得把“QA learned输给fixed-high”写成response PM整体失败，也不得把“QA分数更高”写成四组件QRC通过。
由于公开benchmark及部分旧结果已经可见，V5.3不再事后发明QA非劣阈值；本域直接用客观分数、paired差、
uncertainty和成本回答“有没有运输”，不使用新的二元pass/fail标签。

### 5.5 Baselines

1. `no_memory`；
2. `full_history`；
3. `official_session_rag_top4`；
4. `typed_memory_fixed_high`；
5. `typed_memory_learned_pm`。

`transparent_rule`可作预冻结次要诊断，但不进入官方五条件主表；cost-matched不适用于QA；RS对事实QA结构性N/A；Legacy V1.0不适用。

### 5.6 EvoEmo / ES-MemEval 同源防作弊合同

EvoEmo response 与 ES-MemEval QA 共享18个用户和底层历史。它们是**同一数据来源上的两种互补任务**，
不能写成两份统计独立的外部数据，也不能把两者结果简单合并扩大样本量。

这里“同源”本身不是作弊：RAG本来就应从该用户的合法历史中找答案或连续性证据。作弊发生在**可访问范围
越过任务边界**时。尤其`data/external/evo_emo.json`在同一个user object里同时保存`dialog_history`与
`questions`（后者含question、answer、evidence、capability等评测字段），因此不能把整个user object序列化给
retriever、PM或generator。

合法同源使用：

- 当前用户只检索自己的、相对当前状态严格过去的原始会话；
- QA query参与正常检索；response current turn参与正常检索；
- typed memory由统一compiler从允许的原始历史确定性生成；
- response memory的输入投影只允许预先存在的`basic_info`与严格过去的`dialog_history`；现有
  `build_evo_memory()`正是这个范围，但正式runner仍须做字段级断言；
- QA生成端只接收question文本与该condition允许的session history；answer/evidence等gold另存
  evaluator-only文件，以question_id在生成完成后关联；
- 所有baseline共享相同的可访问源历史、时间截断、query构造规则和gold不可见性；
- `learned-PM / fixed-high / transparent-rule / matched-random`等**路由策略比较**共享同一候选池、检索器和
  Step2；`official-session-RAG / full-history / typed-memory`等**表示方法比较**允许使用各自预先定义的表示与
  检索器，否则就失去baseline含义，但不得改变源历史、时间边界或借助gold。

禁止的同源泄漏：

- QA `answer`、`answers`、`evidence`、`capability`、`question_group`进入query builder、retriever、PM或generator；
- response路径读取或序列化`questions`；任一路径使用无时间索引的`event_experience`、`social_relationship`、
  顶层`summaries`或`subsequent_topics`作为当前时点的可见记忆；
- 使用未来session、当前答案所在supporter turn、其他用户历史或全库身份捷径；
- 从官方evidence ID直接指定Top-k，或用gold answer/evidence调BGE阈值、融合权重、PM阈值；
- 将ES-MemEval精确question/answer/evidence文本复制进P2内部superdomain；
- 把EvoEmo response outcome或ES-MemEval QA分数用于修改同一V5.3版本。

P2/P5前必须生成机器审计：字段访问白名单、每题/每state最大可见session index、candidate owner覆盖、未来/跨用户
违规数（必须为0）、内部superdomain与外部question/answer/session的exact与规范化n-gram重叠、生成messages中gold字段
命中数（必须为0）。另对最终serialized prompt做负向canary测试：把answer/evidence植入被禁止字段，若prompt或query
发生变化则审计失败。BGE等组件可在历史开发诊断上选择，但一旦进入V5.3正式全1,427题评测就冻结，不再按分数修改。

论文统一表述为：`one shared synthetic longitudinal source, evaluated through two task surfaces`；EvoEmo回答response
QRC，ES-MemEval回答retrieval/QA，不把二者称为独立外部复现。

---

## 6. 三个外部实验的 baseline 总矩阵

| Baseline/condition | ESConv response | EvoEmo response | ES-MemEval QA |
|---|---|---|---|
| always-off / no-memory | 主 | 主 | 主 |
| fixed-high-eligible / typed fixed-high | 主（RS） | 主 | 主 |
| transparent-rule | 主 | 主 | 可选次表 |
| learned-PM-full / typed learned | 主（RS子域） | 主 | 主压力测试 |
| cost-matched-fixed | 主，alias则去重 | 主 | 不适用 |
| cost/on-rate-matched-random | 主 | 主 | 不适用 |
| Raw Session Top-4 | 不适用 | 次表 | 主 |
| All Raw Sessions / full history | 不适用 | 次表 | 主 |
| learned-full-minus-one | 内部/可选RS消融 | 次表 | 不适用 |
| Legacy V1.0 current-stack replay | 条件性次表 | 条件性次表 | 不适用 |

---

## 7. 双 Codex 不冲突执行协议

### 7.1 单一领导与共享文件规则

- Leader Codex负责：本运行手册、三份权威事实源同步、P2-READY同版本确认、正式生成计划、外部指标/baseline和最终聚合。
- Worker Codex负责：预先分配的独立诊断/资格脚本、独立output目录和分项报告。
- Worker不得直接修改三份权威事实源；完成后以commit、报告路径和机器结果交给leader，由leader一次性同步。
- 两个会话都不得使用`git add .`、不得amend/rebase对方commit、不得覆盖对方output目录。
- 开始任务前先运行`git status --short`；发现对方未跟踪文件时不得移动、格式化或纳入自己的commit。
- 当前已知Worker所有权文件：
  `scripts/v1_5/64_mp_ms_contribution_slot_domain_comparison_v1_5.py`及其专属output；leader不得修改。

### 7.2 Worker工作流 W：P2前置数据与运输资格

可以并行、零API：

| ID | 任务 | 输出边界 | 完成定义 |
|---|---|---|---|
| W1 | 三态观察器内容独立资格补全 | 新脚本/新output/分项报告 | 不只随机自然频率；另有内容独立、措辞多样的INVITES/DECLINES/UNKNOWN平衡资格集；报告recall/precision/混淆，UNKNOWN不作OFF gold |
| W2 | MP/MS contribution-slot train-vs-external域审计 | Worker现有script 64及独立output | 同一候选级代码；Rank-1与Top-k不混写；按user/family报告，不把state当独立人 |
| W3 | RS域审计 | 新script/output/report | card precondition、nonredundancy、burden fit、already executed在训练域与ESConv/EvoEmo的支持范围可比 |
| W4 | 全新ME superdomain与真实密度审计 | 新数据构造脚本、manifest、零outcome报告 | 外部文本零复制；正/非正、至少8族；同用户多候选、同topic不同事件碰撞；intended-positive compiler-valid且exact Rank-1绑定 |
| W5 | shortcut/leakage/duplicate/同源审计 | 新report | topic/长度/前缀/候选数/subtype与标签解耦；user/family/group split零交叉；没有未来/他人历史；P2与EvoEmo/ES-MemEval外部文本零复制；QA gold字段对生成/检索零可见 |

W1–W5只产生观察/资格/数据构造证据，不生成paired response outcome，不训练head，不读quality/risk。

### 7.3 Leader工作流 L：执行器、账本与外部合同

可与 W1–W5 并行：

| ID | 任务 | 完成定义 |
|---|---|---|
| L1 | 正式runner接入`StagewiseAccountabilityRow` | 每个 expected `state×policy×seed`恰好一行；五层字段完整；缺行/重复行fail-closed |
| L2 | Step2全动作兼容门 | 历史已消费case；覆盖M0+R0、M0+RS、四单组件和联合动作；schema/evidence binding 100%，atomic compliance≥95%，required-contribution自动率≥90%，内部标签/未授权专名数字0 |
| L3 | V5.3 baseline materializer | 六主baseline同候选/执行器/generator/seed；alias物理去重；cost/random在outcome前冻结 |
| L4 | power与评测freeze | FIT/confirmation/sealed的N、split、quality/risk/cost、cluster bootstrap、20% overlap和裁决协议写入机器合同 |
| L5 | 外部plan scaffold | 只物化数据身份、state、candidate lineage和逻辑条件，不生成回复；ESConv/EvoEmo/QA hash与gold边界检查通过 |

L1–L5不读取新的质量/risk outcome。L2若需真实付费compatibility调用，必须另行产生预算identity并请求用户授权。

L2 当前有四项已核实阻塞，不能只跑现有单测后宣布通过：

1. 机器合同目前规定`second_free_llm_fallback_allowed=false`，但当前`call_with_guard_and_rewrite()`会进行一次
   受约束rewrite。这里不机械服从旧合同，而按“系统能用且公平”选择：在已消费开发case上比较
   `一次同generator、同证据、只纠正明确结构错误的rewrite`与`立即M0 fallback`。若rewrite显著降低fallback且
   不新增未授权事实，允许把它正式写入Step2；所有baseline同样使用，并把第二次调用的tokens/cost/latency全部
   计入。若不稳定则直接fallback。选择规则和最终合同必须在P3 outcome前冻结。
2. 当前机器guard尚未完整实现“未授权专名/数字”检查；必须基于运行时授权实体/数值集合，而不是自然语言黑名单。
3. `atomic_move_budget`目前主要是prompt约束；RS原子动作数、列表长度和一点式边界还缺可靠的结构化实现/校验。
   在无法机器确定的语义边界上不得假装硬判，必须在程序输出schema中把response acts结构化，再检查计数。
4. RS资产的同栈漂移已经由leader裁决：V5.3 primary固定使用6-card
   `data/strategy/strategy_cards_v1_5_minimal.jsonl`，SHA256=`04c3af44d54ae9875cd817364ff46e90aa954e0bee81b295af279b1b38964d24`。
   这份Bank低容量、topic-agnostic，且已有独立freeze；80-card V4 Bank自身manifest仍是
   `BANK_CONTENT_FROZEN_H2_RETRIEVAL_PENDING`且`formal_rs_runtime_enabled=false`，本轮不把它混入primary。
   它可以作为未来扩展或版本化次表，但不得与6-card训练/FIT/外测结果混称同栈。P2机器合同仍须写入
   6-card path、SHA和card count，FIT、ESConv、EvoEmo与六个response baseline共享这一份。

### 7.4 唯一系统汇合点 P2-READY

这不是为了增加“门”，而是防止两个会话在不同系统上分别生成数据。以下项目完成后，leader把两边工作合成
唯一可运行版本；个别观察器不完美可以用UNKNOWN/OOD保守处理，不要求所有子模块满分：

- W1–W5报告完成，已独立核对而非只接受结论；
- realistic-density ME superdomain已经建立；其候选可用性、Rank-1支持和UNKNOWN覆盖有真实记录，若覆盖有限则
  由Step1保守关闭并在报告中限定主张，不能拿一个不真实的稀疏候选池代替；
- MP/MS/RS域差异已量化，任何out-of-support轴有明确UNKNOWN/OOD处理；
- Strategy Bank path/SHA/card count唯一冻结，训练、FIT、confirmation、ESConv、EvoEmo和baseline无漂移；
- L1账本runner、L2执行器兼容门、L3 baseline、L4 power/metric、L5外部scaffold完成；
- 新superdomain的内容、用户、family、group split和hash冻结；
- 三份权威事实源由leader做一次同步commit；
- 工作树无双方遗留的冲突修改；
- 明确记录哪些外部数据已暴露，不能称lockbox。
- EvoEmo/ES-MemEval同源审计通过：只共享合法的同用户历史，不共享gold/evidence/outcome，且不作统计独立相加。

Worker round-close以后，W1-W4的当前证据解释如下：

- W1的180条平衡资格集证明三态词法观察器高精度但低召回（INVITES 50%、DECLINES 33%、UNKNOWN 100%）。
  因此它只能作为Step1特征，不能继续充当硬资格门；漏判保持UNKNOWN，不能自动变成OFF gold。
- W2确认MP_PREFERENCE在EvoEmo结构性缺席，属于外测覆盖边界；MS specific-observation在内外域均有支持。
  这不要求删除MP_PREFERENCE，而是内部单独测它、EvoEmo只报告MP_PROFILE。
- W3的最初“训练域RS机会0%”结论是把6-card runtime错误地套到V3 effect-study构造上，已由脚本66纠正。
  同一effect-study hard-off实现下训练100%可进入、ESConv 94.4%、EvoEmo 95.6%；旧结论不得再引用。
  6-card系统的自然语义触发覆盖仍较窄，但在新的候选层分责中只影响透明规则/特征，不再清空候选池。
- W4产生8族、32 state、每state 7候选的ME种子，其中14/32同时满足compiler-valid与intended exact Rank-1。
  43.75%不是“ME无候选”，而是同主题竞争下Rank-1排序不稳。W6随后按预先写定的三种方法完成唯一一次
  零API reranker资格比较：current production与纯BGE均为15/32 intended Rank-1，但纯BGE的
  compiler-valid仅9/32；compiler-filter+BGE仅10/32，且相对production为0胜5负27平。因此V5.3不采用
  ME-BGE，也不再搜索第四种排序器；冻结现有production lexical+typed-tier exact Rank-1，Rank-1编译失败
  则该state的ME不可执行，不偷偷提升Rank-2。正式P2保留这一覆盖边界，依靠paired outcome学习“可用时是否值得开”。
- W5由leader完成了字段投影、gold canary和同源文本重叠的主要部分：18用户、419个response因果投影及
  1,552个QA evaluator row均无未来/跨用户/gold可见性违规；当前14个ME种子的112个model-visible surface
  与3,505个外部question/answer/session surface为0 exact、0 normalized 8-gram overlap。正式完整superdomain
  物化后仍需重跑duplicate/shortcut/split/overlap审计，当前结果不是对尚不存在数据的预先PASS。

正式训练规模不再机械照抄一个“128”数字。Leader必须在读取任何paired outcome前，根据最终可用独立
user/family/group数、每个head的参数量、ON/OFF事件数及聚类精度模拟冻结N；构造数量不足时缩小主张或增加
新独立group，不能复制近重复模板冒充样本量。

### 7.5 P2之后严格串行

```text
P2-READY（两个会话确认使用同一系统版本）
  -> P3: 一次整批 paired ON/OFF FIT生成
  -> 一次primary + 20% overlap + 一次裁决
  -> 一次四head训练/阈值冻结
  -> P4: 一次fresh confirmation；失败不建第二份
  -> 通过后一次sealed internal
  -> P5: 同一release内运行ESConv + EvoEmo + ES-MemEval
  -> 自动机制/cost/QA指标
  -> 一次合并外部human/model panel
  -> 最终聚合与论文表
```

四组件 paired ON/OFF response generation 属于P3，不是P2前置任务。P2前可以构造state/candidate与调用计划，但不能生成或查看paired outcome。

### 7.6 P5内部可并行但必须同一release

P5冻结后：

- ES-MemEval QA生成可与response生成并行，因为不读取response outcome；
- ESConv与EvoEmo可并行生成，但必须共享同一executor/generator release hash；
- 自动cost/coverage/routing可边生成边写账本，但最终聚合须等expected keys完整；
- 人评页面可由冻结plan生成schema，不得在回复未完成时抽样；
- LLM judge和human review可在全部回复seal后并行，双方不得看到彼此结果；
- 最终裁决和聚合最后执行。

任何一个域发现机械错误，只允许版本化重跑受影响单元；质量差、risk高或输给baseline不是机械错误，不能回头修方法。

---

## 8. 给 Worker Codex 的启动指令

Worker开始新任务前必须先读本文件和三份权威事实源，然后回复以下六项，不满足不得开工：

1. 当前任务ID（W1–W5中的一个）；
2. 将创建/修改的精确文件路径；
3. 明确不会修改的共享文件；
4. 是否读取任何已有quality/risk/outcome；
5. 是否调用API及预算identity（默认必须为否）；
6. 完成后提供commit、命令、测试、output和仍未解决限制。

Worker建议读取命令：

```bash
cd /home/tokkio/snap/metacom_v33_pm_v1_5_repair/project
git status --short
sed -n '1,260p' docs/PM_V1_5_V5_3_EXTERNAL_EXPERIMENT_AND_PARALLEL_RUNBOOK_20260806_ZH.md
jq . data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json
```

禁止把“脚本能运行”“开发集回放通过”“候选存在”分别写成“PM学会”“fresh资格通过”“资源产生质量收益”。

---

## 9. 当前下一步（2026-08-06 Worker round-close后的顺序）

### 9.1 现在并行、零API

Worker只做候选/数据面，不改三份权威事实源：

1. **ME reranker资格比较（已完成，commit `6070157`）**：三法没有新方法胜出；leader冻结现有production
   lexical+typed-tier exact Rank-1，编译失败即ME不可执行，禁止Rank-2补位，W6到此关闭。
2. **W7 正式P2候选蓝图**：按上述ME决定物化MP/MS/ME/RS的state、用户、family、counterfactual group、
   exact Rank-1及候选目录；保留MP preference/profile两个内部子域，MS明确走BGE-M3，RS固定6-card；
   只构造候选与split，不生成ON/OFF回复、不训练head、不读取quality/risk。W7必须遵守以下同栈细节：
   - MP记录结构化`field_type/field_value/owner/subtype`，字段名前缀和通用填充词不得冒充相关性；
   - MS使用release绑定的BGE-M3 snapshot和真实全因果同用户候选池，保存真实semantic score；
   - ME使用production lexical+typed-tier exact Rank-1；Rank-1编译失败记`unavailable`，不得拿Rank-2补位；
   - RS先用`rs_mechanical_candidate_pool()`形成机械安全池，再用`rs_shared_candidate_top1()`得到所有policy
     共享的唯一候选：透明机会规则命中时使用其候选，否则使用确定性lexical fallback。透明规则只决定自己的
     ON/OFF，不能替自己换另一张卡；
   - 每个state保存可见对话、current goal、model-visible surface、完整Top-k ID、exact Rank-1 ID、owner/time/
     version、subtype、score/margin、候选数、增量token、语义特征、hard-off原因和外部来源零复制所需hash；
   - 构造单组件paired-effect主状态与多组件interaction状态，但不要求16动作等频；16动作结构兼容已由L2
     单独验证。split按user/counterfactual group/content surface不交叉，family-stratified confirmation另保留
     entire-family-held-out运输诊断。W7只报告真实唯一group数量，不自行复制模板或冻结最终N。
3. 后续脚本使用`70w_...`、`71w_...`前缀，避免再与leader编号碰撞。

Leader并行负责运行面：

1. 已将6-card Bank path/SHA/card count、BGE-M3 snapshot、ME production排序和共享RS Rank-1选择器绑定到
   V5.3静态release；W7产物完成后再生成包含其hash的最终release identity。
2. L2的16动作零API结构计划已完成；W7后从正式蓝图内容独立抽取代表状态，物化显式
   rewrite-vs-direct-fallback开发比较计划；所有baseline共享同一选择，第二次调用
   的token/cost/latency完整计账。未获用户预算授权前只做零API物化和dry-run。
3. L4的metric/cluster/review定义已冻结；W7后按蓝图实际唯一group冻结真实N。L5字段投影/canary已完成，
   正式蓝图交付后补做完整overlap、duplicate、shortcut和runner接线；不读取新的response outcome。
4. Worker提交P2候选蓝图后，leader独立运行shortcut、duplicate、user/family/group split、同源overlap和
   exact candidate-lineage审计。

### 9.2 唯一汇合点

上述两线完成后，leader一次性同步机器合同、计划与失败账本，产生唯一P2-READY release identity和付费预算。
在此之前不生成正式paired outcome、不训练正式heads、不运行外部response/QA。P2-READY以后严格按7.5节
串行，不再用零散十几条人评结果反复改训练定义。

### 9.3 W7交付后的Leader审计与W7R（2026-08-06）

Worker W7已由commit `78dbbf2`交付，静态release绑定真实一致，但Leader独立审计判定当前蓝图尚不能进入
P2-READY。机器报告为
`outputs/pm_v1_5_v5_3_p2_candidate_blueprint_audit_v1/report.json`，完整解释见
`docs/PM_V1_5_V5_3_P2_CANDIDATE_BLUEPRINT_LEADER_AUDIT_20260806_ZH.md`。

主要原因不是候选目录完全不可用，而是训练构念尚未可识别：MP正例把候选内容写进当前消息，MS只有单候选且
未保存真实BGE分数，ME把intended target gold放入Step1 features，82/82 `current_goal`仍是构造标签，且每head
只有7–8个独立counterfactual group。lineage与interaction执行surface也不完整。

因此下一且唯一Worker任务为W7R：只修候选蓝图与真实独立group，零API、零outcome、零训练；不得修改Bank、
MS/ME已冻结方法、Step2或三份权威事实源。W7R交付后Worker停止，Leader复审通过后才冻结split/N、刷新唯一
P2 release并实现formal runner。W7R前禁止paired generation；这项限制来自学习数据与运行schema依赖，不是
根据模型结果新增的性能阈值。

### 9.4 W7R复审与Leader接管P2R（2026-08-07）

Worker W7R由commit `b21edc4`交付。Leader确认其schema/lineage修复真实，但学习蓝图仍不合格：MP通过把
preference缩成两个词绕开三词冗余启发式，16/16实质增量仍已在当前消息中；MS continuity positive
16/16仍冗余且只有5/16 Rank-1为构造target，三候选池也未覆盖EvoEmo观察到的13–33规模；ME 40/40
合法Step1运行时特征被一并清空；state-level group ID放大独立单位；RS/interaction缺少完整学习slot与MP覆盖。

完整复审见
`docs/PM_V1_5_V5_3_P2_CANDIDATE_BLUEPRINT_V2_LEADER_AUDIT_20260807_ZH.md`。下一步不再由Worker迭代
措辞；Worker保持停止。Leader只做一次方法级P2R：结构化MP候选、隐藏答案的真实密度MS池、恢复ME/RS合法
运行时slot、重建group/split单位与真实多组件interaction。零API静态复审通过后立即冻结P2 release并进入
formal runner/P3，不再以审计计数为目标继续表面调词。

### 9.5 三项外部考卷反推训练规格与双线重启（2026-08-07）

Leader已直接读取ESConv、EvoEmo/ES-MemEval原始结构、冻结外部surface和W7R蓝图，形成
`docs/PM_V1_5_V5_3_EXTERNAL_EXAM_BACKWARD_TRAINING_SPEC_20260807_ZH.md`及零API机器报告
`outputs/pm_v1_5_v5_3_external_exam_backward_training_spec_v1/report.json`。

结论：旧训练数据缺少纵向深度的判断成立，但应精确为“13–33条同用户严格过去目录、同主题竞争项、跨会话
状态演化和自然互动轨迹不足”；它与MP题干复述、MS答案泄漏、ME合法特征清空、family-condition混杂和
state-level group膨胀并存。外部结构规定支持范围，不直接提供`worth_opening` gold；正式标签仍由冻结Step2
的paired ON/OFF quality-risk-cost effect产生。

9.4的“Worker保持停止”在本节后改为一个新且边界独立的W8任务：Worker只构造外部文本零复制的longitudinal
catalog asset，不构造current state、不计算Step1特征、不生成label、不修改三份权威事实源。Leader同时实现
P2R state/schema/features/group/interaction。汇合后Leader导入catalog，Worker只读复核，Leader只运行一次最终
静态审计；通过即冻结P2 release并进入formal runner/P3，不继续surface tuning。
