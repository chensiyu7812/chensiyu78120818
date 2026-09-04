# PM V1.5 最终执行方案 V3：有限语义、效用路由、同栈内外验证

状态：`ACTIVE / SINGLE SOURCE OF EXECUTION TRUTH / 2026-08-04`

机器合同：`data/pm_v1_5_contracts/final_execution_plan_v3.json`  
分层架构：`data/pm_v1_5_contracts/staged_policy_architecture_v2.json`

当前外部执行附录：`docs/PM_V1_5_EXTERNAL_VALIDATION_AND_ESMEMEVAL_PLAN_20260804_ZH.md`

本文替代旧 V2 执行方案。旧 H1/H1R、D2/D3、RS 与 Step 2 小包全部保留为开发证据，
但任何“最新十几条”都不能覆盖别的证据层，也不能直接决定论文主张。

## 0. 最终研究问题

> 在固定的四类资源、检索器、generator 和风险口径下，一个低容量、可审计、语义能力有限的
> Policy Manager，能否从当前对话和实际候选的透明描述中，学会何时开启 MP、MS、ME、RS；
> 相比 always-off、fixed-high 与透明人工规则，在不明显降低即时回复质量、不增加明确的
> interaction-and-grounding risk 的条件下，减少不必要的资源注入和 generator 成本？

主张没有变：研究仍然是 `quality + risk + cost` 约束下的资源策略管理，而不是用户需求诊断。
V1.5 不声称 PM 能解决复杂、隐含、临床级的用户语义理解。它只学习在明确、可见、可审计的
语义条件下做一个及格的资源开关；复杂语义模型是 challenger 和未来方向，不是当前前置条件。

保留 MP、MS、ME、RS 四个 bit 和全部 16 个合法动作。`M0+R0` 是正常且重要的正确动作。
不直接训练 16 分类器，而是训练四个 component-effect heads，再经过确定性联合可行性投影。

## 1. 有限 PM 到底能理解什么

### 1.1 V1.5 primary 可以读取的内容

PM 不直接读数据集身份、gold、生成回复或 judge。它只接收以下 outcome-blind 信息：

- 当前是否有明确请求：倾听、复述、一个问题、一个可选建议、事实回忆、行动规划；
- 明确边界：不要建议、不要重复、只要一点、结束、低负担；
- exact Rank-1 candidate 的类型、owner、时间、来源、token 成本；
- 候选是否与当前目标相符、是否已被当前对话完整说出、是否陈旧/冲突/错实体；
- memory 是否包含稳定偏好/档案、过去 session 区分、过去动作及结果/机制；
- RS 卡的 atomic move、负担和显式使用/禁用条件。

这些字段可以由 typed compiler、透明规则或经过独立资格化的小型 factor estimator 得到。

### 1.2 V1.5 primary 不负责的内容

- 隐含且高度模糊的“用户真正需要什么”；
- 临床诊断、风险预测、人格判断或长期疗效；
- 仅凭一次随机生成预言哪条回复一定更好；
- 对任意未来新资源内容零样本保证；
- 检索器从整个库中找出最正确 item；
- generator 如何自然地表达和组合证据。

遇到必须依赖这些能力、而可见因素不足的状态，正确行为是 `abstain/off`，并单独报告覆盖率；
不能把黑箱猜测包装成 V1.5 已学会。

### 1.3 BAAI 的位置

Primary 先用透明、低容量输入训练完整 PM。BGE-small/BAAI 不是前置条件，也不能修复错 candidate、
错 owner、陈旧冲突、标签循环或 generator 误用。Primary 冻结后，才允许一个预注册 challenger：

- 只为 `goal/function fit` 和 `semantic redundancy` 增加冻结 BGE 表征；
- 使用同一 FIT/CONFIRMATION，不读取 sealed 或外部结果择模；
- 若 challenger 在 confirmation 上改善 proper score，并在 sealed 重复，报告“语义表示可能是未来
  改进方向”；否则保留为负向消融；
- 不再在 V1.5 内轮换 Qwen/NLI/更大 embedding 追分。

## 2. 系统分层和责任划分

固定 trace：

`state → retrieval candidates → exact Rank-1 → semantic observation → eligibility mask →`
`Step1 four effect scores → requested action → feasible action → Step2 execution plan →`
`injected evidence → realized action → reply/fallback → quality/risk/cost`

| 层 | 它负责什么 | 失败归谁 | 主要证据 |
|---|---|---|---|
| Retrieval | 从该用户自己的库中找到 exact candidate | 候选错、Top-1 不合适归检索器 | Top-1 fit、Top-3 recall、hard exclusion |
| Observation | 把可见状态和候选变成有限透明语义因素 | owner/time/goal/boundary/redundancy 判断错归观察层 | factor BA、反事实方向、coverage |
| Eligibility | 排除无候选、错人、冲突、边界违反和定义上不可用项 | 资格规则错误归合同 | gate recall/specificity |
| PM Step 1 | 预测“合格候选在固定栈下是否值得开启” | 输入正确但净效用预测错归 PM | BA、Brier、相对 baseline 增益 |
| Joint action | 把四个 bit 编译为可同时执行的动作 | 资源合同互斥或预算投影错误归 joint selector | projection、16-action/Hamming |
| PM Step 2 | 绑定 exact evidence、规定怎样使用 | PM 开对但没用、错用、编记忆归 planner/generator | binding、functional use、misuse |
| End-to-end | 判断整套系统是否值得采用 | 前层都过但回复无收益，归资源容量/生成上限 | blind quality、risk、cost |

任何报告必须写清 `版本 × split/domain × 层`。禁止再只写“MP 失败”或“ME 通过”。

## 3. Step 1 与 Step 2 的最终定义

### 3.1 Step 1 学什么

Step 1 不是“候选看起来相关就开”，也不是“预测某次随机 A/B 谁赢”。它学习：

> 在 candidate 已存在、eligibility 已通过、Step 2 已被证明能执行的前提下，加入这个组件相对
> context-only，是否在重复的同状态单组件实验中表现出正的预期净价值。

四个 heads 分别比较：

- MP：`M0+R0` vs `MP+R0`；
- MS：`M0+R0` vs `MS+R0`；
- ME：`M0+R0` vs `ME+R0`；
- RS：`M0+R0` vs `M0+RS`。

一次只改变一个组件。其他组件、prompt、generator、解码、候选与当前状态均固定。

训练标签来自两个层次，禁止混写：

1. `eligibility`：候选是否真实、相关、安全、非冗余并有明确用途；它是 mask/输入，不是最终 PM gold；
2. `worth_opening`：同状态 clean-pair 的实际效应标签，才是 Step 1 gold。

每个状态使用两个预冻结生成 seed。每对质量标为 `ON-win / tie / OFF-win`，风险另审：

- 两次中 `ON-win > OFF-win`、无 material misuse、执行有效且成本在预算内：`worth_opening=1`；
- `ON-win <= OFF-win` 且执行有效：`worth_opening=0`；tie 因资源有成本而归 OFF；
- treatment 没真正执行、证据错绑、不可裁决：`unknown`，不作为负例训练；
- fabricated recall、明确边界违反或 material misuse：该次记负效应并触发风险审计，不能被质量抵消。

这不是声称单次因果真值，而是用重复、固定栈、可见结果构造一个足够稳定的低成本监督目标。

### 3.2 Step 2 做什么

Step 2 不再学习第二次开关，也不得擅自 ignore/reopen。它把 feasible action 和 exact evidence 编译成：

- candidate identity 与允许功能；
- required contribution；
- attribution/epistemic mode；
- burden cap 与 forbidden inference；
- 失败时的 context-only fallback。

具体合同：MP preference 必须改变表达形式；MP profile 必须提供当前未出现的相关背景；MS 必须
以待核实方式承接 prior-session 主线/区分；ME 必须呈现过去动作—结果/机制和有限类比；RS 必须
落实 atomic move。固定短语“上次……”不算做功。

## 4. 为什么先资格化 Step 2，再训练 Step 1

旧路线曾用 generator 没有真正执行的 treatment 产生“资源无效”标签。这会让 PM 学到的不是
“何时资源有用”，而是“generator 何时没听话”。V3 顺序固定为：

1. 冻结 ontology、candidate 和有限 Observation；
2. 人工审核 eligibility/codebook；
3. 用开发态 H2 先证明 Step 2 能正确绑定和使用资源；
4. 只有合格 treatment 才生成 clean component-effect 数据；
5. 训练 Step 1；
6. 冻结后做 internal sealed、ESConv、EvoEmo 与最终 QRC。

H2 通过标准：exact binding=100%；单组件 functional-use≥.80；多组件 all-functional≥.70；
material misuse≤5%；fabricated recall=0；明确边界违反=0；fallback 可采用率=100%。

## 5. 从外部考卷反推平时学习

### 5.1 ESConv 真正能测什么

ESConv 是跨 dialogue 的情绪支持语境，但没有同一用户的可靠长期记忆。因此它主要测：

- RS 的候选发现、开/关和 atomic move 执行；
- 纯寒暄、结束、listen-only、明确建议邀请等 RS-off/on；
- context-only 与策略资源的即时 quality/risk/cost；
- memory bits 在无可用 memory 时是否正确关闭。

它不能证明 MP/MS/ME 的长期个性化能力。Strategy Bank 只由 ESConv train 建立，validation/test
不进入 Bank、训练、阈值或 prompt。

### 5.2 EvoEmo 真正能测什么

EvoEmo 有同一用户的纵向历史，主要测：

- MP_PROFILE、MS_SESSION、ME_REUSABLE_OUTCOME 的真实候选、检索、开关和 grounding；
- 小到大历史目录下的 transport、abstention、成本与误用；
- RS 在同一固定 Bank 下的辅助表现。

它不能充分测 MP_PREFERENCE、所有显式边界、完整 16 动作或每种 hard negative。p1–p6 只可用来
定义 schema/规模/形状；p7–p12 是 untouched qualification；p13–p18 是 lockbox。

### 5.3 内部受控环境必须补什么

内部数据负责外部考卷测不到但论文定义包含的能力：

- MP_PREFERENCE 的相反偏好与可证伪配对；
- wrong-owner、stale/conflict、current-redundancy、wrong-goal 最小反事实；
- MS 与 ME 的 subtype×current-goal 四格；
- RS 的同 family ON/OFF、结束/寒暄/低负担负例；
- 单组件和多组件、联合冲突、全部 16 动作；
- small/medium/EvoEmo-like large 私有目录与 OOD abstain。

内部不是伪造“外部结果”，而是透明、受控的能力测试台。论文必须把它称为 synthetic controlled
evaluation，并说明没有经过大规模自然分布校准。

### 5.4 学习与考卷同构的硬约束

内部、ESConv、EvoEmo 必须共享 ontology、memory compiler、per-user isolation、Strategy Bank、
query builder、retriever、candidate schema、Observation、PM、Step 2、generator 和 rubric。
只允许每个用户的私有内容不同。

内部训练分布必须是外部可测构念的 superdomain，而不是复制外部文本：

- 覆盖 EvoEmo p1–p6 所揭示的 history size、age、subtype 和 token 形状；
- 不复制 p7–p18 的用户、文本、答案或 outcome；
- 训练/确认/sealed 按 user、dialogue、semantic family、counterfactual group 隔离；
- 同一语义模板换 topic 不算新独立样本；
- 外部出现训练未覆盖的 subtype/规模时只报告 OOD/abstain，不把它说成 PM 失败或成功。

## 6. 一次性正式数据设计

过去 H1/H1R 暴露了回声偏好、ME 目标错配、恒关捷径和跨 split 重复，因此全部只作 development。
V3 不再修补旧 96 条，而建立一个版本化的、内容全新的 effect dataset。

### 6.1 单位和规模

每个 target state 来自独立 synthetic user，并指定一个 target component。其他资源保持关闭，避免
16 动作混杂。资格反事实与效应数据分成两个集合：资格集包含可用/不可用候选，但不生成回复；
效应集中的候选必须先通过 eligibility，负标签只能来自“资源被正确执行但没有净收益”，不能用
错人、陈旧、重复等显然不可用项给 Step 1 凑 OFF。每个 head 的最低规模：

| split | 每 head 独立 states | 人工标签最低形状 | 用途 |
|---|---:|---:|---|
| ELIGIBILITY_AUDIT | 32 | 16 eligible、16 ineligible | Observation/资格门，不生成回复 |
| EFFECT_FIT | 64 | 至少16 ON、16 OFF、其余可为 unknown | 训练与 grouped OOF |
| FRESH_CONFIRMATION | 32 | 至少8 ON、8 OFF | 一次资格确认 |
| SEALED_INTERNAL_TEST | 32 | 至少8 ON、8 OFF | 最终内部复现 |

总计每 head 32 个资格反事实和128个效应状态；四 head 共128个资格审核状态与512个 component
contrasts。每个 effect contrast 使用两个固定生成 seed。
同一 current state 可在资源完全独立且 trace 可分时共享 context-only 回复，但独立统计单位仍是 user。

数值是“及格学习”的工程下限，不支持生产级 80–90% 准确率。若人工真值达不到最低正负数，
说明当前资源/状态没有足够可学习效应，不能靠换 seed、补同模板或强制平衡伪造。

### 6.2 生成顺序

1. 先冻结 capability matrix、semantic families、反事实 group 与 split；
2. deterministic compiler 构造时间、owner、subtype 和目录，LLM 只做非关键表面改写；
3. 正式 retriever 重新发现 exact Rank-1，出题 intent 不进入模型或 gold；
4. 人评 eligibility 和 evidence codes；
5. 只对 eligible state 用已过 H2 的 Step 2 生成两次 clean pairs；
6. 盲评 quality，独立审核 interaction-and-grounding risk；
7. 按第3.1节机械派生 `worth_opening/unknown`；
8. 只在 EFFECT_FIT 训练；confirmation/sealed 不回流。

### 6.3 数据进入训练前的硬门

- exact candidate binding=100%，wrong-user retrieval=0；
- 跨 split exact/near/template/counterfactual 泄漏=0；
- 每 head 的 actual gold 满足最低正负数，恒定多数 baseline accuracy<.70；
- state-only、candidate-family-only、topic/length/count-only probe BA<.65；
- eligibility 双评 raw agreement≥.80、κ≥.60、每组件 raw agreement≥.70；
- Observation 主因子 BA≥.70，recall/specificity≥.60，最小反事实方向≥.75；
- Step 2 已过第4节门；未执行 treatment 不得进入 worth-opening 负例；
- p7–p18 外部内容、generator reply、judge、自报 reason、condition 名不得进入特征。

## 7. 训练方法

Primary 是四个标准化 L2 logistic heads：

\[
L_c=-\sum_i w_i[y_i\log p_i+(1-y_i)\log(1-p_i)]+\lambda\lVert\beta_c\rVert_2^2
\]

- 每 head 5–7 个预注册 factor；`C=0.3`；阈值`.5`；
- FIT 按 user/semantic-family grouped 5-fold OOF，固定5个 fold seeds；
- 输入是 Observation、eligibility 后的候选描述和预计 token cost；
- 不读 raw ID、dataset、gold condition、生成回复或外部 outcome；
- 不使用直接16分类、PPO/RL、端到端 generator 微调；
- HGB 只作旧方法对照；透明 rule 是必须击败的主 baseline；
- BGE challenger 按第1.3节执行。

## 8. “及格且学会”的冻结标准

### 8.1 单 head 学习门

FRESH_CONFIRMATION 上每个 head 必须同时满足：

- balanced accuracy≥.65，recall≥.60，specificity≥.60；
- Brier 同时优于 prevalence-only 和预冻结 transparent-rule；
- 预测 ON/OFF 各至少15%；
- 最小反事实方向正确率≥.70；
- 五 seed BA 标准差≤.05；
- leave-semantic-family-out FIT BA≥.60；
- 不允许单一模板、cue、family 或目录规模恢复标签。

这些是“基本学会”的门，不要求80%–90%。若只与规则持平，系统可以是可用规则系统，但不能
声称 learned PM 超过基本方法。四个 heads 全过才称 `full four-component PM learned`；不足四个
必须报告 partial，不能悄悄关闭失败组件后仍写完整主张。

### 8.2 联合16动作

四 score 先过 hard risk/boundary，再按预算和合同冲突做 deterministic projection；无可行 bit 返回
`M0+R0`。报告 per-head BA、Hamming loss、exact-action accuracy、requested→feasible→realized、
projection rate。exact-action 不是唯一主指标，因为四个及格 head 的联合全对率天然更低。

### 8.3 Step 2 门

沿用第4节门。若 Step 1 正确但 Step 2 未过，结论是 execution unsupported，不是 PM 不会开关。

### 8.4 最终 QRC 门

不用模糊加权总分，采用 risk-first 的约束式判断：

1. 对 always-off：整体盲评 quality 非劣，预测 ON strata 的 win-loss 点估计>0；
2. 对 fixed-high：quality 非劣、material risk 不恶化、generator input tokens 至少下降10%；
3. 对 transparent-rule：risk 不恶化、quality 非劣，并在 quality 或 cost 至少一个维度严格更好；
4. material misuse 不能被质量或成本抵消；
5. adoption-score 非劣 margin=`-.10`，risk-rate 非劣 margin=`+.05`，报告 user-cluster bootstrap CI；
6. sealed/internal 与外部结果均不回流改模型、阈值、prompt 或 Bank。

达到以上才可写“及格的 learned PM 在测得范围内找到 QRC 平衡”。若只超过 always-off、未超过
transparent-rule，只能写资源路由有用但学习算法没有证明额外价值。

## 9. Baselines 与必要消融

同一候选、检索、Step 2、generator 和评测栈下比较：

1. `always-off / M0+R0`；
2. `fixed-high eligible resources`；
3. `transparent-rule`；
4. `learned-PM-full`；
5. `cost-matched-fixed`；
6. `Legacy V1.0`：仅在旧动作可精确映射并能用当前栈 replay 时进主/次表，否则只作历史结果；
7. `Raw Session Top-4 + Strategy`、`All Raw Sessions + Strategy` 作 memory 次表 baseline。

内部 sealed 做 `full-minus-MP/MS/ME/RS` 单 bit 消融；BGE 是表示 challenger；Top-3 仅作 retrieval
oracle，不全部注入 generator。`learned-qualified` 可作保守敏感性分析，但不能替代 full 主结果。

## 10. 人评为什么存在，以及到哪里结束

人评不是每次十几条“决定最新版本”，而是只在四个固定位置做不同工作：

| 包 | 回答的问题 | 是否训练标签 | 不能做什么 |
|---|---|---|---|
| H-Eligibility | exact candidate 是否可用、边界/owner/goal 是否正确 | 只给 Observation/eligibility | 不能证明回复变好 |
| H-Step2 | 请求的资源是否绑定、做功、误用 | 否，是 treatment 资格门 | 不能评价 PM 开关 |
| H-Effect | 同状态 component-on/off 哪个有实质优势及风险 | 是，派生 worth_opening | 不能混入其他组件 |
| H-Final | 冻结系统相对 baseline 的最终 QRC | 否，只作论文结果 | 不能回流训练 |

Primary reviewer 完成 H-Effect；预冻结20%由第二评审复核，所有 uncertain/risk flag 复核。只有
构念级分歧需要裁决，不再连续发10/16/32条探索包。自动 judge 只作描述性敏感性分析。

## 11. 外部与内部最终考试

顺序固定：

1. FIT/OOF；
2. FRESH_CONFIRMATION；
3. 冻结 PM、threshold、retrieval、Step 2、generator；
4. SEALED_INTERNAL_TEST：完整16动作、MP preference、hard negative、QRC和消融；
5. ESConv：RS与memory-off；
6. EvoEmo p7–p12：memory transport qualification；合格后才开p13–p18 lockbox；
7. H-Final：internal/ESConv/EvoEmo 分域盲评和风险审核；
8. 只报告，不修改。

外部每组件至少20个独立、candidate-present、in-support state 才报告组件级泛化；否则明确写
dataset coverage limitation。内部成功不能冒充外部成功，外部不覆盖也不能抹掉内部受控能力。

## 12. 当前证据怎样处理

可直接复用：80-card V4 Bank、来源审核、RS direct-effect、D2/D3 rubric、EvoEmo 分区、memory
compiler、16动作 compiler、source-matched Step 2、risk taxonomy、V1.0 baselines。

只作 development：旧 H1/H1 v2/H1R 标签、旧 fit/fresh/sealed、所有10/16/32条 Step2 小包、旧
HGB/BAAI结果、自动 judge、自报 reason。它们用于定义新数据和失败账本，不进入正式 fit/confirm。

H1R 的正式结论保持：当时没有训练 PM，也没有运行 Step2；失败在 candidate-state/goal 配对、
MP preference 回声、ME goal-fit codebook 和真实标签形状。不能把它叫作 PM 或 generator 失败。

## 13. 唯一后续阶段

| Phase | 工作 | 结束条件 |
|---|---|---|
| V3-P0 | 冻结本文、机器合同、失败账本 | COMPLETE |
| V3-P1 | 建 capability matrix 与全新 effect blueprint；静态审计 | COMPLETE：640 rows，零泄漏/捷径失败 |
| V3-P2 | 资格化 Retrieval/Observation/Eligibility 与 Step 2 | IN PROGRESS：V2 Retrieval/exact Rank-1 已通过；H-Eligibility 双评可靠性通过，但24个修复决策面待双审；Observation/Step 2 待资格化 |
| V3-P3 | 生成并标注 EFFECT_FIT/CONFIRM/SEALED clean pairs | 标签形状合格，未执行项为unknown |
| V3-P4 | 训练四 heads，做一次 confirmation | 第8.1节四 head 全过或明确 partial |
| V3-P5 | 冻结后 sealed internal、消融与 baselines | 无修改地完成 |
| V3-P6 | ESConv、EvoEmo qualification/lockbox | 分域、in-support/OOD 完整报告 |
| V3-P7 | 最终 H-Final 与论文 | 停止实验 |

V3-P1 已完成：冻结128个 eligibility audit states 和512个 effect contrasts，共640个独立 synthetic
users、320组split内反事实；主题、四档历史规模和高/低收益富集均反平衡，construction intent不作
gold或模型输入，API/human/external lockbox读取均为0。静态报告见
`outputs/pm_v1_5_v3_p1_effect_blueprint_static_audit_v1/audit_report.json`。

V3-P2 的前半已完成：640 个状态全部由同一 bounded memory compiler、source-specific query、typed
memory Rank-1 和冻结80-card RS pre-PM ranker重放。四组件各160个目标候选均存在；subtype、目标
construction item、source catalog size 与 RS family 均为 `160/160`，完整 `visible state + target
candidate` 决策面重复为0。期间修正了两类仅属构造/审计的 bug：ME context/unresolved 不能谎报为
reusable outcome；MP target identity不能依赖JSON字典顺序。两项均在任何人评、回复生成和外部读取前
修复。初版报告保留在 `outputs/pm_v1_5_v3_p2_exact_rank1_v1/materialization_report.json`；人工后
构造修复另存为V2，不覆盖V1。

128项主审与32项独立重叠的最终eligibility为31/32一致，raw agreement `.96875`、Cohen κ
`.9375`，四组件raw agreement分别为MP `1.0`、MS `.875`、ME `1.0`、RS `1.0`，因此四门
codebook的评审可靠性通过。但把裁决事后仅用于出题QA时发现14个construction-intent mismatch：
MS distinction正例被实现成泛泛目标、MS wrong-owner负例把当前问题也改成friend、RS suggestion有
两条实际Top-1功能错配、RS current-request负例仍提供额外功能。MS恰好一组正例与一组负例互换，
故总量仍显示16/16平衡；aggregate balance不能替代semantic-family审计。V1标签不得冻结为gold。

修复仅改outcome-blind realization，使用正式Python 3.13.2 `.venv-pm-v1-5`重建为V2：640/640
candidate present、subtype/constructed-target/catalog均匹配，RS family 160/160，完整Step1决策面重复0，
API/回复/外部lockbox读取均为0。Eligibility的128项中只有24项完整decision surface变化；其余104项
只有在exact surface hash不变时才能继承既有标签。当前唯一下一步仍属于`V3-P2`：对这24项做两份
独立复核，合并104项carry-forward后冻结Eligibility；随后资格化Observation与全新H-Step2。修复页面在
`outputs/pm_v1_5_v3_h_eligibility_repair_delta_v2_candidate/`。此阶段继续禁止生成正式effect replies、
标旧H1R、训练旧opportunity heads、扩Bank、打开EvoEmo p7–p18或下载新embedding。

## 14. 失败、bug 和停止规则

- 代码未执行冻结合同、错 ID、split 泄漏、current 进入 memory 槽属于 bug，可修后新版本重跑；
- gold 可靠但 model confirmation 不达标属于科学失败，不能在同一 confirmation 上调到通过；
- eligibility 通过但 effect 正例不足，说明资源/固定 generator 在该范围内缺少可学习增益；
- transparent-rule 与 learned 持平，说明学习方法未证明额外价值；
- internal 通过、external in-support 低，说明学习与考卷 transport 受限；
- routing 过而 Step 2 失败，归 execution；两层都过而 quality 无收益，归资源/generator ceiling；
- 完整四 head 是原始主张。任何 partial/qualified arm 都必须明确降级，不能叫完整16动作已学会。

## 15. 论文结论分级

- `Level A`：四 heads、Step 2、internal 与外部 QRC 均过；完整主张成立；
- `Level B`：四 heads 在内部受控环境过，外部只支持其可测子域；主张按数据覆盖限定；
- `Level C`：部分 head 学会或 learned 不超过 rule；报告部分支持，不冒充完整 PM；
- `Level D`：可靠数据上未学会；报告有限 PM 的失败边界和可复现根因。

无论得到哪一级，研究目的仍是质量、interaction-and-grounding risk、成本的平衡；区别只在证据
实际支持多大的范围，而不是在结果不好时偷偷改变主张。

## 附录：先行研究只按失败层接入

- LongMemEval：仅在 memory indexing/retrieval/reading 失败时用于 V2；
- ESCA：仅在 Step 2 strategy planning/expression 失败时用于新 generator 版本；
- CASE：仅在 routing/grounding 正确但回复认知—情感表达不足时作 generator 消融；
- ESC-Eval：单轮结果成立后扩展多轮评价；其自动 ranker 必须本地资格化；
- BGE/NLI/Qwen：只有 gold、candidate 和透明输入可靠而表示不足时，才作为预注册 challenger。

这些方法都不能修复标签定义、candidate 错绑、数据泄漏或外部污染，也不能回填 V1.5 lockbox。

## 16. 2026-08-03 Observation 最终修订（覆盖第13节中已过期的 V3-P2 下一步）

16项语义修复复审已经完成并可用：5条过期/被替代的MP偏好全部判为不合格；RS的11条中，5条
合格、5条因本轮明确拒绝该动作而不合格、1条因当前已明说同一动作而无增量。它与80条decision
surface未变化的旧裁决精确合并为96条V2人工参考，不覆盖旧证据、不重复计数：FACTOR_FIT中每个
适用的组件×门均为8 yes/8 no，ELIGIBILITY_CONFIRMATION中每组件均为4 eligible/4 ineligible。
这96条只监督候选资格四门，不是`worth_opening`，也不评价回复质量。

冻结候选比较得到三个结论：lexical logistic只通过owner与goal；BGE-small四门均未通过；固定
DeBERTa NLI四门也未通过。因此BAAI/NLI不是当前关系判断的自动解法，V1.5不再继续换Qwen、改
hypothesis或搜索阈值。随后代码审计确认透明Observation V1错误地让goal与increment依赖其他门，
首次修复还把`leaves room`误识别为`leave`拒绝。两项实现bug均已修复，四门在64条FIT上全部过线。

但一次性32条未见语义表面确认仍失败：综合balanced accuracy `.71875`、recall `.9375`、
specificity `.50`，未过冻结的specificity `.60`门。goal/function单门确认BA为`.96`、specificity为
`1.0`；owner/time、boundary、increment则明显默认放行。确认集已经消费，不能继续根据其错误扩写
regex、换模型或调阈值。当前没有训练Step1、没有授权H-Step2、没有评价generator，也没有读取外部
lockbox；因此失败责任只属于pre-Step1 Observation表示，不能归给PM、检索器、Step2或generator。

方法边界据此作如下修订，完整四组件与16动作不变：

1. 后端已知的`user_id/owner_id`、候选时间、撤回/替代/version状态和candidate presence改为确定性
   hard gates，不再要求一个小文本模型从自由文本猜数据库事实；
2. goal/function保留为V1.5有限语义任务，并使用实际candidate subtype及RS卡片级前提；
3. boundary与increment采用`allow/deny/unknown`或`increment/redundant/unknown`三值判断；只有明确
   支持才放行，unknown在运行时OFF，但不能伪装成监督学习的negative；
4. Eligibility仍只决定“是否允许进入Step1”。Step1仍学习通过资格的组件是否有预期边际净收益，
   Step2仍负责绑定和真正使用被选资源，generator仍负责写回复；四层的错误分别报告；
5. 分组件、分数据域报告eligibility coverage与unknown/off比例。ESConv不能证明纵向记忆能力，
   EvoEmo只验证其覆盖的MP/MS/ME子域，内部受控实验补测外部数据不覆盖的偏好、边界与16动作；
6. 不再制作零散Observation人评包，不在已消费确认集上调模型。下一证据必须是版本化的结构化资格
   栈系统级资格赛；其通过后才进入既定H-Step2与effect clean-pair生成。

机器可执行边界冻结在
`data/pm_v1_5_contracts/v3_observation_structured_hard_gate_amendment_v1.json`。本修订不改变论文主问题：
仍检验一个有限、可审计的四组件PM能否相对always-off、fixed-high和transparent-rule，在固定资源与
生成栈下实现回复质量、interaction-and-grounding risk与cost的平衡；它只撤回“PM靠自由文本自行
理解全部身份、时效、边界和增量语义”的过度主张。

## 17. 2026-08-03 结构化边界 V2 与唯一 H-Step2 资格赛

第16节提出的“所有语义unknown在Step1前OFF”经真实640项候选覆盖审计后被否决：它会使四组件
各`0/160`通过，系统在学习前就退化为always-off。正式V2只把后端能够客观证明的candidate absent、
owner/version/active错误和本轮明确边界冲突作为hard denial；goal/function和specific increment等
有限语义信号可以是unknown，但只能进入Step1作为输入，既不是ON，也不是OFF gold。512条
COMPONENT_EFFECT状态因此全部进入后续effect学习入口，每组件128条；Eligibility仍不替代
`worth_opening`。

随后只构造一批版本化H-Step2：20个单组件（MP/MS/ME/RS各5）和8个兼容多组件动作。候选选择不读
人评、回复输赢、risk judge或外部lockbox。首次运行发现两个工程bug并整批作废保留：动作名曾把
`MPE/MSE`错误拼成`MPME/MSME`；通用结构化schema还允许generator声明未请求组件，而旧validator
静默忽略。正式V2改用canonical 16-action编码和每次调用动态exact-action schema，额外、缺失或重复
组件一律校验失败并走M0+R0回退。新身份运行结果为28/28 exact component trace、28/28
requested=realized、0 fallback。

> **历史状态（已由第18节覆盖，不再执行）：**

机器结果只证明绑定与结构合同运行，不能证明资源真正做功或没有语义误用。当时唯一剩余P2证据是
同一28项H-Step2功能审核：单组件每类至少4/5真正做功，多组件至少6/8全部做功，material misuse
最多1/28，fabricated recall=0，明确边界违反=0。通过后才进入H-Effect clean pairs和四head训练；
失败则只修Step2/generator，不把责任转嫁给Step1、检索器或人评，也不读取外部测试调到通过。

本节的机器合同为
`data/pm_v1_5_contracts/v3_observation_structured_hard_gate_amendment_v2.json`；H-Step2审核页为
`outputs/pm_v1_5_v3_h_step2_human_review_v2_candidate/human_review.html`。

## 18. 2026-08-03 记忆模块根因审计（覆盖第17节“失败只修generator”的过期表述）

28条H-Step2人工结果已完成，但联结exact candidate、当前消息、资源和回复后，确认该包同时混入
candidate eligibility、数据冗余、Step2执行和validator责任，不能作为纯Step2 promotion gate：

- 单组件做功为MP `3/5`、MS `3/5`、ME `4/5`、RS `4/5`；material misuse为`6/28`；
- 8条multi中5条MP候选的private-window事实已被current message逐字重复，五题在生成前无增量；
- 两个MS失败候选只说“有一个exact wording question”，没有保存问题正文；
- 单组件从structural admissible pool抽取而非冻结eligible reference，导致不合格RS卡被强制执行；
- bundle机器门用资源union审核每个组件，允许ME outcome错误地给MS claim背书；
- atomic RS与通用prompt允许的动作数不一致。

因此旧28条完整保存为development diagnosis，但撤销`FAIL_REPAIR_STEP2_BEFORE_EFFECT_GENERATION`
作为纯generator结论，也不得删除坏题后事后重算通过。当前V3仍未训练Step1；不能再说“MP/MS/ME
反复没学会”。

唯一下一步改为一次版本化replacement H-Step2：24个单项全部从冻结H-Eligibility reference中
选择eligible exact candidates（每组件6），另造8个由合格单项组成、逐资源非冗余且joint-feasible
的multi。Step2新增component-specific response/resource evidence binding，禁止union为单组件claim
授权；MS必须携带具体旧内容，ME必须是action+outcome/mechanism，MP preference只有当前move触发时
才进入，RS显式携带atomic move上限。门槛为每组件单项≥`5/6`、multi≥`6/8`、material misuse≤
`1/32`、fabricated recall=0、明确boundary violation=0。

该门通过后才执行既有512个effect contrasts和H-Effect；无效Step2 treatment记`unknown`而不是
Step1 OFF gold。Primary仍为每head 5–7个source-specific透明factor的L2 logistic；FIT/confirmation/
sealed仍为每head`64/32/32`，并要求至少`16/16、8/8、8/8`的ON/OFF、grouped OOF BA≥`.65`、
recall/specificity≥`.60`、Brier胜prevalence与transparent rule。BAAI只作冻结challenger，不能替代
candidate、label或Step2修复。

详细根因和执行合同分别见`docs/PM_V1_5_MEMORY_MODULE_ROOT_CAUSE_AUDIT_20260803_ZH.md`与
`data/pm_v1_5_contracts/memory_module_root_cause_repair_v1.json`。研究问题仍是完整四bit/16动作下的
quality-risk-cost平衡；本修订只确保训练数据可辨识、责任可归因，不预先承诺真实边际效应必然强。

## 19. 有限终局路线：禁止“十几条—改prompt—再十几条”

后续严格执行`docs/PM_V1_5_TERMINAL_EXECUTION_ROUTE_20260803_ZH.md`。流水线为：T0零API静态修复；
T1唯一32条replacement H-Step2；T2仅FIT的256个effect states；T3一次训练；T4一次128-state
confirmation；T5冻结后的128-state sealed、ESConv、EvoEmo和最终baselines。T1以后，任何人工、
confirmation、sealed或外部结果均不得回流修改prompt、特征、阈值或样本选择。

T1人工结果若未通过，停止Step2调试并按组件报告；FIT数据形状或OOF未通过，停止扩量；confirmation
未通过，不再建立第二个confirmation。终局只允许`FULL_V1_5_PASS`、`BOUNDED_COMPONENT_PASS`、
`NOT_LEARNED_ON_FROZEN_PROTOCOL`三种。旧Bank、Eligibility reference、640状态蓝图、compiler、
retriever和外部分区全部复用；只重生成受新Step2合同影响的回复。

## 20. 2026-08-03 方法级重置：Step1 路由与 Step2 执行正式分责

第19节的replacement H-Step2已经消费。机器合同虽然没有结构化输出错误，但实际发生20/32次
M0+R0回退；人工审核确认单组件MP/MS/ME/RS做功分别为`5/6、3/6、5/6、6/6`，8条多组件中仅
`4/8`全部做功，material misuse为`10/32`。该结果证明旧Step2把过多职责交给同一个8B generator：
它既解释资源、决定如何使用、复制审计span、撰写回复又自报理由；其失败不能转写成Step1 OFF标签。

因此，自本节起，第19节以前的设计和结果全部冻结为历史证据，不再作为未来执行入口。唯一正式入口
改为`data/pm_v1_5_contracts/final_execution_plan_v4.json`：

1. Step0只处理后端可证明的owner/version/active/显式边界硬错误；未知语义不是OFF gold；
2. Step1保留四个独立低容量head，只预测固定有效Step2下的期望净组件价值，再投影到16个合法动作；
3. Step2改为确定性typed resource adapter。owner、时间、来源、允许和禁止断言由运行时绑定；
4. generator只接收已编译约束/证据并输出自然语言，不再输出use/ignore、资源ID、span或自由解释；
5. 后置guard失败只进入一个确定性的context-only安全回复，不再第二次调用自由LLM回退；
6. V4只允许一份新的32项Step2 confirmation；其通过后一次性生成完整FIT并训练四个head。

Step1当前结论是“尚未在有效V4 treatment上训练，因而既不能宣称失败，也不能宣称已达到能力上限”。
它的V1.5能力边界限于显式/粗粒度goal、边界、owner/time、冗余、检索强度和cost；不主张诊断潜在
心理需求、理解讽刺或完成复杂多实体时序推理。透明模型为primary，冻结BAAI/BGE仅作challenger。

版本关系、失败方案与禁止复用规则见
`data/pm_v1_5_contracts/method_version_registry_v1.json`；完整方法说明见
`docs/PM_V1_5_STEP1_STEP2_METHOD_RESET_20260803_ZH.md`。本节不改变论文主张：仍检验有限、可审计的
四资源PM能否相对always-off、fixed-high、transparent-rule和V1.0 historical baseline，在同一冻结
资源/检索/生成栈下取得即时回复质量、interaction-and-grounding risk与cost的更好平衡。

## 21. 2026-08-03 ITT 修订：Step1 与 Step2 分开，取消语义人评训练许可证

第20节仍把一份新的32项Step2人工语义门设为正式Step1训练前提。这会重建已经持续三天的开放循环：
Step2未达主观阈值→修改prompt/adapter→再出小包→再次等待人评，而正式PM始终不能训练。该耦合现被
撤销。Step1与Step2不共享参数、梯度或训练标签；Step2在V1.5中不是一个待训练模型，而是被冻结的
typed adapter与response-only generator。

正式Step1保留两层：已有outcome-blind opportunity/eligibility层可独立训练或复用，只作mask/输入；
新增ITT value层学习“在当前冻结真实执行器下，请求某组件”的期望净价值。ITT treatment是
`requested_component_bit`，不是`realized_component_bit`。generator忽略资源、语义未做功、触发合法
context-only fallback、质量tie/lose或出现material misuse，均是请求动作的真实结果，不得排除或记
unknown。只有assignment错、wrong owner/resource绑定、请求资源未进入执行器、最终回复/usage缺失或
冻结身份漂移等机械实验失效才作废行。

因此下一步不再制作新的Step2小包。先用已经消费的失败证据一次性实现typed deterministic adapter、
确定性无历史fallback和机器不变量测试；随后在任何新outcome前冻结完整FIT计划、executor/generator
身份、分组、seed、特征与评测口径，整批生成一次。相同panel同时提供ITT quality/risk/cost训练结果
以及Step2 functional-use/misuse机制字段；后者解释结果但不再决定能否训练。确认集只消费一次，失败
不修、不建第二份确认。

该修订不降低主张，反而把estimand对准实际部署系统：PM学习的不是“理想generator下资源的内在疗效”，
而是“当前固定资源执行器下请求资源是否值得”。若未来更换executor或generator，环境已经改变，必须
建立新PM版本而非重解释旧标签。机器合同为
`data/pm_v1_5_contracts/staged_policy_architecture_v4.json`与
`data/pm_v1_5_contracts/final_execution_plan_v5.json`；二者自本节起是唯一未来执行入口。

## 22. Guard 与完整人评的最终边界

`me→ME` 与 `before/earlier→未授权记忆` 暴露的是既有失败类型复发，而不是需要继续补词表。自然语言
关键词不再允许决定机械实验有效性。assignment、candidate/owner/version、资源进入执行器、回复与
usage、冻结身份属于机器事实；陈旧/冲突使用、无依据个人断言、因果泛化和虚构记忆属于人工
grounding-risk outcome。运行时fallback即使误触发也保留为有效ITT结果，不得删除或据此新建修复包。

V2 后只允许一个完整256-state FIT人评panel，覆盖两个seed的ON/OFF质量与ON-arm grounding risk；
20% overlap预先hash固定，只做一次分歧归并。时间词、历史提及、通用措辞或风格偏好本身都不足以判
material risk。完整定义见
`data/pm_v1_5_contracts/v5_guard_and_human_outcome_boundary_v1.json`。人评结果只决定标签、机制分析和
论文结论，禁止再修改guard、prompt、样本、seed、feature或threshold。

## 23. 2026-08-04 V2 FIT 生成完成：语义词面误杀关闭，真实执行误差保留

V2 已按冻结计划完成 `1024/1024` 次调用：256 个独立状态、每组件64个状态、每状态ON/OFF各两个
同seed回复。1024个call ID与计划逐项一致，256个状态均为4行，ON/OFF seed完全配对；回复、usage、
candidate绑定和executor/generator身份均完整，机械无效行为`0`。因此该批可以进入唯一完整FIT
outcome review，不能再以新小包替代。

运行时共有60次确定性fallback，全部是`MISSING_PAST_ATTRIBUTION`：MS为`7/128`个ON调用，ME为
`53/128`个ON调用；MP与RS均为`0/128`。V1中的`me→ME`与`before/earlier→未授权历史`没有在V2
复发。这只证明机械guard修复成功，不证明语义grounding、功能做功或四个Step1 value head已经通过。
ME的高fallback是冻结generator/adapter在过去事件迁移上的真实执行结果，必须进入ITT质量、风险和
成本，而不是触发prompt修复或从训练数据删除。

Observation责任边界同时维持第17与21节的最终修订：owner ID、candidate version/time/status、presence
和显式机器可读边界属于后端事实；goal/function、候选相对当前目标的增量/冗余与无显式边界时的
不确定性属于有限Step1输入，不由regex硬判，也不把unknown伪装成OFF gold。早期“所有unknown先关”
曾导致四组件`0/160`覆盖，已经被正式否决，不能因担心复杂语义而恢复always-off。

## 24. 2026-08-04 唯一完整 FIT panel 与训练面最终冻结

V2的256条记录严格说是256个`state × exact candidate`实验单位，而不是256个彼此无关的自然语义组。
冻结蓝图从一开始就把它们组织成128个两成员最小反事实组：每组件64个unit、32个group。两成员用于
检验候选属性变化，不是重复垃圾；但若按state ID拆fold，就会形成伪独立。最终统计规则据此明确为：

- 两个generation seed先在各state内按预注册规则聚合，不把相同seed输出当新state；
- 同一counterfactual group的两条unit永远进入同一个CV fold，每条基础权重`.5`；
- 20%独立overlap每组件从13个不同group各抽一条，共52 state、104个seed-level质量对与104条风险项；
- 完整panel仍保留512个质量对和512条ON-arm grounding-risk/functional-use审核，不删除fallback、nonuse、
  tie、loss或misuse；
- 质量页不显示component/resource/ON-OFF；风险页只显示生成时授权的证据与边界，不显示组件名或arm；
- overlap有真实分歧时只允许一次集中仲裁，不再生成任何十几条修复小包。

在任何FIT人评标签产生前，既有`FINAL_FEATURE_NAMES`的四个5–7维透明feature schema已逐行物化；训练器
冻结为`StandardScaler + L2 logistic`，`C=.3`、`class_weight=balanced`、threshold`.5`，五fold seed为
`17/29/43/71/113`。这次物化发生在response生成完成后、人工outcome标签产生前；feature builder没有读取
回复、fallback、guard、functional-use、quality或risk。该时间偏差必须在方法附录披露，但不能作为
人评后再改feature的许可证。

人工完成后聚合器只执行机械步骤：解盲每seed的ON/OFF赢家；至少一个ON实质胜且没有OFF胜、没有
uncertain，并且ON回复无material grounding risk，才记hard positive；tie因非零上下文cost记
nonpositive；functional-use仅作机制解释。随后一次性训练四head并按冻结BA、recall、specificity、
Brier、预测开关比例、独立正负group、五seed稳定性与leave-family-out门判断是否进入唯一fresh
confirmation。FIT不过不补样本、不换BAAI、不调阈值；只报告`NOT_LEARNED_ON_FROZEN_FIT`。BAAI/
BGE-small仍只允许作为同一数据上的一次冻结表示challenger，不能产生标签或覆盖transparent primary。

执行资产：

- 完整panel：`outputs/pm_v1_5_v5_single_full_fit_outcome_review_v1/manifest.json`；
- 训练面：`outputs/pm_v1_5_v5_fit_training_surface_v1/freeze_report.json`；
- 训练合同：`data/pm_v1_5_contracts/v5_fit_training_surface_freeze_v1.json`；
- 唯一聚合/训练入口：`scripts/v1_5/26a_aggregate_and_train_v5_fit_v1_5.py`。

## 25. 2026-08-04 V5.2 内容独立系统确认冻结

V5.2 FIT 的系统级 grouped-OOF 回放出现了值得确认、但仍属开发证据的QRC Pareto形状：相对
component-fixed-high，质量为`139胜/112负/261平或同臂`，material risk为`47/512`对`97/512`，
prompt token减少`7.1%`；相对always-off，质量为`172胜/55负`，同时risk由`37/512`升至`47/512`。
这正是论文要检验的质量—风险—成本权衡，但不能用FIT自身宣布成功。

旧V3内部身份不能再充当V5.2确认：实际outcome账本显示EFFECT_FIT `256/256`、FRESH_CONFIRMATION
`128/128`、SEALED_INTERNAL_TEST `128/128`均已在此前执行中暴露。为避免测试回流，现已一次性冻结新的
V5.2确认身份：128个`state × exact Rank-1 candidate`单位、64个二元反事实组、四组件各32个单位、
8个未在旧内部split使用的普通非临床主题各16个、四档历史规模各32个。新集与所有旧内部split在
state ID、counterfactual group ID、规范化可见表面以及`current state + candidate`完整决策面上均为零重叠。

在任何生成结果前已冻结always-off、component-fixed-high、transparent-rule、learned-PM四种策略；
learned-PM沿用V5.2最终四head、`.5`阈值和完整16动作组合，请求ON为`65/128`，不是always-off退化。
每个状态使用两个协议哈希seed，ON/OFF同state同seed配对，共512次调用。backend-locked executor、
调用计划与实现哈希均已写入execution seal；确认后不允许修改模型、特征、阈值、资源、executor、
prompt或seed。当前API调用为0，预算代理上界约`$0.1220`。

本次确认只决定冻结V5.2系统能否把FIT内Pareto形状带到内容独立内部环境。它不重新裁定四个单head
是否通过旧分类门，也不把复杂语义理解列为V1.5主张。ESConv和EvoEmo已被历史V5.1执行暴露，后续
只能称为同一冻结系统下的corrected/post-hoc repaired replication，不能替代本次内容独立确认，
也不能用其结果回调系统。

冻结资产：

- 数据质量报告：`outputs/pm_v1_5_v5_2_confirmation_freeze_v1/freeze_and_data_quality_report.json`；
- 确认合同：`data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json`；
- 调用计划：`outputs/pm_v1_5_v5_2_confirmation_plan_v1/freeze_manifest.json`；
- 执行封印：`outputs/pm_v1_5_v5_2_confirmation_plan_v1/execution_seal.json`；
- 唯一执行入口：`scripts/v1_5/27j_run_v5_2_confirmation_generation_v1_5.py`。

### 25.1 确认生成与唯一结果面板状态

确认生成已完成`512/512`调用，ON/OFF各256；fallback与invalid ITT均为0，requested action和realized
action完全一致，因此执行层通过，不能因状态名包含`AWAITING...PANEL`而重跑。实际报告token为prompt
`113,180`、completion `16,189`。

结果面板保持三个构念边界：质量页只显示匿名A/B；grounding-risk页显示当前对话、该臂获准使用的证据
及回复；cost直接读取API usage。Step2 functional-use不在confirmation重复人评，因为它只作机制解释，
不属于冻结系统门。协议哈希选13/64个反事实组作为独立overlap，并在一次分歧裁决后停止。

为减少无信息劳动，公开surface逐字相同的行只评一次，再用私钥传播回全部底层行。最终人工工作量为主评
quality `255`、risk `423`；独立overlap quality `52`、risk `86`。聚合时仍恢复全部256个quality pair、
512个risk arm和64个counterfactual group，不改变样本量、策略选臂或bootstrap分母。

### 25.2 一次性确认最终结论与停止规则

唯一V5.2内容独立确认已完成最终标签归并和冻结策略回放，正式状态为
`FAIL_SINGLE_USE_V5_2_SYSTEM_GATE`。13项预冻结门中11项通过、2项失败：

- learned PM相对component-fixed-high的质量点差为`+.03125`，但64组cluster bootstrap 95% CI为
  `[-.109375, .1796875]`，下界低于预设非劣界`-.05`；
- learned PM选择的回复含4次critical fabricated recall，高于预设上限2。

因此V1.5不能写成“完整系统已通过确认”。但它也不是退化或未学习：learned PM在256个seed-level
决策中开启130次（50.8%）；相对always-off质量`+.2734375`，95% CI
`[.16015625, .38671875]`；相对transparent-rule质量`+.171875`、risk`-.04296875`、平均prompt
`-12.484375`，三项点估计同时改善；相对fixed-high risk降低`.0625`，其95% CI
`[-.1171875, -.015625]`，prompt降低`7.1783%`。最终可支持的表述是：**有限、可审计的PM学到了有意义的
保守资源取舍，并优于透明规则；相对全开呈现风险与成本优势，但质量非劣证据和critical grounding门
不足，完整主张未确认。**

责任诊断保持Step1/Step2分离：MS ON回复常有质量收益，learned PM漏开是fixed-high质量非劣失败的
主要来源，属于Step1泛化问题；4次critical事件来自两个RS状态、同一“可信联系人”卡和两个seed，
冻结generator四次虚构用户以前提过Sarah，属于candidate/card与Step2执行交互失败，同时Step1没有过滤
这一组合。ME是当前确认中最稳定的低风险正收益记忆组件；MP质量增量仍弱。

第三裁决中3条quality将盲页未展示的授权历史误认成虚构，2条risk将自然用户可见来源说明误标为内部
标签泄漏。最终只作枚举式构念修正，保存完整lineage；用未作这5处修正的primary-only标签重放，仍因
相同两项门失败，故正式结论稳健。

从本节起执行预冻结FAIL stopping rule：不建立第二份V5.2确认集，不根据本结果修改head、阈值、资源、
executor、prompt或seed。ESConv/EvoEmo只能作为同一冻结系统的post-hoc repaired replication，不能救活
内部确认。若未来修复MS漏开或RS虚构联系人，必须升为新方法版本并使用全新考卷；V5.2保留为当前论文的
bounded/negative confirmation证据。

最终资产：

- 最终标签与裁决lineage：`outputs/pm_v1_5_v5_2_confirmation_final_labels_v1/`；
- 冻结聚合：`outputs/pm_v1_5_v5_2_confirmation_analysis_v1/confirmation_analysis.json`；
- primary-only敏感性：`outputs/pm_v1_5_v5_2_confirmation_analysis_primary_sensitivity_v1/confirmation_analysis.json`；
- 技术报告：`outputs/pm_v1_5_v5_2_confirmation_final_report_v1/report.html`。

## 26. 2026-08-04 外部验证、基线与 ES-MemEval 修订

本节覆盖第5.2、9、11和13节中已经过期的外部执行细节；内部 V5.2 结论和停止规则不变。
完整可执行版本见 `docs/PM_V1_5_EXTERNAL_VALIDATION_AND_ESMEMEVAL_PLAN_20260804_ZH.md`。

### 26.1 三条证据线，不再互相冒充

- ESConv 只承担 RS、memory absent、即时 quality/risk/cost 的外部 repaired replication；
- EvoEmo response 承担同一用户私有 MP_PROFILE/MS/ME 的运输、路由、grounding 与 cost；
- ES-MemEval QA 承担信息提取、时间推理、冲突检测、abstention、用户建模以及检索/回答诊断。

QA 不能替代回复质量，response 盲评也不能证明检索答案正确。内部受控环境继续承担
MP_PREFERENCE、wrong owner、stale/conflict、联合16动作等外部数据未覆盖能力。

### 26.2 官方数据版本口径

论文报告1,209道QA，但当前官方GitHub/Zenodo v1.0.0和本地
`data/external/evo_emo.json`实际包含1,427道：IE309、TR284、CD267、UM306、Abstention261。
本地SHA256为`f30698e87fddaeff51270a666c654da604f487a3456ec60d2b6ae08a6fecd420`，与官方
commit/tag `692624208acc077b8867698c1d6fcd998dee641a`一致。因此本项目报告“公开v1.0.0 artifact
1,427题”，不声称精确复现论文1,209题筛选集。

p1–p6/p7–p12/p13–p18分别含519/490/418题。V5.1已暴露p7–p18的response任务，所以p13–p18
不能再称用户级未触碰lockbox；只有静态审计证明其QA answer/evidence从未被读取时，418题才可称
task-disjoint QA evaluation，否则全1,427题均按development-informed benchmark报告。

### 26.3 最终基线矩阵

response 主表固定为always-off、component-fixed-high、transparent-rule、learned-PM-full和
cost-matched-fixed。旧D3表面冻结的`MP+RS`只作历史证据；当前cost-matched必须在V5.2表面上只按
input-token一次性重算，禁止读取回复、quality、risk、judge或外部outcome。若某域的cost arm与已有
策略完全同臂，只记alias，不重复调用。

Raw Session Top-4 + Strategy和All Raw Sessions + Strategy只进入EvoEmo次表；Legacy V1.0只作
历史次表，除非能证明旧动作到当前同栈的精确投影。learned-PM-conservative只有既有冻结定义能无歧义
映射到V5.2时才作敏感性分析，不能根据当前结果重新挑组件。

ES-MemEval QA 比较no-memory、full-history、官方session-RAG Top-4、typed-memory fixed-high和
learned PM；主指标F1、BERTScore与按五能力分层结果，LLM judge只作稳健性。gold answer、capability、
evidence ID不得进入检索或生成。

### 26.4 E1–E5前置合同已冻结，下一步为显式批准后的单次E4生成

E1零API审计已通过，产物为
`outputs/pm_v1_5_v5_2_external_e1_static_audit_v1/`。封存V5.2实现、外部surface输入、每用户隔离、
官方1,427题身份均通过hash/结构检查；未发现QA字段被response路径访问，也未在既有T5 prompt/outcome
中发现逐字QA question。由于完整JSON曾被loader技术解析，p13–p18的418题只能称带技术解析披露的
task-disjoint QA evaluation。

当前outcome-blind cost-match冻结为ESConv=`M0+R0`、EvoEmo=`M0+RS`；前者与always-off完全别名，
不重复生成。V5.2原子ME在EvoEmo自然历史上的可执行覆盖仅为p7–p12 `8/60`、p13–p18 `0/78`，
这是外部表示覆盖边界，不允许从外部结果回改compiler。response核心去重后为1,576次调用；QA主实验
为2,090次调用。

E2零API逻辑计划已通过，产物为`outputs/pm_v1_5_v5_2_external_e2_plan_v1/`：核心response 1,576次，
EvoEmo Raw Top-4/All Raw次表双seed共552次，response合计2,128次；QA为418题×5条件=2,090次。
question可进入generation plan，answer/evidence/capability只存在于单独的evaluator-only映射。只有418条
no-memory QA现已具备exact messages；其余1,672条明确等待E3检索或上下文装配，未伪报exact预算。

E3本地retrieval-only审计已通过，产物为
`outputs/pm_v1_5_v5_2_external_e3_retrieval_v1/`。BAAI/bge-m3使用固定本地snapshot、完整session单元和
Top-4；552条raw response与2,090条QA均已封exact messages/hash并通过20k保守上下文上限。341个可完整
映射evidence的QA题上Recall@4=`.6963`、nDCG@4=`.5997`，另77题不强算。QA的learned treatment只在
392题选择MS、26题全关，MP/ME/RS不构成该条件中的外部成功证据。

E4统一付费runner和执行封条已经完成，分别为
`scripts/v1_5/28e_run_v5_2_external_e4_generation_v1_5.py`与
`outputs/pm_v1_5_v5_2_external_e4_plan_v1/`。runner从第一版即调用active
`require_paid_run_release`，校验E2/E3、V5.2 seal、generator、seed、exact prompt与预算hash，并用持久
attempt ledger断点续跑。4,218个逻辑call包括2,128次response与2,090次QA；一次成功费用代理上界
`$3.373812`，三次传输尝试绝对硬上界`$10.121435`，fresh identity为
`2ae8f4f047f6998c8f7bd58ffeaae1e5e1cd9e2fdffff095ca56fdc109fd1579`。合法`M0+R0`不等于fallback；
资源调用或postcondition失败不得改变requested action。当前API=0，只有中央manifest明确批准该identity
后才执行。旧23m及已封存runner不在依赖链、不原地改写。E4完成后才做自动评测、独立LLM judge和唯一
最终分层人评；外部结果不得修改V5.2，也不能救活内部完整系统门。

E5自动评分也已在任何E4 outcome产生前冻结，产物为
`outputs/pm_v1_5_v5_2_external_e5_scoring_v1/metric_freeze.json`。QA按官方公开实现逐字复现token F1与
BERTScore，并在首项评分前记录`bert-base-uncased`唯一local snapshot及完整文件树hash；response自动项
仅计算cost/coverage/routing，不用词法guard冒充质量或grounding risk。LLM judge仍是E6稳健性指标，
最终主张仍由冻结的人评构念与系统比较支持。

## 27. 2026-08-05 V5.3 新方法边界：证据整合替代机械追加

外部E7解盲与源码核验确认，V5.2的generator只根据当前对话写`primary_response`，随后后端将
`locked_clauses`逐字追加。该结构能防止资源被自由改写，却不能让generator围绕证据组织、桥接或约束
整条回复。EvoEmo learned核心状态中46/48包含MS，而这些动作相对always-off几乎全部失利；责任是
Step1的MS过度开启与Step2机械追加联合造成，不能靠重评旧质量或把全部旧信息判stale解决。

这里的`An earlier session recorded`、`I am keeping that as past context`和`current practical constraint
on record`不是generator读到内部prompt后的复述泄漏：generator根本没有收到MS/ME正文。它们由后端renderer
确定性编译并被guard要求逐字出现在最终回复中。EvoEmo risk overlap的25条中11条、主risk packet的124条中
53条出现该wrapper，证明外部表现真实存在；但自然来源说明不自动等于material risk，控制scaffolding暴露、
treatment解盲和回复是否真正有害必须分别评测。

V5.2保持冻结，旧结果继续作为该架构的有效诊断；任何修复必须升为V5.3，且executor改变后旧
`worth_opening`标签和value heads不再适用。V5.3保留Bank、retriever、候选身份、四资源ontology与16动作，
只重做受executor变化影响的执行、ITT效用标签、Step1 value heads和确认链。

V5.3使用`Top-k发现、exact Rank-1执行`、typed response program和evidence-aware whole-response
generation；禁止`primary_response + clauses`。Step1新增component-specific contribution slot，尤其区分
MS“有同主题旧记录”和“当前回复确实需要该具体连续性证据”。所有主baseline共享同一新executor和
generator，并增加cost/on-rate matched random以证明learned收益不是仅来自少开资源。

V5.3在任何新API前还必须冻结逐state `stagewise_accountability_ledger`，分别记录retrieval Rank-1与owner/time、
四门资格、PM概率和动作、requested/realized、evidence use、functional contribution、grounding、fallback及QRC，
并按`retrieval_fit × pm_correct × execution_valid`分层归因。generator自报trace只作遥测，不能作为做功gold。
正式语义评测仅允许一次FIT、一次fresh confirmation和一次合并外部复制；每波只含primary、预冻结20% overlap
和一次集中裁决，禁止再以小评审包驱动prompt/阈值修正。

新方法的完整阶段、数据规模、评测上下文、baseline和停止门见
`docs/PM_V1_5_V5_3_INTEGRATED_EVIDENCE_EXECUTION_PLAN_20260805_ZH.md`及
`data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json`。当前状态只是`PROPOSED`，尚未授权
API、生成新outcome或修改V5.2结果。科学结果不能预先保证；可保证的是一次性同栈执行、清晰归因和
不再以旧外测反馈循环调到通过。
