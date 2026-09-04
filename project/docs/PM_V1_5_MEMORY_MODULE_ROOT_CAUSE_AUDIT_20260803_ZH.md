# PM V1.5 记忆模块历次失败根因审计与一次性修复方案

状态：`ACTIVE ROOT-CAUSE DECISION / 2026-08-03`

## 1. 结论

MP、MS、ME 并没有在同一个正确任务上“反复训练失败”。过去被口头合并为“记忆模块失败”的
结果，实际来自六个不同层级：候选构造、检索、Eligibility/Observation、Step 1 标签与特征、
Step 2 执行、端到端回复效应。当前 V3 的正式 `worth_opening` 四个 head **尚未训练**；最新
28 条 H-Step2 也只检查强制给定资源后的执行，不检查 PM 是否开对。

最新 28 条不能按原门槛判定“MP、MS和多组件协调不合格”：

- 8 条多组件中的 5 条含 MP，而这 5 条都把 MP 候选的“短时私密窗口”逐字写进当前消息；
  MP 在生成前已经冗余，不可能获得“提供新增信息”的做功判定。
- 另 1 条 MSE 的 MS 把“区分两类压力”说成“过去证明有帮助”；机器校验把全部资源合成一个
  `authorized_corpus`，ME 中的 `helped` 错误替 MS 提供了授权，掩盖了跨组件错归因。
- 单组件 MP 的两个失败中，一条是 generator 没执行“先反映再提问”；另一条偏好仅在“采用反映
  时”生效，而回复选择了提问，题目没有激活可观察功能，属于测量歧义。
- 单组件 MS 的两个失败使用同一种不完整候选：资源只说“仍有一个 exact wording question”，却
  没保存该问题的具体文本。generator 客观上无法恢复不存在的内容；其中一条又把当前话语误称为
  `Last time`。
- 单组件 ME 为 4/5，唯一失败是有动作—结果资源但只复述过去，没有把它变成当前可选方案；这是
  清晰的 Step 2 执行深度问题。
- RS 的 4/5 做功不能掩盖 3 条 misuse：一条 `trusted person` 卡在没有已识别安全对象且目标为环境
  调整时被错误送入，属于候选资格错配；两条复述后追加问题，暴露“原子卡”和通用 prompt 允许
  `paraphrase + one question` 的合同不一致。

所以本轮 22/28 `no material misuse` 是有价值的开发证据，但不是正式通过率。原 28 条冻结为
`DEVELOPMENT_DIAGNOSTIC_INVALID_FOR_STEP2_PROMOTION`，不事后删除坏题重算，也不拿来否定 PM。

## 2. 先把三个问题拆开

| 问题 | 正确责任层 | 需要的证据 | 不能用什么替代 |
|---|---|---|---|
| 是否调用了可由后端证明的错用户、失效版本、缺失候选或显式禁用资源 | Retrieval + objective hard gate | exact Rank-1、owner/version/active、显式边界 | 回复质量胜负 |
| 候选在语义上是否服务当前目标、是否非冗余 | PM Step 1 的透明输入因子 | goal/function、specific increment、current echo | 把不确定语义一律硬关 |
| 合格候选是否值得花成本开启 | PM Step 1 | 已资格化 Step 2 下的同状态、单 bit、重复 seed 净效应 | 候选看起来相关 |
| 开启后是否把记忆正确写进回复 | Step 2 + generator | exact evidence binding、functional contribution、misuse | trace 自称 `applied` |

“最起码不要调用错”的可证明部分由第一行保证；无法从结构元数据确定的目标匹配和增量性进入
Step 1，而不是伪装成确定性 hard gate。“它是否真的带来额外收益”仍是 Step 1 的学习目标；
“已经开了但回复没用好”属于第三行。三者必须分别合格，不能继续用一个 A/B 人评覆盖全部责任。

## 3. 历史时间线：为什么看起来每次坏的组件都不一样

| 阶段 | 当时的表面结果 | 真实测量对象 | 后来确认的根因 | 当前效力 |
|---|---|---|---|---|
| controlled memory heads | MP BA .703、MS .895、ME .625 | 人工构造 applicability proxy | 没有 clean component treatment，不能证明效用路由 | 历史诊断 |
| transport-repaired 256 contrasts | MP/MS/ME OOF BA .503/.468/.445 | 单次回复效应标签上的首轮 Step 1 | 标签噪声、特征数超过 group 容量、背景身份被压成计数 | 根因 baseline |
| real-text opportunity heads | 三个 memory heads 曾显示 promoted | 构造 condition 的 opportunity | label 与 feature 同源，模板/背景捷径；不是独立 gold | 晋级撤销 |
| scale-stable repair | ME 通过，MP/MS specificity .50 | 内部构造确认与数值 transport | candidate subtype/语义覆盖没有随数值范围自动成立；ME 后又发现 candidate-gold 错绑 | 仅表示诊断 |
| D3 回复效应 | MP 7 on / 13 off / 20 tie；ME 15/19/6；MS 1/1/38 | 固定 treatment 的回复效应 | MP/ME 有部分真实效应但 repeat 稳定性弱；MS 首版 query 受 supporter 文本污染，替换后 treatment 仍缺增量 | 系统效应开发证据 |
| 10/32/16 条 Step2 门 | 不同轮次轮流出现 MP/MS/ME 空转或误用 | generator 功能与 grounding | prompt 只要求提及、时态升级、虚构 recall、candidate 无增量、联合合同冲突 | 小型开发门，不是训练 |
| 当前 V3 28 条 H-Step2 | 单组件 MP 3/5、MS 3/5、ME 4/5、RS 4/5；multi 2/8；misuse 6 | 强制资源后的 Step2 | packet 选择未绑定人工 eligibility、multi MP 回声、MS 候选缺正文、跨组件授权、原子性合同错位 | 正式晋级资格撤销 |

这些数字变化不是“ME 先学会又忘了、MS 忽然取代 ME”。每轮题目、层级和标签都不同；旧项目
最大的治理错误就是省略 `版本 × domain/split × 层级` 后只说“某组件过/没过”。

## 4. 全部根因

### 4.1 数据与候选合同（首要根因）

1. **同名异构**：内部 MP 曾几乎只有一种支持偏好，EvoEmo MP 是 profile facts；内部 ME 曾把
   任意过去事件当可复用结果；MS 曾混入 current-text fallback。
2. **candidate—gold 错绑**：标签描述预设的好记忆，实际 Rank-1 却可能是背景句、重复句或另一
   subtype。ME 的一次“完美晋级”已经因此撤销。
3. **候选内容不完整**：`there was one exact wording question` 是关于记忆的元描述，不包含问题
   本身，无法支持连续性回复。
4. **当前回声**：资源内容已在当前消息完整出现，却仍被当作增量 treatment。最新 multi MP 5/5
   属于这一类。
5. **训练子集不代表外部候选面**：完整 compiler 的数值支持不等于被抽入标签集的 subtype、
   exact candidate 和语义机会覆盖外部 EvoEmo。
6. **有效语义族太少**：换 topic 不能增加独立信息；旧 48 行往往只有 7–8 个 condition family，
   模型可复现造题规则却未必能迁移自然候选。

### 4.2 Observation、Eligibility 与 Step 1 被混写

1. “安全、相关、非冗余”曾被直接当 Step 1 gold；这只能保证不明显调用错，不能证明值得付成本。
2. 另一阶段又把一次随机 A/B 赢家直接当 gold；这会把 generator 随机性和主观偏好写进 PM。
3. 当前正确拆分是：objective hard gate只挡后端可证明的错误；语义goal/increment作为透明因子进入
   Step 1，由它学习**候选的预期净增益**。只有为了单独隔离Step2能力的32条gate，才要求人工确认
   四门全部eligible。
4. 最新 H-Step2 单例只用 `step1_candidate_admissible` 的结构化 hard gate 选样，没有要求冻结的人评
   Eligibility 为 yes；因此 trusted-person 等不合格候选被送给 generator，不能归责 Step 2。

### 4.3 训练表示与算法

1. 旧特征只含相似度、token、age 和目录规模，无法表达“同主题但错 owner”“过去有事件但无
   outcome”“偏好只在当前 move 下适用”等关系。
2. 首轮正式 fit 在 28–32 个 group 上用了 8–9 个特征，并把其他三个组件压成一个 count，违反了
   预冻结低容量和背景身份合同。
3. BGE-small/BAAI 提供相似度，不天然提供 owner、否定、时效、条件作用域、结果字段和冗余关系；
   它失败不是反逻辑，也不能通过换更大 embedding 修复上游错标签。
4. 固定 L2 logistic 仍适合 V1.5；前提是输入为 5–7 个独立、可观察、source-specific factors，
   且标签由另一条链产生。当前没有证据表明必须换深模型或复杂 loss。

### 4.4 Retrieval

1. 共享 compiler/schema 不等于同一语义 treatment；Top-1 文本必须单独审核 subtype 和用途。
2. MS 首版 query 含 supporter 文本，实验性元提示污染排序；seeker-only 修复证明 64/64 相关，
   但相关不等于有增量。
3. 当前 item 排序主要是词法检索；BAAI 并未成为 MP/MS/ME 的正式 item ranker。论文不能把
   source-level embedding 或 feature challenger 写成 BGE 已负责完整记忆召回。

### 4.5 Step 2 与 generator

1. **表面引用不等于做功**：提到“上次”后给通用建议，或提到 ME 后没有把它变成当前选项。
2. **时态和认识论错误**：过去事实被升级为当前事实/原因，或把当前话语说成“上次说过”。
3. **fabricated recall**：资源没有 outcome，却生成“你上次发现这很有帮助”。
4. **跨组件授权漏洞**：bundle validator 用全部资源的 union 审每个组件；一个组件的 outcome 可替
   另一个组件的不实 claim 背书。
5. **原子性不一致**：RS card 想要一个 atomic move，通用 prompt 却允许复述后再问一个问题。
6. **联合动作只合法、不一定可执行**：16 个 bit 组合必须保留，但每个 state 还要做 joint
   feasibility；不能在三句话内强迫四个资源各自增加一个动作。

### 4.6 评测与治理

1. 10/16/28/32 条多数是资格门或 smoke test，不是每次新训练集；按“最新一包”覆盖旧结论是错的。
2. `no material misuse` 不是完整通过：当前 22/28 只说明 22 条没有阻断性误用，不说明所有请求
   组件做功，也不说明 PM 开关正确。
3. Step2 gate 必须只使用预先确认 eligible、nonredundant、joint-feasible 的 exact candidates；
   否则题目失败和 generator 失败无法区分。

## 5. 一次性修复，不再逐组件打补丁

### 5.1 冻结三类 memory surface

- `MP_PREFERENCE`：只在当前回复动作会触发该偏好时 eligible；硬偏好由确定性执行约束落实，
  软偏好必须在相反偏好配对中可证伪。
- `MP_PROFILE`：必须是当前未出现、与当前目标相关、不会被当作原因的稳定事实或实际约束。
- `MS_SESSION`：必须保存具体旧观察、具体区分或具体未完成问题的正文；禁止仅存“曾讨论过/有个
  问题”的元描述。
- `ME_REUSABLE_OUTCOME`：必须含过去动作或选择，以及结果或机制；context-only 与 unresolved
  event 不得进入 ME-on treatment。

每个已开启 memory 只注入 exact Rank-1 一条。Top-k 只用于 candidate discovery/诊断，不把三四条
全部塞给 generator。

### 5.2 替换 H-Step2，而不是重做训练或外测

当前 28 条保留但撤销晋级效力。只做一份版本化 replacement gate：

1. 单组件样本只从已经冻结的 128 条 H-Eligibility reference 中 `eligible=yes` 的 exact decision
   surface 选择；每个 component 6 条，覆盖 MP 两 subtype、MS 具体事实/区分、ME action-outcome、
   RS card family。
2. 8 条 multi 由这些已合格单项组合；构造前检查每个候选相对当前消息非冗余，并检查全部
   `required_contribution` 在当前负担上可同时成立。
3. 共 32 条，一次生成、一次功能/misuse审核；它仍只是 Step2资格门，不是 PM 训练或论文效果样本。
4. 门槛：单组件每类至少5/6真正做功；multi至少6/8全部做功；fabricated recall=0；明确边界违反=0；
   总 material misuse≤1/32；所有机器 fallback 可采用。

### 5.3 Step 2 合同修复

1. structured trace 增加每组件的 `response_evidence_excerpt` 和 `resource_support_excerpt`；两者分别
   必须逐字存在于最终回复和该组件自己的 exact resource。
2. fabricated/outcome/temporal 检查按组件自己的 evidence 做，不再用资源 union 给每个组件背书；
   union 只允许做全局安全扫描。
3. MS 使用两种确定性模式：`answer_past_fact` 或 `tentative_continuity_check`；ME 使用
   `past_action_outcome → tentative_present_option`。来源和当前适用性必须分开。
4. MP preference 先作为生成约束，再用可机检的格式/负担 postcondition 审核；如果当前 move 不
   触发偏好，Step1 前就标为不适用，不能强行要求“可见做功”。
5. RS atomic card 写入 `max_support_moves=1`；joint planner 计算 requested→feasible，冲突时关掉
   较低净值 bit，完整 16 动作接口不变。
6. 任一 component 证据错绑、无法做功或机器门失败：该 bit realized OFF；若共享回复已受污染，
   整条走一次 `M0+R0` fallback。此类样本在 H-Effect 中记 unknown，不能当 OFF gold。

### 5.4 真正训练 Step 1

通过 replacement H-Step2 后才执行既有 V3 effect blueprint，但每行必须重新满足：

- exact Rank-1 candidate 与 candidate_id/text/subtype 全绑定；
- objective hard gate通过；无错owner、失效版本、候选缺失或显式边界冲突；语义goal/increment
  正负情况按设计进入Step1，不能先用人工答案筛成全正；
- Step2 两个固定 seed 都有可验证执行；无效执行记 unknown；
- 每 head FIT 64、confirmation 32、sealed 32 个独立 user；至少 16/16、8/8、8/8 的 ON/OFF；
- semantic family 跨 split 隔离，topic、长度、history scale、background bits 在 target 内反平衡；
- nuisance-only BA<.65、零同特征异标签冲突、5–7个 primary factors、显式三个 background bits；
- grouped OOF：BA≥.65、recall/specificity≥.60、Brier 优于 prevalence 和 transparent rule；
- fresh confirmation 不用于调阈值；learned 必须超过或至少在 proper score 上稳定优于人工规则，
  才能称“学会了”，不是要求 80–90% 每条全对。

Primary 仍是 source-specific transparent factors + L2 logistic。BGE-small 只作一次冻结 challenger；
不再用 Qwen/NLI/调参替代数据合同。

## 6. 学习与外部考卷

- ESConv：只检验 memory unavailable 时 MP/MS/ME 正确 OFF，以及同一 80-card Bank 下的 RS；不能
  证明长期记忆能力。
- EvoEmo：只检验其真实覆盖的 `MP_PROFILE`、`MS_SESSION`、`ME_REUSABLE_OUTCOME`。内部训练使用
  content-disjoint synthetic users，但共享 compiler、query、retriever、candidate schema、Step1、
  Step2、generator 和 rubric；不替换成另一套 memory 机制。
- 内部 sealed：补 `MP_PREFERENCE`、明确边界、wrong-owner/stale/current-echo、joint conflicts 和
  全16动作。外部测不到的能力只作受控内部证据，不冒充自然泛化。
- 外部 subtype/目录/特征超出训练支持时 abstain/off 并报 coverage；不得把外部结果回流改 PM。

## 7. 现在能否确保“学出来”

能确保的是下一轮数据**可学习、可辨识、责任可归因**，并且 learned head 不会靠恒关、topic、长度、
背景 bit 或 construction marker 取得假高分。不能预先保证真实边际效应一定足以让四个 head 都超过
baseline；那正是论文要回答的科学问题。

不过“调用不要错”不必全部押在 learned effect head 上：结构化 owner/version/active/显式边界
hard gate先挡可证明的错误；learned Step1再处理目标匹配、增量性和是否值得花成本。这样即使最终某
个效用 head 只及格或无显著优势，也不会退回 V1.0 的全开误用，更不会把 generator 空转误写成
PM 学不会。

## 8. 当前唯一下一步

停止 effect generation、外测和新增零散人评。先版本化实现：

1. exact per-component evidence binding；
2. concrete MS payload 与 typed ME/MP application；
3. atomic/joint feasibility；
4. 从冻结 eligible reference 构造一次 32 条 replacement H-Step2。

只有该门通过，才进入 512 个 V3 clean-pair 的既定生成、H-Effect、四 head 训练、confirmation、
sealed internal、ESConv/EvoEmo 和最终 QRC。过去已完成的 Bank、Eligibility reference、compiler、
外部分区和绝大多数 blueprint 不重做。
