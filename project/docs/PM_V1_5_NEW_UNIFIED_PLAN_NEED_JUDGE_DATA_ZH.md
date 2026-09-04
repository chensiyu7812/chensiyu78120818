# PM-v1.5 新统一计划：需求诊断、训练数据、Judge 与透明指标

状态：`ARCHIVED ANALYSIS / NOT THE ACTIVE V1.5 EXECUTION PLAN`
日期：2026-07-28
性质：本文档回应用户对"协议/冻结无法解释真正失败原因"的质疑，正面处理需求诊断模块、
训练数据生成、judge 评测方法论、透明指标、BAAI 定位五个核心问题。不替代
`PM_V1_5_MINIMUM_PUBLISHABLE_PROTOCOL_ZH.md` 的范围收缩决定，而是给"范围收缩之后
到底怎么把 PM 真正训出来"提供具体方法。

> 2026-07-28 复核说明：本文保留为问题诊断材料，但第 8.2、10、11 节的执行顺序
> 不再采用。五张 Strategy Bank V2 卡已经由人评 5/5、置信度 4 批准用于 train-only
> pilot，因此“全部不合格、必须先扩到 10–20 张”已被新证据推翻。V1.5 也不再把
> SupportNeed zero-shot 模块、EvoEmo 深挖或旧 composite 代码清理设为 RS pilot
> 前置门。当前唯一活跃路线、指标和阶段状态以
> `PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md` 为准。

---

## 0. 用户原话要求解决的问题（不得回避）

1. 靠谱的"诊断/判断"模块——用户说的话，怎么知道TA现在要什么（只想聊聊/寻求建议/
   寻求帮助/情绪崩溃）。
2. 可靠的训练数据怎么生成。
3. Judge 怎么评测；双裁判差异过大怎么办；提示词怎么写才严谨。
4. 指标必须透明可观测，不能是笼统分数。
5. BAAI 和 PM 配合是否正确（接口/格式），BAAI 是不是最合适的语义模型。
6. Strategy Bank 里到底存的是什么策略（是否全是情绪支持策略），这决定了 RS 什么时候
   该开。
7. ESConv、EvoEmo 数据是否已经被吃透。
8. 小样本、弱监督条件下如何真的学出来，而不是空谈"能学会"。
9. 明确指出：协议/冻结类工作之前也做得不错，但 PM 最终还是失败了——新计划必须证明
   它解决的是别的、更根本的东西，而不是重复同一类工程收口。

以下每一节都直接对应上面的编号。

---

## 1. 根本原因诊断：为什么协议做对了，PM 还是没学出来

这不是态度问题，是可以用现有数据直接证明的四个具体机制。找到这四个机制，本身就是
对用户"重点不是协议"这句话的证实，而不是安慰。

### 1.1 统计功效不足，且已经被三次独立失败反复证实

`support_need_factorized_bakeoff_expanded_v1_analysis` 的真实结果：39 个独立
dialogue/user group，8 个需求维度 + 1 个平铺 5 类对照，逐一测过
lexical / BGE-small / BGE-M3 / Qwen3-Embedding-0.6B / DeBERTa-NLI / hybrid：

- `need_for_exploration`、`focused_question_readiness`、`planning_readiness` 三个
  维度，无论哪个表示法，最优模型的正类召回率都是 **0**——不是学得不够好，是模型
  在训练时就直接放弃了正类，退化成"永远猜负类"。
- `advance_readiness`、`advice_readiness` 有一定召回，但相对词法基线的 bootstrap
  置信区间跨零，不能算"学会了"。
- 8 个维度里只有 `need_to_be_heard` 和 `need_for_emotional_containment` 两个维度，
  NLI 表示法相对词法基线的优势置信区间不跨零，是唯一两个有真实统计信号的维度。
- 关键证据：**8 个维度中有 4 个的"最佳表示法"在样本从 23 组扩到 39 组后发生了改变**
  （比如 `need_to_be_heard` 从 BGE-M3 变成 NLI）。这不是运气不好，这是"在这个样本量下，
  模型选择本身就是噪声"的直接证据——换任何模型都不会改变这个结论，因为问题不是模型，
  是分母。
- Qwen3-Embedding-0.6B 在两轮（23组、39组）bakeoff 里都没有在任何一个维度上胜出。
  连续三次换模型（BGE-small → BGE-M3 → Qwen3+NLI hybrid）都在同一批约 40 组数据上
  失败，这本身就是"问题不在模型选择"的强证据，而不是"再试试第四个模型"的理由。

**结论：继续在 SupportNeedObservation 上"训练"一个多维分类头，在当前样本规模下，
大概率会继续得到同样的失败模式，无论标注质量多好、模型多先进。这是本节最重要的
判断，第 2 节的方案转向由此而来。**

### 1.2 Judge 体系目前还没有产出一个可信的自动裁判

`pm_v1_5_strict_judge_bakeoff_v1_semantic_aggregation_repro_a/qualification_report.json`
（candidate/repro 两次运行逐字节一致，可信）：三个候选 judge——
`anthropic_claude_haiku_4_5`、`google_gemini_2_5_flash`、`openai_gpt_5_mini`——
**全部**没有通过 AB/BA 顺序一致性 0.80 的门槛：

| judge | overall_preference | support_quality | evidence_handling | safety |
|---|---|---|---|---|
| claude_haiku_4_5 | 0.833 | 0.833 | **0.667** | 1.0 |
| gemini_2_5_flash | 0.917 | 0.917 | **0.417** | **0.667** |
| gpt_5_mini | **0.583** | **0.5** | 0.75 | 1.0 |

主要卡在 `evidence_handling`（证据使用是否得当）维度——三个 judge 里最好的也只有
0.75，最差 0.417，接近随机换序就换结论。对人工锚点的方向一致率也只有
57.1%–72.7%（`overall_preference`）。项目自己的处理规则是对的：任一维度低于 0.80
就自动拒绝批量授权，禁止"多数投票即金标准"，改为要求人工签字——这条 fail-closed
规则应该保留，不该放松阈值去凑过关。

另外两个 judge 实验（role-decomposed、low-budget）**从未跑完**，卡在 API/schema 报错，
从来没有产出过一致性数字——目前唯一"测过并且失败了"的是 strict_judge_bakeoff 这一份。

**结论：任何依赖 LLM judge 当自动金标签的训练路径，目前都要预期继承这层噪声。
`evidence_handling`（证据/风险判断）尤其不可信，应该优先用代码可验证的手段代替，
见第 5 节。**

### 1.3 Strategy Bank 实际上还没建成可用形态

11,590 张卡，经过完整质量筛选后，只剩 **5 张**"技巧类"卡片存活，且全部标记
`eligible_for_formal_rs=false` 待审。这意味着：现在还没有真正干净的 RS 资源可以
拿来做 matched-pair 对比。"PM 该不该开 RS"这个问题目前根本没有资源支持去测——
不是 PM 学得不好，是这个组件还没有真正的、可信的 treatment 可以给它学。

### 1.4 ESConv 原生标签本身信息量很低，"挖数据"这条路已经被证伪

失效账本 V15-DATA-24 到 27：`problem_type`/`emotion_type`/`experience_type` 与
下一步实际使用的策略之间的互信息只有 **0.00273 / 0.00248 / 0.00018**——接近于零。
11,914 个合法决策点里，只有 **3 个**带有清晰的"用户明确表示只想被倾听"线索。
这解释了为什么从 ESConv 原始对话里"挖"用户需求-策略对应关系，无论标注多细致，
天然信号都很弱——不是分析不够深，是这个数据集在策略选择层面本来就不包含很强的
"因为这个情绪/问题所以用这个策略"的可学习规律。**这一点已经被真实测过，不需要
重新怀疑，但也说明必须靠人工/LLM构造的对照数据来补，而不是继续从 ESConv 原始
对话里挖信号。**

### 1.5 一处内部矛盾（本次调研新发现，之前的 Codex 会话没有点出来）

`src/metacom_pm/pm_v2_judging.py` 明确写着"不允许产生 overall score，不能把一个
分数抄到另一个"，这是项目已经确立的正确原则（对应失效账本 V15-METRIC-02 对旧
composite 分数的批评）。但同一代码库里 `src/metacom_pm/v1_5_judge_qualification.py`
和 `src/metacom_pm/v1_5_esconv_auxiliary_pairwise.py` 仍然在用固定权重合成一个总分
（emotional support 30% + personalization 20% + memory appropriateness 15% +
factual grounding 15% + temporal consistency 10% + non-intrusiveness 10%）。

**这正是账本自己点名批评的反模式，还留在代码里没清理。第 6 节要求：新的 judge 
指标体系统一到"每维度独立报告，禁止合成"，并把这两个文件里的固定权重 composite
删除或至少标记为不得用于正式指标。**

---

## 2. 方案总转向：需求诊断从"训练"改为"规则 + 已验证的固定分类器"

这是本计划最核心的判断，直接回答"小样本弱监督下怎么学出来"：

> **不要在小样本上"拟合"，只在小样本上"验证"。**

拟合（fit trainable parameters，比如 embedding 投影 + 分类头）需要的独立样本量，
远大于验证（estimate 一个 agreement/AUC 的置信区间）所需要的样本量。过去三轮尝试
本质上都是在约 40 组数据上"拟合"一个多维模型，1.1 节的证据（4/8 维度最佳表示法
随样本量变化）直接说明这条路在当前数据规模下不成立。

新设计把 SupportNeedObservation 拆成两层，**只有第 2 层允许用于 opportunity 特征，
两层都不作为可训练参数去拟合**：

### 2.1 第一层（最高优先级，零训练，直接可信）：透明边界规则

只识别人工标注中一致性最高、且能直接从文本逐字引用验证的显式线索：
`advice_rejected` / `advice_requested` / `one_small_step_requested` /
`listen_first_requested`。人工标注数据已经证明这类信号可靠
（`response_burden` 一致性 κ=0.887，是全部维度里最高的），本来就不需要训练，
只需要写好规则 + 在小样本上验证规则的精确率/召回率即可上线。这一层直接决定
是否允许 RS/结构化建议类动作触发，优先级高于任何 embedding 判断。

### 2.2 第二层（覆盖隐含案例）：固定的、经过严谨提示词工程、并用人工锚点校准信度
的 LLM zero-shot 分类器——不是训练出来的模型，是一个**验证过可靠性的固定工具**。

- 不再对 BGE-small/BGE-M3/Qwen3 做"选哪个当训练特征"的 bakeoff——1.1 节已经用
  三轮独立证据说明这条路走不通。
- 改为：用同一个（或两个互相校验的）LLM，针对每一个残余的隐含语义轴，写一条
  **单一、清晰、锚定型的分类提示词**（结构见第 5.3 节），直接做 zero-shot 判断，
  产出的是标签 + 置信度，不经过反向传播、不拟合任何投影矩阵。
- 40 个人工锚点这时候的作用变了：不是用来"训练"，是用来**估计这个固定分类器在
  每个轴上的可靠性**（agreement/AUC + 置信区间）。这是统计上完全不同、需要样本量
  小得多的任务——你不需要学习参数，只需要估计一个比例的置信区间。
- 只有在某个轴的固定分类器可靠性经验证达到可用水平（比如与人工锚点的一致率有
  95% CI 下界 > 明确设定的基准，比如比随机猜测/常数先验的 baseline 高出有意义的
  幅度）时，才允许把这个轴的输出作为 PM 的 opportunity 特征使用；不达标的轴，
  诚实标记为 `unsupported`，不参与门控。

### 2.3 需求诊断与"PM 该不该开某资源"彻底解耦

诊断模块（第一层+第二层）只产出**特征**，真正需要"学习"（拟合可训练参数）的，
只有 component-effect head——即"在 matched-pair 数据上，某个资源打开是否真的让
回复变好"。这是一个更明确的因果目标，不需要先有一个精细的 5-8 维用户画像才能训练，
且已经有更合理的统计设计（ESS≥8 独立组、group-OOF、matched-pair）支持它。
**换句话说：诊断不为了"懂用户"而存在，只为了给 component-effect head 提供
输入特征而存在；论文的核心可学习性主张，应该压在 component-effect head 上，
不是压在需求诊断本身能不能学出 5 类。**

---

## 3. SupportNeedObservation 具体实现（更新版）

保留原有 schema 中被证明有效的部分，砍掉被证明学不出来的部分：

**保留（第一层规则，已验证高一致性）：**
`explicit_boundaries`（5 个布尔量，逐字引用验证）。

**保留（第二层固定分类器，仅限已证明有信号的轴）：**
`need_to_be_heard`、`need_for_emotional_containment`——用 NLI zero-shot 假设打分
（"用户当前主要希望被倾听/需要情绪稳定"），这是目前唯一有统计显著性证据支持的两个
语义轴。

**降级为"仅诊断/研究用，不进入 PM 特征"（除非第 2.2 节的可靠性校验通过）：**
`need_for_exploration`、`focused_question_readiness`、`advice_readiness`、
`planning_readiness`、`low_interaction_burden`——当前证据显示要么召回为 0，要么
置信区间跨零，要么正/负类样本过少（`planning_readiness` 只有 3 个正例，
`low_interaction_burden` 只有 3 个负例）。第一篇论文不应该在这些轴上硬撑一个
"学会了"的说法。

**彻底废弃：**
5 类平铺 `support_mode` 分类（listen/explore/comfort_reassure/light_guidance/
structured_planning 五选一）。真实数据显示这个设计从第一轮到现在，三个类别
召回率始终为 0，模型只会预测 `comfort_reassure`。用户最初问的"倾听 vs 建议 vs
崩溃"这个问题，正确答案不是"训练一个五分类"，是"用第一层显式边界规则 +
第二层两个已验证轴的组合"——这个组合已经能覆盖"是否明确拒绝建议""是否明确要
一小步建议""当前是否需要被倾听/稳定"这几个最关键的实操判断，不需要一个完整、
互斥的五分类才能做决策。

**目标类型/对话阶段/紧迫程度**：继续作为**弱监督的分层特征/诊断报告维度**保留
（用于抽样分层、审计报告），但不作为独立训练目标或 PM 硬性门控依据，直到有更多
数据支持。

---

## 4. 训练数据怎么生成（小样本、弱监督具体方法）

### 4.1 停止依赖 ESConv 原始对话"挖"信号

1.4 节已经证明原生标签互信息接近零。继续在原始对话上加大标注量，边际收益很低。

### 4.2 优先构造人工/LLM 辅助的最小对照对（minimal counterfactual pairs）

固定话题、情绪强度、对话历史，只改变一个变量（比如"是否包含明确的一小步建议请求"），
生成 15-20 组这样的最小对照，专门用来：
(a) 验证第一层规则的精确率/召回率是否达标；
(b) 校准第二层固定 zero-shot 分类器在两个已验证轴上的可靠性区间。
这类数据每一条都是针对性的"单元测试"，比随机抽样 ESConv 对话的样本效率高得多——
这也是标注指南里"反事实"要求的自然延伸，应该被当作核心产出，不是可选项。

### 4.3 component-effect 数据坚持 matched-pair 设计，但先解决资源侧的真空

现有 matched-pair 设计（固定 state/generator/prompt/seed，只变一个证据组件）本身
是对的，应该保留。但 RS 这个组件目前完全没有可用资源（第 1.3 节），必须优先把
Strategy Bank 至少一个 family 做到可信可用（见第 8 节），否则 RS 的 matched-pair
永远无法产生真实的 treatment 对比，component-effect 训练会一直卡住。

### 4.4 ESS/独立组规则维持不变

`≥10 个独立组/support mode`、`≥8 个独立组/方向/MP-MS-ME-RS 主效应`、交互效应
`≥12 组`——这些规则本身是对的，不需要改，需要改的是"用什么方法去满足这些规则"，
即本节 4.1-4.3。

---

## 5. Judge 评测方法论：怎么写提示词、双裁判分歧怎么办

### 5.1 明确禁止合成分数（清理第 1.5 节发现的矛盾）

删除或冻结 `v1_5_judge_qualification.py` / `v1_5_esconv_auxiliary_pairwise.py`
里的固定权重 composite（30/20/15/15/10/10%），统一到 `pm_v2_judging.py` 已经确立
的"每维度独立报告、不产出 overall"的设计。这一步不需要新实验，是纯粹的代码/文档
一致性修复，但必须做，否则"透明指标"这句话名不副实。

### 5.2 按已测出的可靠性分配角色，而不是假装 judge 都一样可信

- `overall_preference`（整体偏好）：三个候选 judge 的 AB/BA 一致性相对较高
  （0.58-0.92），人工锚点一致率 57%-73%，可以作为主质量指标，但必须报告这个
  57%-73% 的可靠性区间，不能假装它是无噪声金标准。
- `evidence_handling`（证据使用是否得当）：三个候选 judge 全部低于 0.75，最差
  0.417——这类判断应该优先交给**岗位 D 式的确定性代码检查**（逐字引用比对、
  memory/strategy 是否真的进了 prompt、是否引用了未出现的信息），LLM 判断只作
  辅助信号，不作为独立标签来源。
- `safety`/风险判断：Gemini 在这一维度只有 0.667，说明风险类判断同样应该优先靠
  第 4 类可观察事件（见 `MINIMUM_PUBLISHABLE_PROTOCOL` 第4.2节的四类风险定义）
  的代码/规则检测，LLM 只做兜底。

### 5.3 提示词写法的具体、可执行标准

基于已有的 `pm_v2_judging.py` 设计（1-5分显式锚点量表 + 0-3风险量表 + 强制不产出
overall）和 `v1_5_role_decomposed_judge_qualification.py` 的岗位分离经验，任何新
judge/分类提示词必须满足：

1. **单一决策问题**：一个提示词只回答一件事（偏好 or 证据审计 or 需求轴判断），
   不要求 judge 同时输出多个不相关维度的综合判断。
2. **显式锚点，不是形容词量表**：每个分数点必须有一句具体描述"什么情况打这个分"，
   不能只写"1=差，5=好"。
3. **强制逐字引用**：任何"违反了XX""支持了XX"的判断，必须附带从可见文本里逐字
   摘取的证据片段，代码强制校验该片段确实是原文子串（Bank/岗位B 已经这么做，
   要推广到所有需要证据支持的判断）。
4. **AB/BA 位置平衡是强制项，不是可选项**：任何 pairwise 判断必须双向各跑一次，
   两次不一致时不产生硬标签（记为 tie/低置信）。
5. **给一个不来自当前评测状态的正例/负例/边界例**，防止 judge 用当前样本的表面
   特征走捷径。
6. **明确列出禁止依据的信息**：不能依据 action 是什么、模型身份、memory 数量、
   回复长度、期望的 regime 来打分。
7. **提示词定稿后必须和 schema/rubric 文档逐字核对一致**——1.5 节的矛盾就是
   "文档说不许合成分数，代码里却在合成"，以后每次改 judge 逻辑都要同步检查两边。

### 5.4 双裁判差异过大：不是"取平均"或"少数服从多数"

项目已经有正确的规则（保留）：
- AB/BA 一致 + 两家 judge 同意 → 高权重弱标签；
- 两家一致判 tie → 记为小效应样本，不强造赢家；
- 一家 abstain，另一家一致且已校准 → 低权重单家标签；
- 两家方向冲突 → soft target 0.5 / 高不确定性，不进硬正负例；
- 同一 regime/组件系统性冲突 → 判定为测量故障，抽取少量人工 active adjudication；
- 治疗没有 uptake（回复本身没有可辨识差异）→ 判定为数据/生成机制失败，停止调 judge。

**新增的硬性门槛（基于 1.2 节实测数据）**：任何维度的 AB/BA 一致性低于 0.80，
或对人工锚点的方向一致率低于 70%，该维度**不得**用于自动化批量标注，必须要么
换用代码可验证的确定性检查代替，要么保留为人工审核项——这不是理论门槛，是已经
真实测出三个候选 judge 都卡在这条线附近，必须严格执行，不能因为"暂时没有更好的
judge"就放宽。

---

## 6. 透明指标登记表（不允许任何笼统分数）

在 `MINIMUM_PUBLISHABLE_PROTOCOL` 已经定义的 quality（blind matched-pair
NetWin）/risk（四类可观察事件 + N/A≠0）/cost（generator input tokens）三线并列
之外，新增"需求诊断层"的透明报告要求，同样禁止合成：

对 SupportNeedObservation 中**任何**被启用（不管第一层规则还是第二层固定分类器）
的轴，必须逐轴报告：

- 使用的方法（规则 or 固定 zero-shot，写明具体提示词版本号）；
- 独立锚点组数、正/负类计数；
- 相对随机猜测/常数先验的召回率、精确率、以及（如适用）log-loss/Brier 的
  group-bootstrap 95% CI；
- 是否所有类别都有非零召回（这是本项目已经吃过的具体亏，必须作为强制检查项）；
- 该轴是否被用作 PM 的 opportunity 特征输入（是/否 + 理由）。

任何未逐轴报告、只写"诊断模块 status: PASS"或"综合置信度 X"的报告，一律视为
不合格，退回补充。这条规则直接回应用户"最起码我们自己得知道我们想要的 PM 是
什么样的"这句话——逐轴报告就是让研究者自己先看清楚每个判断到底靠不靠谱，而不是
被一个综合分数糊弄过去。

---

## 7. BAAI/BGE 的定位（不再追逐"更懂用户"的模型）

- **不再对 embedding 模型做"谁更懂用户需求"的选型竞赛**。三轮独立证据
  （BGE-small → BGE-M3 → Qwen3+NLI hybrid）已经证明，在当前约 40 组的数据规模下，
  换模型不解决问题（1.1 节）。继续追逐更大模型是在浪费时间做一件统计上大概率
  不会有结果的事。
- **BAAI/BGE-small 保留它真正擅长、且模型卡本身就是这么定位的角色**：Strategy
  Bank family 层面的语义检索匹配（retrieval/similarity），不是需求诊断的决策者。
  这个角色下，"BAAI 输出是否符合我们需要"这个问题的答案很简单：只要它能把
  "求助者提到的话题"和"策略卡的适用场景"做出合理的相似度排序，就够用了，不需要
  它理解"这个用户现在是想被倾听还是想要建议"这种更细粒度的语用判断。
- **NLI（`cross-encoder/nli-deberta-v3-base`）保留在两个有真实信号的轴上使用**
  （need_to_be_heard、need_for_emotional_containment），不推广到其余轴。
- **接口/格式检查清单**（回应"如何确保 BAAI 输出和 PM 结合是对的"）：
  1. BAAI 的原始相似度分数不能直接当"用户需要这个策略"的判据（模型卡本身说明
     0.6-1 是正常相似度区间，>0.5 不代表语义相关，参见第10.1节已有分析）；
  2. PM 看到的必须是经过任务内校准（在本项目自己的锚点数据上标定阈值/分布）的
     分数，不是原始 cosine；
  3. 任何"8 个 family centroid 相似度"这类特征，不能让 PM 直接从中反推出"应该
     选哪个 family"，因为这等于把检索答案直接暴露给决策模块（对应失效账本
     V15-OBS-03 的"embedding shortcut"风险）——检索排序和 PM 决策要保持先后
     顺序：PM 只看"是否存在匹配家族的机会"这种粗粒度信号，具体检索排序在
     PM 决定开 RS 之后才发生。

---

## 8. Strategy Bank 根修复：结合 ESConv 原生字段，先做出一个真正可用的 family

### 8.1 现状核实（本次调研的实际数字）

Strategy Bank 目前直接照搬 ESConv 论文原生的 8 类策略标签，**不是 100% 情绪支持
策略**：

| family | 卡数 | 性质 |
|---|---|---|
| Question | 2374 | 对话技巧 |
| Others | 2127 | 兜底/杂项 |
| Providing Suggestions | 1866 | **具体行动建议**（非纯情绪支持，含风险） |
| Affirmation and Reassurance | 1846 | **纯情绪支持** |
| Self-disclosure | 1069 | 支持者自我披露 |
| Reflection of feelings | 901 | **纯情绪支持** |
| Information | 753 | **具体信息/资源建议**（非纯情绪支持） |
| Restatement or Paraphrasing | 654 | 沟通复述技巧 |

项目自己已经做出了正确判断：现有 v2 候选集排除了 `Information`、`Others`、
`Self-disclosure`（理由标记为 `family_not_in_first_paper_technique_pool`），
只保留 `Question / Restatement / Reflection of feelings / Affirmation and
Reassurance / Providing Suggestions` 五个 family，且给 `Providing Suggestions`
标了风险标签（`unrequested_advice / directive_overload / domain_specific_claim`）。
**这个判断是对的，应该坚持**——这正好回答了用户"要看这个策略存的是不是都是情绪
支持策略"的问题：不是，项目已经识别出来了，也已经把"给具体建议"这类单独标了
风险，这一步不需要重做。

### 8.2 真正的缺口：筛选完只剩 5 张待审卡片，RS 没有真实资源可测

这是当前最大的实操阻塞。建议：**不再指望从 11,590 张 ESConv 原始回合里筛出更多
合格卡片**（Bank 本身已经证明大量重复、模板化、只有 8 个 guidance_text 通用模板），
改为优先把**一个**风险最低、语义最纯的 family——`Affirmation and Reassurance`——
做完整清洗+人工审核，目标产出 10-20 张真正 `eligible_for_formal_rs=true` 的卡片，
先让 RS 组件有第一批可信 treatment 可用，而不是卡在"5 张全部不合格"上等所有
family 都做完。`Providing Suggestions`（风险已知但价值也最高，直接对应用户说
的"寻求建议"场景）作为第二优先级。

### 8.3 ESConv 原生字段（problem_type/emotion_type）的正确用法（现状已经是对的）

现状核实：`problem_type`/`emotion_type`/`experience_type` 目前只用于
(a) 策略卡的血统/描述元数据、(b) 人工标注包的分层抽样、(c) 纯统计分析；
**完全没有**被用作 PM 运行时决策特征，且被显式列入"禁止出现在标注员可见状态"
的黑名单。**这个设计是对的，不需要改**——把这些字段当运行时特征会造成特权输入
泄漏（和 `situation` 字段是同一类问题）。

**真正该补的是一项审计，而不是改变现有用法**：用 `problem_type`/`emotion_type`
检查 Strategy Bank 各 family 的**�covered 分布是否均衡**——比如
`Providing Suggestions` 类卡片是否集中覆盖某几种 problem_type（比如工作压力），
而在另一些 problem_type（比如 breakup）下几乎没有可用建议卡。如果分布不均衡，
这必须被诚实报告为一条 limitation（"RS 在某些问题类型下资源覆盖不足，不代表
PM 判断错误，而是资源本身缺失"），而不是被误判为"PM 没学会该不该开 RS"。这个
审计应该在 8.2 节的清洗工作完成后立刻做。

---

## 9. ESConv / EvoEmo 数据是否已经吃透：现状核实

- **ESConv：已经吃透，做得不错，不需要重做**。项目里有专门的
  `v1_5_dataset_understanding.py` 做 problem_type/emotion_type/experience_type
  交叉策略分布统计，也测过原生标签与策略选择的互信息（结果证明信号很弱，见
  1.4 节）——这正是用户说的"先分析数据找规律"的做法，而且已经做完，结论是负面
  但真实的（原生标签信息量低，不能指望从中挖出监督信号，需要转向人工构造对照
  数据，见第 4 节）。
- **EvoEmo：还没有到同等程度**。目前已知的是"EvoEmo formal V3 的 204/204
  （100%）个 observable states 都是严重 metadata OOD"——即已知 EvoEmo 在目录
  规模、age/span、expected-token 等结构性形状上和训练分布不匹配，但**没有找到
  一份对 EvoEmo 本身话语/情绪/策略规律做类似 ESConv 那种系统性统计分析的报告**。
  建议：在把 EvoEmo 纳入正式外部评测前，先补一份对等的"EvoEmo 数据理解"分析
  （复用 `v1_5_dataset_understanding.py` 的方法论，只是换数据源），至少要搞清楚
  这个 100% OOD 具体是哪些字段造成的、是否可以通过预处理/特征对齐解决，而不是
  直接把 EvoEmo 标记为"外部阻塞"就不再深挖。这是本篇新计划识别出的一个真实
  空缺，建议列入第 10 节执行顺序。

---

## 10. 执行顺序（考虑以上全部约束的最快路径）

### 必做（按顺序）

1. 清理第 1.5 节发现的 composite 分数矛盾（`v1_5_judge_qualification.py` /
   `v1_5_esconv_auxiliary_pairwise.py`），统一到不合成分数的设计——工作量小，
   但是"透明指标"的前提条件，必须先做。
2. 完成 Strategy Bank `Affirmation and Reassurance` family 的完整清洗+人工审核
   （第 8.2 节），产出至少 10-20 张 `eligible_for_formal_rs=true` 的卡片；用
   problem_type/emotion_type 做覆盖度审计（第 8.3 节）。
3. 用现有 40 组人工锚点，重新定位 SupportNeedObservation 到"第一层规则 + 第二层
   仅两个已验证轴的固定 zero-shot"（第 2-3 节），把其余轴降级为诊断/研究用，
   不进 PM 特征。
4. 构造 15-20 组最小对照对（第 4.2 节），验证第一层规则精确率/召回率，校准第二层
   两个轴的可靠性区间。
5. 用第 2 步产出的 Bank 卡片，跑 RS 的 clean matched-pair pilot（~40 train
   groups），检查 treatment uptake。
6. uptake 成立后，拟合 component-effect head（唯一允许训练参数的地方），
   grouped-OOF 对比 prior/rule。
7. 按第 6 节的逐轴透明登记表，输出最终报告——quality/risk/cost 三线并列 +
   需求诊断逐轴报告，不产出任何综合 PASS。
8. 补一份 EvoEmo 数据理解分析（第 9 节），再决定 EvoEmo 外部评测的实际范围。

### 明确不做/留给 V2.0

- 5 类平铺 support_mode 分类（已经三次证明学不出来，不再投入）；
- 继续 embedding 模型选型竞赛（Qwen3 之后不再引入新的大模型候选）；
- `need_for_exploration`/`focused_question_readiness`/`planning_readiness`/
  `low_interaction_burden` 的正式监督学习（样本量不支持，留给 V2.0 有更多数据
  之后再做）；
- Strategy Bank 除 `Affirmation and Reassurance`（以及资源允许时的
  `Providing Suggestions`）之外其余 family 的完整清洗。

---

## 11. 一句话总结每个问题怎么解决

- **诊断模块靠谱吗**：不再靠"训练"，靠"规则 + 已验证信度的固定 zero-shot"，
  且只在两个真正有统计信号的轴上使用，其余诚实标记为 unsupported。
- **训练数据可靠吗**：停止依赖 ESConv 原始对话（互信息已证明接近零），转向
  人工/LLM 构造的最小对照对，样本效率更高。
- **Judge 怎么评、分歧怎么办**：用实测出的可靠性数字分配角色（偏好判断信 LLM，
  证据/风险判断信代码），双裁判分歧走已有的 soft-label 规则，硬性门槛
  （AB/BA<0.80 或人工锚点一致率<70% 不得批量标注）严格执行。
- **指标透明吗**：逐轴报告，禁止任何合成分数，清理已发现的一处代码矛盾。
- **BAAI 配合对不对**：角色收窄到它真正擅长的检索匹配，不再追更大模型；接口上
  绝不能把原始相似度或全库 centroid 直接暴露给 PM 当决策捷径。
- **Strategy Bank 是不是都是情绪支持策略**：不是，8 类里只有 2 类是纯情绪支持，
  项目已经正确识别并排除了不合格 family，缺口在"清洗完只剩 5 张卡"，第 8.2 节
  给出了具体补法。
- **ESConv/EvoEmo 吃透了吗**：ESConv 吃透了（含负面但真实的结论）；EvoEmo 还没有，
  第 9 节给出补法。
- **小样本弱监督怎么学出来**：不在小样本上拟合参数，只在小样本上验证固定方法的
  可靠性；唯一允许拟合参数的地方是 component-effect head，且有 matched-pair
  设计和 ESS 门槛支持。
