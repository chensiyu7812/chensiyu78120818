# PM V1.5 反向设计执行方案（历史）

状态：`SUPERSEDED FOR FUTURE EXECUTION / 2026-08-02`

> **规范入口已迁移：** 后续唯一执行方案为
> `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`，机器合同为
> `data/pm_v1_5_contracts/final_execution_plan_v3.json`。本文保留反向设计、H1/H2 与早期
> 256-state 思路的历史证据；其中与候选先行、四组件 ontology、Step 2 或停止规则冲突的
> 内容不得继续执行。

机器合同：
`data/pm_v1_5_contracts/reverse_designed_training_and_evaluation_v1.json`

> **优先修订：** 本文原第 2 节把“资源适用人评”直接定义为 PM gold，与实际研究目标
> “预测固定资源的边际回复收益”不一致。RS 的当前优先合同已改为
> `data/pm_v1_5_contracts/rs_open_close_construct_v2.json`：资源适用只负责 scope、
> coverage 与特征设计；RS-on hard label 必须来自 qualified same-state paired material
> benefit，再通过 atomic risk 与 cost。本文其余关于四 heads、16 actions、同栈、数据
> 隔离和 grouped evaluation 的约束继续有效。MP/MS/ME 的正式 treatment 尚未执行，
> 不由本次 RS 结果代写。

当前进度：256-state allocation 已冻结并通过自动审计：64 users、每人 4 states；
train/calibration/internal=`160/48/48`；每个 split 的 16 动作完全均衡，四组件分别
50% on/off，三个 split 用户零重叠。自然对话与资源内容尚未生成，H1/H2 尚未开始。
证据见 `outputs/pm_v1_5_reverse_designed_blueprint_v1/allocation_audit.json`。

H1 总包现已准备完成，等待一次性人评：50 个 core judgments 覆盖 100 个 variants，
并含 64 个 train-only 自然检索状态和 16 个确定性 abstention controls。查询与卡片
来源例、既往 RS 包和 formal ESConv test 零重叠；检索选择不读取下一回复、native
strategy 或生成 outcome。入口为
`outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/human_review.html`。

H1 人评现已完成。标签 130/130 精确对齐：11 core/22 variants 可直接冻结，36 core/
72 variants 仅需重连来源，3 core/6 variants 因重复删除；50/50 core 的定义、
when-to-use 和 boundary 均通过。检索侧 RS opportunity=57/64，但 Top-1 fit=33/64、
hard exclusion=8、真实 opportunity 的 Top-5 recall=50/57。当前 Bank/ranker 未资格化，
但 opportunity 构念成立；失败拆为 17 个排序错、7 个 coverage gap 和 7 个本应
pre-retrieval abstain。H1 只允许一次修复，fresh 资格由 sealed H2 internal RS slice
承担，不再新增 standalone retrieval 人评包。正式报告见
`outputs/pm_v1_5_h1_complete_rag_review_v1_analysis/report.html`。

## 1. 先冻结研究主张

V1.5 要验证的是：

> 在固定 generator、固定资源库、固定检索与过滤流程下，一个低容量、可审计的 PM，
> 能否根据回复前可见信息，学习选择 MP、MS、ME、RS 四类资源的开关，在保持或改善
> 支持回复质量、不过度增加预先定义的风险时，减少不必要的资源与 token 成本。

不主张：

- PM 能诊断全部用户需求；
- PM 能估计个体因果处理效应；
- Strategy Bank 覆盖医疗、临床、高危或专门人群知识；
- 新增任意资源后无需重训；
- 四个组件都必须学得优秀。

四个 bit 组合成 16 个合法动作。V1.5 不改成只做 RS/ME，也不改成直接 16 分类。

## 2. 三种证据必须分开

此前反复失败的核心，是把三件事混成一种标签。

1. **资源适用标签**：回复前，这个 state 是否在组件 scope 内、是否存在可实现的候选。
   它来自透明规则和人评，用于覆盖、过滤与特征设计，但不单独构成 RS-on gold。
2. **资源效应证据**：在合格 state 上控制开/关，固定资源是否带来实质回复质量收益。
   对 RS 而言，qualified matched response pair 是 benefit head target；它不是用单次
   随机胜负估计个体真实因果效应，而是在冻结生成协议下定义可审计的训练代理。
3. **端到端结果**：冻结 PM 后，其选择是否带来更好的 quality/risk/cost 平衡。

RS 的 48 对结果只否定“六特征预测一次 pair winner”这个模型，不否定 RS 的资源能力
或正式路由任务。旧 MP/MS/ME 468-state 结果只暴露模板捷径；正式 clean treatment
尚未运行，不能写成通过或失败。

## 3. 平时学习与考卷

### 3.1 同一套东西

训练、内部测试、ESConv 和 EvoEmo/ES-MemEval 外部测试必须使用同一：

- Strategy Bank 版本；
- MP/MS/ME catalog schema；
- query builder、retriever、filter、Top-k 和 prompt compiler；
- generator 模型、解码参数与 system prompt；
- PM 特征定义；
- quality/risk/cost 评测规则。

外部集只允许 mask 不存在的资源。例如 ESConv 没有可靠长期记忆时，MP/MS/ME
availability 为 false；不能为 ESConv 临时换一套 RAG。

### 3.2 覆盖关系

生成训练集必须同时含：

- ESConv-like：单 session、情绪支持、RS 可用或不必要、寒暄/结束/只想倾听等负例；
- longitudinal-like：稳定偏好、前次摘要、近期事件、过时/冲突/重复/错人记忆；
- 组件交互：全部 16 个动作都有训练、校准和内部测试样本；
- 外部考试的能力边界内扩展，而不是复制外部测试文本或查看其 outcome。

ESConv 正式 test、EvoEmo/ES-MemEval outcome、外部 judge 分数不得参与 Bank 建立、训练
生成、阈值选择或模型选择。

## 4. RAG 应该怎样建立

六张卡只保留为已验证的最小 pilot，不作为最终覆盖充分的 Bank。

最终 Bank 从 ESConv **train split** 的支持动作开放编码和现有 100-card 候选出发，
按覆盖矩阵去重、修订，而不是先拍脑袋规定五类。目标范围为 60–100 张；最终数量由
去重和人审决定，不为凑 100 保留近重复卡。

每张卡必须包含：

- atomic move；
- when_to_use；
- when_not_to_use；
- 可见 cue；
- dialogue phase / goal；
- question/directive burden；
- risk flags；
- topic-agnostic technique；
- train-only source provenance。

Bank 不包含寒暄、bye、平台事务话术、医疗/法律/临床 factual advice。这些是 RS-off
负例或确定性安全流程，不是策略知识。

覆盖矩阵至少包括：邀请继续、聚焦澄清、tentative paraphrase、grounded emotion
reflection、validation、现实而非虚假安慰、经许可的单一步骤建议、选择支持、
努力确认、总结/过渡/结束；每类只保留真正不同的条件变体。

只再做人力上的一个 **RAG 总包**：

1. 审完每张最终卡的定义、来源支持、排除和冗余；
2. 对 train-only 查询审 Top-5，标记所有可接受项、最佳项和 hard exclusion；
3. 冻结 Bank、规则、检索器和 Top-k。

检索资格门：

- 最终卡 100% 完成人审，未批准卡不进入 Bank；
- Top-1 suitable 至少 0.80；
- Top-3 acceptable recall 至少 0.90；
- hard-exclusion 进入注入集合为 0；
- lexical 与 BGE-small 在同一人评标签上比较。只有 BGE 在 grouped holdout 有明确增益
  才使用；BAAI 不是前置条件。

检索先取 Top-3 供过滤和审计，注入默认最多 1 张；Top-2/3 只有在消融实验显示稳定
改善且不增加风险/成本时才可改变。

## 5. 生成训练数据：按答案蓝图生成，而不是生成后猜答案

冻结一个 256-state 的受控数据集：

- 16 个动作各 16 个独立 state；
- train 每动作 10 个，共 160；
- calibration 每动作 3 个，共 48；
- internal test 每动作 3 个，共 48；
- 64 个独立 synthetic users，每人 4 个 states，按 user 分割；
- 每个组件在 train 中有 80 on / 80 off；
- 当前 user 文本和历史不能含 action 名、组件名或固定模板暗号。

每个 state 先由蓝图定义四个 bit 和原因，再生成自然对话与资源。蓝图规则：

| 组件 | on 的必要条件 | 典型 off |
|---|---|---|
| RS | 有合格且非重复的技术卡，能改变回复动作/约束 | 寒暄、结束、无合格卡、重复、边界冲突 |
| MP | 稳定 profile 与当前回复相关且非敏感、非重复 | 无关、错人、不稳定偏好、当前已明确说出 |
| MS | 前次 session 摘要对当前上下文连续性必要 | 单 session、摘要无关/过时/已在当前窗口 |
| ME | 一个具体、较新事件与当前问题直接相关 | 过时、冲突、错人、同义重复、无关事件 |

正负例必须在 topic、文本长度、resource count、age、token、相似度上匹配。任何单一
metadata 或模板词在 grouped development 上达到 BA 0.70，数据先修复，禁止训练。

PM 在 Step 1 只能看到部署时可见的信息：当前对话、显式边界、各资源 availability、
count/age/token 预算，以及固定检索器产生的**不含候选正文和 ID**的 source-level
opportunity 摘要。不能看到 gold bit、generator 回复、judge 分数或未来 outcome。

这里明确把 V1.5 的“基础资源机会判断”作为受控、透明的已解决前提。它不是需求诊断
论文，也不要求 PM 从完全不可辨识的目录摘要猜出隐藏 item。

## 6. 人评到哪里结束

从现在起不再发零散小包。只允许三次完整且有终点的人评：

1. **H1 RAG 总包**：最终卡 + retrieval Top-5；完成即冻结。
2. **H2 训练数据总包**：一名主审审核全部 256 个 state/action；第二名只审预注册的
   64 个分层样本和全部 flag。分歧只裁决 flag 与第二审样本，不重做整批。
3. **H3 最终回复盲评**：PM 完全冻结后，一次评 learned / always-off /
   high-resource-fixed 的端到端回复。

过去两天的人评作为 formative evidence 和 codebook 来源继续使用，不冒充正式 test，
也不重复标注。

## 7. 稳定训练方法

主模型为四个独立的 L2 logistic heads；可共享同一特征表，但不联合微调文本 encoder。
每个 head 使用 group-weighted binary cross entropy：

\[
L_c=-\sum_g w_g[y_{gc}\log p_{gc}+(1-y_{gc})\log(1-p_{gc})]
    +\lambda\|\beta_c\|_2^2
\]

选择 logistic 的原因是样本少、可解释、概率可校准、不会用树深度吸收模板捷径。
HGB/浅树只作 challenger，不作为默认训练器。BGE-small 只可作为冻结 embedding
challenger；若透明特征已经过门，V1.5 不为追求小增益增加复杂度。

训练协议：

- 按 user grouped 5-fold OOF；
- C 与阈值只用 train OOF 选择；
- calibration split 只做温度/Platt 校准和最终阈值冻结；
- internal test 只运行一次；
- 固定 5 个预注册 seeds，报告均值、标准差和最差 seed；
- 每个 head 必须产生 on 和 off，任一比例低于 0.20 视为退化；
- 比较 always-off、always-on、透明规则、logistic、浅树；
- 四个概率经过 risk hard guards 和 cost budget 后组合成 16 动作。

训练晋级门不是 100%：

- 每 head internal balanced accuracy 至少 0.70；
- 四 head macro balanced accuracy 至少 0.80；
- positive recall 每 head至少 0.60；
- Brier 优于 prevalence prior；
- 5-seed macro BA 标准差不高于 0.03；
- exact 16-action accuracy 明显优于 always-off 与透明规则；
- 无 user/group 泄漏、无单变量捷径、无 test 调参。

达不到的 head 诚实报告未学会，但不能在实验前因一次失败把组件永久删掉。先检查数据
覆盖、标签构念和捷径；只允许按预注册修复一次，再冻结结果。

## 8. 透明评测

### 8.1 PM 是否学会

逐 head 报告 BA、precision、recall、F1、Brier、ECE、on/off rate；联合报告 Hamming
loss、exact 16-action accuracy、每动作混淆和 5-seed 稳定性。

### 8.2 回复质量

盲评顺序固定为：

1. 是否尊重用户明确请求/拒绝和结束边界；
2. 情绪与语境是否贴合；
3. 是否提供实质帮助；
4. 互动负担是否合适；
5. 是否清楚自然。

只有会改变使用判断的差异才记 A/B；轻微措辞偏好记 tie；无效生成记 unknown，不转成
off 标签。

### 8.3 Risk

V1.5 的 risk 不是一个含糊总分。逐事件报告：

- `boundary_or_permission_violation`；
- `unsupported_or_overstated_claim`；
- `stale_conflicting_wrong-person_memory_use`；
- `excessive_directiveness_or_burden`。

每项有 yes/no/not-applicable、literal evidence 和明确分母。高危临床安全由固定 guard
处理并披露为范围外，不用稀少事件训练一个假装可靠的 risk head。

### 8.4 Cost 与总决策

报告检索次数、注入 item 数、input/output/total tokens 和调用费用。决策按词典序：

1. hard risk 失败则关；
2. 没有合格机会则关；
3. 预测质量收益未过阈值则关；
4. 超预算则关；
5. 多个动作都合格时选择成本最低者。

不能用 cost 抵消明显质量下降，也不能用模糊“可能更安全”掩盖质量无收益。

## 9. 外部实验

- ESConv：主要检验 RS、无长期资源时的正确关闭、即时质量与 burden；
- EvoEmo/ES-MemEval：检验 longitudinal MP/MS/ME、stale/conflict 和跨分布稳健性；
- 内部 48-state test：保证 16 动作及交互都有已知答案的完整考试。

外部只运行冻结 policy，不重新训练、不换 RAG、不调 threshold。若某外部域缺少某组件
机会，报告 opportunity-conditioned 指标，不能把“数据集没有记忆”写成 PM 失败。

## 10. 立即执行顺序

1. 冻结研究合同和 256-state blueprint；
2. 从 ESConv train 与现有 100-card 候选整理 60–100 张最终 Bank；
3. 一次完成 H1，冻结 RAG；
4. 生成并自动审计 256 states，未过 shortcut/coverage gate 就在 H2 前修；
5. 一次完成 H2；
6. 训练四个 heads，跑 OOF、calibration、5-seed stability；
7. internal test 一次；
8. PM 冻结后生成外部与端到端对照，完成 H3；
9. 写论文，所有失败组件和范围限制照实报告。

现在可以开始训练准备。真正的阻塞条件只有三个：最终 RAG 尚未 H1 冻结、256-state
数据尚未通过自动 anti-shortcut 审计、H2 标签尚未冻结。它们是一个整包流程，不再以
新的小规模 RS 人评无限前置。

## 11. 2026-07-30 H1 后实际执行状态

H1 已完成，但结论不是“100 张卡全部失败”，而是三层结果必须分开：

- 定义层：50/50 core 的 atomic move、when-to-use、边界均通过；
- 来源层：11 个 core 直接通过，36 个非重复 core 只需重新绑定字面来源例，3 个
  duplicate/reject core 删除；
- 检索层：普通 RS opportunity 构念成立，但旧 Top-1 与 hard-off 未过资格门。

因此当前 Bank 状态固定为：

- 22 个 variants（11 cores × minimal/dialogic）已冻结；
- 72 个 variants（36 cores）只等待一次来源重绑；
- 6 个 variants（3 cores）已拒绝，不补位、不再复审定义。

36-core 来源重绑页面位于
`outputs/pm_v1_5_h1_source_relink_review_v1_candidate/human_review.html`。候选来自
ESConv train，排除了旧 H1 来源、H1 查询和最终 H2 保留边界；每个 core 五个候选，
勾出至少两个直接体现 atomic move 的不同对话即通过。旧 BGE、词法和 DeBERTa NLI
只用于三路找候选；抽查发现 NLI 会把泛化安慰误判成“具体进展”，因此已禁止它单独
决定候选或标签。另用 36 次 GPT-4o 调用做候选搜索排序，分数和结果对审核者隐藏，
最终来源门仍只认盲评勾选。这是 H1 唯一一次有界纠正，之后不再发新的 card 小包。

检索修复使用单独 post-H1 overlay，保持原 H1 可复现。修复只加入：

- 最新可见轮必须来自 seeker；
- 暴力/自伤、药物加儿童安全、工作场所违法行为转独立安全流程；
- 事实/外部资源请求退出 topic-agnostic technique Bank；
- 寒暄、明确停止和常规收束关闭 RS；
- card-specific 可见 cue 先过滤，语义相似度只能在安全、适用层级内排序。

在 H1 开发标签上，旧开关为 73/80，修复后为 80/80；这是用 H1 发现问题后回放 H1，
只能证明代码按纠正方向工作，不能作为论文成绩。可评 Top-1 为：

- 透明 tier + lexical：34/50 = 0.68；
- 同一安全 tier 内 BGE-small：37/50 = 0.74；
- 配对差 +0.06，bootstrap 95% CI `[-0.06, 0.18]`。

因此 BAAI 保留为 H2 challenger，但没有足够证据成为前置依赖；当前默认仍是透明
tier + lexical。报告位于
`outputs/pm_v1_5_post_h1_retrieval_bakeoff_v1/report.html`。

为防止继续看答案改题，48 个 H2 状态已在最终 Bank 和 H2 排名结果产生前冻结：
advice、effort/progress、emotion、uncertainty、ordinary hard-off、
phatic/stop/closing 各 8 个，48 个独立 ESConv-train dialogue，与 H1 查询及本轮
来源 finalists 零重叠。唯一事实源是
`outputs/pm_v1_5_h2_retrieval_state_freeze_v1/manifest.json`。

接下来只有一条线性路径：

1. 完成 36-core 来源重绑页；
2. 自动冻结通过的最终 Bank；
3. 在已冻结的 48 个 H2 states 上同时生成透明/BGE 的盲化候选并一次审核；
4. 按 H2 选择检索器，随后进入 256-state 生成、自动 anti-shortcut 审计和 H2 数据总包；
5. 四个 heads 一起训练，不再单独扩大 RS 小包。

### 11.1 来源重绑完成与最终 Bank 冻结

36/36 cores、180/180 候选已完成审核：96 个候选接受，29 cores 达到至少两个独立
ESConv-train 对话的来源门，7 cores 未过门且不再补救。连同原 H1 的 11 个通过 core，
最终冻结为 40 cores / 80 minimal-dialogic cards：

- Question 10 cores / 20 cards；
- Providing Suggestions 9 / 18；
- Affirmation and Reassurance 8 / 16；
- Restatement or Paraphrasing 7 / 14；
- Reflection of feelings 6 / 12。

最终 Bank 唯一事实源：
`outputs/pm_v1_5_strategy_bank_v4_final_v1/manifest.json`。它含 127 条来源血缘，但
runtime cards 不含任何原始来源回复。H2 通过前 80 张卡全部保持
`eligible_for_formal_rs=false`；Bank 内容已冻结，不得根据 H2 结果换卡或补卡。

### 11.2 H2 检索资格页

H2 页面：
`outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate/human_retrieval_review.html`。
它消费预先冻结的 48 个 states，透明 tier+lexical 与 BGE-small 在相同 hard-off、
card applicability 和 80-card Bank 下各取 Top-3，再形成最多五张的盲化并集。

结果前冻结的门：

- opportunity uncertain rate `<=0.10`；
- opportunity balanced accuracy `>=0.90`；
- selected ranker Top-1 acceptable `>=0.80`；
- Top-3 acceptable recall `>=0.90`；
- method Top-3 hard exclusion 为 0。

只有 BGE 的 H2 Top-1 比透明检索至少高 0.05、Top-3 不下降且不增加 hard exclusion
时才采用 BGE；否则保留透明检索。分析器已经在查看 H2 标签前实现并测试：
`scripts/v1_5/24aq_analyze_h2_retrieval_qualification_v1_5.py`。H2 未过门时不得在
同一批状态上调规则，RS 保持 formal disabled，并作为 V1.5 限制报告。

### 11.3 H2 一次性资格结果

48/48 条独立审核已完整绑定，`uncertain=0`，Bank 内容未在 H2 后改变。预注册分析
结果为 `H2_NOT_QUALIFIED_FORMAL_RS_REMAINS_DISABLED`：

- opportunity：TP=28、TN=14、FP=3、FN=3，balanced accuracy=`0.8634`，
  未达到 `0.90`；
- 透明 tier+lexical：Top-1 `19/31=0.6129`，Top-3 `25/31=0.8065`，
  Top-3 中 10 个 state 含 hard exclusion；
- BGE-small：Top-1 `15/31=0.4839`，Top-3 `24/31=0.7742`，
  Top-3 中同样有 10 个 state 含 hard exclusion；
- BGE 相对透明 Top-1 为 `-0.1290`，因此不采用 BGE，保留透明方法作为后续
  development baseline，但它尚不是正式合格 RS。

失败不是“所有场景都没有信号”。透明 Top-1 在 `advice_welcome` 和 `emotion`
均为 `7/8`；主要失败集中在 `uncertainty=2/7`、`effort_or_progress=3/6`，
以及原本作为 hard-off 抽取的 8 个状态中有 2 个被人工判为真实 RS opportunity。
10 个 hard-exclusion state 进一步定位出三类执行缺口：

1. 已有明确请求时，探索问题仍进入候选，造成目标替换或额外问题负担；
2. `trusted_support`、`lingering_feeling` 等卡缺少“当前上下文必须有直接证据”
   的可执行过滤；
3. 法律/安全、任务收尾和需要事实纠正的请求仍可能被普通支持卡错误开启。

因此本 H2 的科学结论是：最终 Bank 中存在可用子区域，但当前 opportunity gate 与
card applicability/filter 不能可靠守住使用边界；BAAI 不是补救方案，并在同一候选
空间内显著劣于透明排序。不得用这 48 条继续调参后再把同批结果称为资格通过。
唯一事实源为
`outputs/pm_v1_5_h2_retrieval_qualification_v1/summary.json`，冻结标注、逐条绑定
诊断和可读报告位于同目录。

后续若继续争取正式 RS，只允许把 H2 当 development diagnosis：把上述三类缺口
改写为一般、可部署、与 card 文本一致的确定性约束，冻结实现后在全新的独立状态上
做一次 fresh qualification。不能改 80-card Bank 内容、不能重新选择这 48 条，
也不能降低既有门槛来宣布通过。其他三个 memory heads 与 16 动作研究不因本次
RS 失败被写成已做或已失败。
