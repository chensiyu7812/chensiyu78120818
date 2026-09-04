# PM V1.5 完整方案审计与最短可发表路线

状态：`已撤销的判断快照 / 仅作错误审计 / 2026-07-30`

> **撤销说明：** 本文把一个六特征、单次随机回复胜负标签的 RS 分类器失败，错误扩大
> 成了 RS 资源和 RS 研究问题失败；又把旧 synthetic proxy 诊断误当成 MP/MS/ME 的
> clean-treatment 结果，并据此擅自改成 retrieve-then-admit。三者均不成立。本文不得
> 作为执行入口；有效方案见
> `docs/PM_V1_5_REVERSE_DESIGNED_EXECUTION_PLAN_20260730_ZH.md`。

本文是 2026-07-30 的研究判断快照。历史细节仍以
`PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md` 为唯一问题账本；旧运行计划若与本文的
最新实证判断冲突，不得继续据其扩量。

## 1. 结论

当前 V1.5 已经修正了旧 V1.0 和旧 V1.5 最致命的实验设计错误，但还没有从实证上证明
一个可发表的 learned PM。

更准确地说：

- treatment 同一性、分组切分、component factorization、默认关闭、quality/risk/cost
  分离、同一 RAG 和失败停止规则，已经基本正确；
- RS 的真实补充证据已足够做出停止决定：固定六定义卡 treatment 总体近中性、成本明显
  增加，且状态级收益不可预测；不再补 RS 人评、不扩卡、不用 BAAI 救它；
- MP/MS/ME 的受控训练管线可以产生 on/off，但 MP/MS 的内部高分主要可由合成模板长度和
  聚合统计恢复，不能称为真实记忆路由已学会；ME 未通过最低门；
- 根本剩余问题不是“还没找到更大的模型”，而是旧决策点放在检索前，PM 看不到候选
  evidence，却被要求判断目录中是否有真正有用、不过时、不冲突的 item。这在信息上不可
  辨识；
- 最快且科学的修正，是把本地检索当作固定、低成本的候选生成步骤，把 PM 的开关重新
  定义为“候选 evidence 是否进入 generator”，而不是“是否允许检索器运行”。

因此，本项目现在不是回到原点。已经知道哪些路线应停止，也第一次定位到 PM 任务定义中
最后一个结构性错误。下一轮不需要重新做需求诊断、扩大 Strategy Bank 或再次评 RS。

## 2. 最新补充人评的结论

新提交的 16 对补充盲评结构完整：

- 16/16 pair 与 private key 对齐，blind ID 唯一；
- quality：RS win=2、R0 win=7、material tie=7；
- RS NetWin=`-0.3125`；
- material-risk response：R0=4、RS=4；
- 没有任何原子 risk 类达到 8 event + 8 non-event 的训练资格；
- RS prompt tokens 比 R0 增加约 `77.8%`；
- RS total tokens 比 R0 增加约 `54.9%`。

一条 risk 引文使用了按原文顺序排列、以省略号连接的多个 literal fragments，而不是单个
连续 substring。这是可追溯格式偏差，不改变对应 quality 标签。

与第一批 32 对合并后的描述性结果为：

| 数据 | RS win | R0 win | tie | RS NetWin |
|---|---:|---:|---:|---:|
| 第一批 | 14 | 10 | 8 | +0.1250 |
| 新补充 | 2 | 7 | 7 | -0.3125 |
| 合计 | 16 | 17 | 15 | -0.0208 |

这表明第一批的 RS 方向没有独立复现。合计结果不支持“RS 普遍更好”，也不支持继续寻找
一个 state-level RS winner classifier。

## 3. RS 是否学会

在补充 outcome 之前冻结的模型只使用六个透明特征：当前用户文本长度、历史轮数和
AM01/AM04/AM05/AM10 move one-hot。没有使用 BAAI、blind identity、回复或评测结果。

### 3.1 32 训练组 → 16 前瞻确认组

- model log-loss=`0.6815`，训练 prevalence prior=`0.6068`；
- model Brier=`0.2442`，prior=`0.2070`；
- 0.60 阈值下 0 on / 16 off；
- raw accuracy=`87.5%`，但 always-off 也是 `87.5%`；
- balanced accuracy=`0.50`，positive recall=`0`。

所以“87.5% 选对”不是成功。它只来自确认集中 14/16 为 nonbenefit 的类别不平衡。

### 3.2 48 组 nested OOF

- model log-loss=`0.7067`，fold prior=`0.6365`；
- model Brier=`0.2564`，fold prior=`0.2222`；
- 0.60 阈值下 0 on / 48 off；
- Brier gain bootstrap 95% 区间约 `[-0.0708, 0.0039]`。

正式结论是 `RS_INDIVIDUALIZED_ROUTING_NOT_LEARNED_STOP_EXPANSION`。这不是要求 PM
达到 100% 才算学会，而是模型连 constant prior 都没有胜过、也没有产生任何正决策。

## 4. 受控 MP/MS/ME 训练告诉了什么

冻结的六特征 L2 logistic regression 使用 36 个 development users，随后一次评估
16 个原 internal users：

| Head | Accuracy | Balanced accuracy | Positive recall | on/off | 原冻结门 |
|---|---:|---:|---:|---:|---|
| MP | 0.868 | 0.703 | 0.406 | 13 / 131 | 表面通过 |
| MS | 0.924 | 0.895 | 0.844 | 33 / 111 | 表面通过 |
| ME | 0.833 | 0.625 | 0.250 | 8 / 136 | 未通过 |

这证明训练器、group split、proper score 和 on/off 编译能够工作；它反驳了“任何 PM 都
绝对学不出来”。但随后 post-hoc 数据有效性审计发现：

- 仅用 MP `estimated_tokens <= 67`，内部 balanced accuracy 已约 `0.857`；
- 仅用 MS `estimated_tokens >= 85.5`，约 `0.842`；
- 仅用 MS 聚合相似度阈值，约 `0.826`；
- “需要该 source”的正类平均相似度反而更低。

这些规律跨 user 保持，是因为 helpful 与 semantic-decoy 的生成模板在长度和聚合表示上
留下了稳定差异。它不是直接读取 `needed_memory_sources`，但仍是构造捷径，而不是一个
可部署的语义判断。因此 MP/MS 的结果只能称为 pipeline learnability diagnostic，不能
晋级为论文中的 learned routing 证据；此前 internal split 也已被本轮消费，今后只能视为
development。

## 5. 旧 V1.0 和旧 V1.5 的问题是否解决

| 历史问题 | 当前处理 | 状态 |
|---|---|---|
| 训练、fixed、external 使用不同 RAG/prompt/filter | same-stack contract；同一组件只能复用同一 catalog、retriever、prompt、generator | 方法上已解决 |
| interactive policy 产生不同 seeker 世界线 | fixed visible state / matched response | 已解决 |
| AB/BA 顺序效应和含混 composite judge | blind pairwise quality；risk 原子化；tie/uncertain/N/A 分开 | 大幅解决，仍是 proxy |
| 16 action 直接 argmax、winner's curse | 四个 component heads，合法 bitset 再编译成 16 action | 已解决 |
| action rows/turns/judges 虚增 N | user/dialogue group 是独立统计单位 | 已解决 |
| 高维 HGB、小样本 collapse | 低容量、L2 logistic、grouped CV、proper-score baseline | 已解决 |
| cost 权重事实上不起作用 | cost 不进含混加权分；质量和 risk 合格后作硬预算/平局裁决 | 已解决 |
| risk 定义含混 | 统一为 interaction-and-grounding risk proxy，明确分母、N/A、severity 与引用 | 基本解决 |
| BAAI 被当作需求或效用 oracle | BAAI 只可作候选检索/表示；任务内无增益不采用 | 已解决 |
| raw 11,590-card/Top-3 噪声与训练测试两套 RAG | ESConv-train-only 建 Bank；同一 RAG；六卡最小 treatment 已测 | 同一性解决，覆盖/收益未解决 |
| 100-card 自然 Top-1 检索错误 | 已用盲法暴露失败并回退，不再冒充可用 | 失败已正确处理 |
| RS 一次随机胜负被当个体效应 oracle | 独立补充确认和 proper-score 检验后停止 RS | 已解决 |
| 合成 source label 可由模板/metadata 猜中 | 本轮新发现；必须重做数据平衡和决策边界 | 尚未解决 |
| pre-retrieval PM 看不到 item，却要判断 item utility | 改为 retrieve-then-admit 是下一版核心修正 | 尚未实施 |
| EvoEmo 204/204 metadata OOD、ESConv 无长期 memory positive | 只能降级为 secondary transport/abstention test，另需受控 memory-positive holdout | 尚未解决 |
| clinical/真实用户获益被过度表述 | 明确只主张 synthetic-state、即时回复和 interaction/grounding proxy | 已解决 |

总判断：当前方案已经避免重复旧实验的主要因果错误，但尚未从根本上解决“真实候选 evidence
是否适合进入回复”的可学习性。旧 V1.0 数字不能靠降低及格线复活；旧 V1.5 的 all-off
模型也不能因 raw accuracy 高而复活。可以继承的是数据、失败证据、同栈实现和评测工具，
不是旧 checkpoint 或结论。

## 6. 最终 V1.5 的 PM 应怎样定义

### 6.1 决策边界

固定执行顺序改为：

```text
可见对话
  → 同一套本地 retriever 为 MP/MS/ME/RS 各取 Top-k 候选
  → 固定 hard exclusions（未来信息、明确冲突、高风险领域、用户边界）
  → PM 看候选及其可部署 metadata，判断是否注入
  → 最多每 source 注入 1 条；固定 generator 生成
```

“资源开”在论文中明确定义为“候选 evidence 进入 generator prompt”，不是“是否执行廉价
本地向量检索”。这样 PM 才能看到完成判断所必需的信息。retrieval 计算仍计入成本，但主要
成本差异来自 prompt evidence 和 generator tokens。

### 6.2 输入、输出和动作

每个 component head 只看：

- 当前可见对话；
- 该 source 的 Top-1 候选文本；
- rank-1 score、rank-1/rank-2 margin；
- source、age、token 数；
- 透明 conflict/staleness/boundary flags。

不得看：

- regime、`needed_memory_sources`、item utility 标签；
- gold response、future turn、judge outcome；
- external split identity 或目标 action。

输出仍为 MP/MS/ME/RS 四个 bit，理论合法空间仍为 16 action。V1.5 不强迫四个 head
全部通过；unsupported head 按预注册规则关闭。因此保留 16 个合法 action 不等于伪造
16 种已学会的行为。

RS 在当前版本保持合法但 learned status=`NOT_LEARNED`，默认 R0。六卡 Bank 保留用于
fixed ablation 和失败分析；不再扩卡。

### 6.3 标签

训练目标改为两层，不能混合：

1. `candidate_admissible`：候选是否与当前状态相关、非重复、未过时、无冲突，属于
   pre-response evidence label；
2. end-to-end matched outcome：注入该候选后，quality/risk/cost 是否达到最终策略门；
   只用于资格化 treatment 和评估最终 policy，不再把一次随机回复胜负当逐 state oracle。

第一层使“判断候选能否使用”可学习；第二层防止“候选相关”被偷换成“回复一定更好”。

## 7. quality、risk、cost 的最终统一定义

### Quality

主结果只用 blind immediate-support preference：

- policy materially better；
- comparator materially better；
- materially equivalent；
- uncertain/abstain。

判断顺序固定为：明确请求/边界匹配 → grounded emotional fit → 本回合直接帮助 →
互动负担 → 清晰自然。轻微文风偏好必须是 tie。

### Risk

论文主表固定四项，不再随 packet 新增训练 head：

1. explicit boundary violation；
2. unsupported personal claim；
3. stale or conflicting memory use；
4. excessive directiveness / interaction burden。

`false reassurance/minimization` 和 `domain/high-stakes overreach` 保留为 severity-3
事件清单或映射到第 2/4 项的诊断子类，不另造一套不兼容的总体 risk。每项单独给适用
分母；memory-off 对第 3 项是 N/A，不是零风险。V1.5 不主张 clinical safety。

### Cost

分别报告：

- 本地 retrieval calls；
- 实际注入 item 数与 tokens；
- generator input/output/total tokens；
- 若有稳定计价，再报告 provider USD。

不构造 `quality - λ risk - μ cost` 总分。顺序是 quality 非劣 → 每项 risk 不明显恶化
→ 在合格策略中选择更低成本。

## 8. 下一步的最小执行路线

### P0：立即冻结结论

- RS 不再补标、不扩卡、不换 embedding、不调阈值；
- 当前 468 states 和已打开的 16 internal users 全部降为 development；
- MP/MS 表面 PASS 标为 construction-shortcut-limited，不晋级；
- ME 标为 not learned。

### P1：重建可辨识的 candidate-admission 数据

复用现有 468 corpus 的内容和 evaluator annotations，但训练 grain 改为
`state × retrieved candidate`。先做三项零 API 修复：

1. helpful/irrelevant/harmful item 的 count、age、token 分布匹配；
2. 禁止任一 metadata 单阈值在 grouped development OOF 达到 balanced accuracy `0.70`
   以上；
3. 用固定 lexical retriever 先召回，再让模型判断候选；不再用 source centroid 代替 item。

现有 `PMV2EvidenceFilterModel` 可以复用为工程起点，但必须降容量、按 user group CV，并
与透明 lexical rule、always-off 比较。BAAI 只有在 item-level grouped retrieval/admission
上真实优于 lexical 才采用；它不是前置条件。

本轮零 API 盘点已经完成：原始 2,808 个 candidate 不能直接训练，因为 compiler-owned
negative phrase 的 absence rule 在三个 split 都达到 balanced accuracy `0.71875`。把这些
显式 synthetic negatives 固定为 hard exclusion 后，仍剩 1,716 个 marker-neutral
development rows（312 positive、1,404 negative、52 users）；item token/age 单阈值最高
约 `0.557`，低于 `.70` shortcut ceiling。因此不必重做整套 468-state 训练数据，只需用
filtered development pool，并新建一次 12–16 user holdout。

### P2：一次新鲜小 holdout

原 internal 已消费，不能再次冒充确认集。只需新建 12–16 个独立 synthetic users，不做
16-action 全扫：

- 数据生成前冻结模板平衡和 shortcut gate；
- 每个 memory source 覆盖 admissible、irrelevant、stale/conflict 和 no-hit；
- 只评 learned policy、always-off、always-on/高资源 fixed；
- 每 source 最少 8 positive 和 8 nonpositive 独立 user/state groups；
- learned head 必须胜 prior，并同时产生至少 8 on、8 off；raw accuracy 不是晋级门。

### P3：最小 end-to-end 响应实验

只对 P2 中 PM on/off 都有覆盖的 24–32 个状态运行同栈 matched responses。每个状态不再
生成全部 16 action，只生成：

- learned policy；
- always-off；
- high-resource fixed。

盲评只做这一批。若没有足够 memory-on 风险事件，risk 明确作 descriptive limitation，
不再为凑事件无限加人评。

### P4：外部实验

- ESConv：主要测试无长期 memory opportunity 时能否稳定关闭 MP/MS/ME，以及固定
  R0/RS treatment；不能作为 memory-on 泛化证明；
- EvoEmo/ES-MemEval-derived：使用同一 retriever、filter、PM、prompt、generator，测试
  ME 候选 admission 和 stale/conflict；此前分布和 metadata 已被反复查看，只能称
  development-informed external stress test；
- 不因外部结果修改 Bank、阈值、特征或 k；
- external 没覆盖的正类由 P2/P3 controlled holdout 支撑，不把“没有对应场景”写成失败。

## 9. 可以发表的主张

若 P2/P3 通过，第一篇可以主张：

> 在合成、受控且 group-held-out 的情绪支持状态中，一个低容量、可审计的
> retrieve-then-admit policy 能为至少一个 memory component 学到非退化的 evidence
> admission signal；在固定生成栈下，相对 always-on/high-resource policy 减少 prompt
> 资源，同时保持冻结 rubric 下的即时支持质量，并报告 interaction-and-grounding risk
> proxy。Strategy RAG 的状态级边际收益未被学会，外部结果属于
> development-informed stress tests。

不能主张：

- PM 诊断了真实用户需求；
- 四个 component 或全部 16 action 都已学会；
- 证明了临床安全、长期情绪改善或真实用户偏好；
- BAAI 是最优判断模型；
- EvoEmo 是完全未触碰的独立外部集。

这条主张比旧“16 动作全学会”窄，但证据链完整、可审计，而且保留了研究的核心：
在候选资源真实可见的条件下学习何时注入，并以 quality、risk、cost 三组不混合的指标
验证。

## 10. 当前可复跑证据

- 补充人评分析：
  `outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_analysis/supplement_analysis_report.json`
- RS 正式前瞻/OOF：
  `outputs/pm_v1_5_corrected_pm_rs_v1/pm_rs_corrected_report.json`
- 受控 MP/MS/ME 诊断：
  `outputs/pm_v1_5_controlled_memory_routing_heads_v1/controlled_memory_routing_report.json`
- 构造捷径审计：
  `outputs/pm_v1_5_controlled_memory_routing_heads_v1/posthoc_shortcut_audit.json`
- candidate-admission 数据复用审计：
  `outputs/pm_v1_5_candidate_admission_data_audit_v1/candidate_admission_data_audit.json`
- 本文对应脚本：
  `scripts/v1_5/24ac_analyze_rs_corrected_supplement_v1_5.py`、
  `24ad_train_confirm_corrected_pm_rs_v1_5.py`、
  `24ae_train_controlled_memory_routing_v1_5.py`、
  `24af_audit_controlled_memory_routing_shortcuts_v1_5.py`、
  `24ag_audit_candidate_admission_data_v1_5.py`。
