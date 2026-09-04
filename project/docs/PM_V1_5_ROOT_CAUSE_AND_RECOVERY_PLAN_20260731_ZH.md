# PM V1.5 首轮四组件训练失败：根因与恢复方案

状态：`ROOT CAUSE IDENTIFIED / RECOVERY CONTRACT FROZEN / 2026-07-31`

机器合同：
`data/pm_v1_5_contracts/first_fit_root_cause_recovery_v1.json`

审计事实源：
`outputs/pm_v1_5_first_fit_root_cause_audit_v1/root_cause_audit.json`

当前执行状态（2026-07-31）：

- D1/D1b 合同与零 outcome 审计已完成；
- D2 已冻结 32 个 state、72 个新 pair、144 个 response calls；
- 144/144 response calls 已完成，72/72 pair 两臂齐全；
- 当前只等待主审核 72 项与第二审核者固定重叠 32 项；
- 交接报告：
  `docs/PM_V1_5_INTERNAL_EXTERNAL_AND_D2_HANDOFF_20260731_ZH.md`。

## 结论先行

这一次确实是“没有训好”，但目前不能把原因简化成“数据差”或“BAAI 不行”。
按证据强弱排序，根因是：

1. **训练实现没有真正执行已经冻结的低容量、state × candidate × background 合同。**
2. **标注行数看似够，独立用户数和候选覆盖其实不够。**
3. **内部与 EvoEmo 只覆盖 MP 的不同子类型，且内部 preference 高度模板化。**
4. **旧 MS 把当前用户话语 fallback 当成历史 session summary，且内外当前会话可见范围不同。**
5. **单次生成、单人裁决是否足够稳定尚未测量。**
6. **BAAI 不是已经证实的主因，只是没有救活前五个问题。**

因此，当前结果不能支持 PM 已学会，也不是对研究主张的一次干净否证。最合理的恢复
不是立刻换大模型、继续扩 RAG 或再做几百条零散人评，而是先用一个有终点的小实验
测出标签是否可重复，再按严格容量重建四个 head。

## 1. 这轮结果究竟说明了什么

开发集 grouped-OOF balanced accuracy：

| 组件 | 透明主模型 | BGE-small challenger | 预注册及格线 |
|---|---:|---:|---:|
| RS | 0.622 | 0.642 | 0.700 |
| MP | 0.503 | 0.525 | 0.700 |
| MS | 0.468 | 0.572 | 0.700 |
| ME | 0.445 | 0.482 | 0.700 |

四个 head 都未过门。BGE 对 MS 有一定提升，但仍未及格；它没有把其余 head 从随机
附近救出来。因此可以排除“只缺一个通用语义 embedding”是主要解释，但不能据此说
所有更强语义模型都无用。

这轮最重要的价值是暴露了实验设计与执行之间的断裂，而不是得到一个可以写进论文
主结果的 PM 分数。

## 2. 已确认的第一根因：方法实现偏离合同

原计划明确要求：

> 主特征数不超过 `floor(独立开发组数 / 5)`。

实际情况：

| 组件 | 开发行数 | 独立用户 | 允许特征上限 | 实际特征 | 是否合规 |
|---|---:|---:|---:|---:|---|
| RS | 48 | 28 | 5 | 9 | 否 |
| MP | 48 | 31 | 6 | 8 | 否 |
| MS | 48 | 32 | 6 | 8 | 否 |
| ME | 48 | 29 | 5 | 8 | 否 |

这不是“代码是否优雅”的工程问题，而是统计容量问题：少量独立用户支撑不了这么多
自由度，OOF 结果会对分组和 seed 敏感。

更关键的是，MP/MS/ME 把背景动作写成：

- 其他 memory 组件一共开了几个；
- RS 是否开启。

但本研究的 estimand 是“在**一个固定的具体背景动作**下，只增加当前组件的边际
收益”。`MS+RS` 与 `ME+RS` 都会被压成同一个数字，模型看不到两种背景的差异。
开发标签却已经显示背景差异很大。例如 MP 在 `M0+R0` 背景为 6/6 on，在 `MS+RS`
背景仅为 1/6 on。

用现有数据做了一个只用于诊断、不能晋级模型的 post-hoc probe：保留两个 BGE
相似度，再显式放入三个其他组件 bit。ME BA 从约 0.482 升到约 0.561，证明背景
表示错误确实存在；仍未到 0.70，证明它又不是唯一原因。

## 3. 已确认的第二根因：数据“有 48 行”不等于“有 48 个独立样本”

每个 head 的开发标签有 48 行，但只来自 28–32 位不同用户。完整 train+calibration
池也只有 36 位用户，根本达不到原计划的 48 个独立组。

另外，contrast blueprint 主要按 `top1 lexical score` 分位数选样，没有同时守住：

- 具体 background action；
- candidate count；
- age；
- token cost；
- relevance margin；
- internal/external transport range；
- ESConv-like 与 longitudinal-like 状态。

已有覆盖审计显示，完整内部状态对外部候选模式尚有部分覆盖，但被选去标注的开发
子集覆盖显著下降。也就是说，“原料池里有”不等于“训练样本里有”。

这解释了为何继续机械增加同类 pair 很可能仍然不稳定：如果新增的仍是相同模板、
相同背景压缩和相同分数区间，只会增加行数，不会增加可识别信息。

## 4. 已确认的第三根因：MP 子类型覆盖错位

这里修正一个过重的旧表述。项目历史语义合同对 MP 的定义本来就是：

> 稳定的 profile、preference 或 boundary memory。

因此，内部 preference 与 EvoEmo profile 不是互不相关的两个构念，而是同一个 MP
上位构念下的两个合法子类型。真正的问题是两域的 subtype support 完全错开。

内部 MP 共有 156 个 memory ID、72 段精确文本，看起来很多；去掉 preference 序号
与 topic 后实际只剩 1 个行为模板，并且 100% 都含有：

- “prefers acknowledgement before suggestions”；
- “with one gentle question at a time”。

EvoEmo 的 MP 则是 `name / age / gender / job / location / nationality / education`。
内外字段交集为 0。

所以“同一个 compiler”只解决了运输机制一致，没有解决 subtype coverage。模型在内部
只看到单一交互偏好，在外部却只面对人口与生活 profile。这会让外部 MP 成绩无法解释为
对完整 MP ontology 的泛化。

恢复合同保留原共同上位构念：

> 能直接、非刻板化地改变当前支持回复的稳定用户信息。

其下明确标注两个 subtype：

1. `MP_PREFERENCE`：稳定支持偏好；
2. `MP_PROFILE`：稳定人口/生活 profile。

V1.5 不新增第五个动作 bit。两类候选进入同一个 typed MP catalog，检索和过滤后最多
保留一条，仍由一个 MP bit 决定是否注入。MP head 额外使用一个透明
`candidate_is_preference` 特征。内部训练必须同时包含两类，以及成对的无关、敏感但
不相关、错人、已在当前窗口出现、把临时请求误当稳定偏好等负例。外部仍只换成当前
真实用户自己的 memory 内容，不换 schema、特征、检索器或 PM。

偏好还要分两层：

- 当前轮明确边界必须直接遵守，不算 MP 带来的收益；
- 明确、仍有效且无冲突的 hard durable preference 编译为有界的确定性 preference
  contract，不交给 PM 猜；
- 可能随情境改变收益的 soft support preference 才进入可选 MP 检索与开关。

## 5. 已确认的第四根因：MS 构念与当前会话运输错位

项目历史计划其实区分了三样东西：

1. **当前 session context**：当前对话本身，所有动作都能看到，不是一个资源 bit；
2. **`MS_SESSION`**：一段严格发生在当前 session 之前的、单个完整 session 摘要；
3. **`MS_PATTERN`**：至少由两个历史 session 支持的重复模式或状态演化。

当前代码只实现了“每个历史 session 一条 MS item”，并没有实现跨 session pattern
compiler。因此 V1.5 正式范围冻结为 `MS_SESSION`；`MS_PATTERN` 留给 V2。动作空间
仍是一个 MS bit、总共 16 个动作，不增加新 head。

零 outcome lineage 审计发现，旧 64 个 MS clean pairs 中：

- 59 个 treatment 至少注入了一条把 `current_user_text`/最后一句用户话冒充 session
  summary 的 fallback；
- 只有 5 个 pair 的 MS item 全部来自真实提供的 session summary；
- 只有 4 个 pair 同时满足“summary 全部合格”和“当前 session summary 为空”。

这不是从 64 个里挑 4 个继续训练的理由。因为这一缺陷是在看过旧 outcome 后发现的，
事后挑 4 个会引入选择偏差；所以旧 64 个 MS pair 全部降为诊断证据。

还有一个独立错位：内部状态有一半带 current-session summary，EvoEmo runtime 则始终
为空，并且在第 8 个 seeker turn 只保留最近 8 条消息，遗漏更早的 6 条消息。这样
“MS 是否补足历史”会与“当前 session 是否被截断”混在一起。

V1.5 的修复很简单而且可审计：

- bounded 10-turn 实验中，内外都给 generator/PM 完整当前 session dialogue；
- 不另加 current-session summary；
- MS 只允许使用人工或固定 compiler 产生的、严格过去时的 session summary；
- 禁止 last-message fallback；
- D2 为 MS 新建 8 个 outcome-blind 合格 state，不复用旧 MS state。

## 6. 尚未确认：标签本身到底有多噪

当前 256 对标签不是完全不可信：

- schema 完整；
- 52/52 个 A/B 位置反转复测得到相同语义裁决；
- 109 个组件开启胜例做过最小 interaction-and-grounding risk 审核。

但这些只能证明同一审核者对位置稳定。尚未测量：

- 同一 state 换一次 generator sampling，开/关方向是否还一样；
- 第二位审核者是否同意“实质更好 / 更差 / 等价”；
- 单次 pair winner 能否代表“较大概率带来实质收益”。

因此，现在说“数据标签坏了”证据不足；直接相信 256 个 hard labels 同样证据不足。

## 7. 研究主张不需要改，Step 1 / Step 2 也没有跑偏

保留的主张是：

> 在同一套资源库、检索、过滤、prompt、generator 和评测下，一个低容量、可审计的
> 回复前 PM，能否学习 MP、MS、ME、RS 各自的预期边际质量收益开关，再施加独立的
> atomic risk guard 与确定性 cost 约束。

因果顺序保持：

1. 固定检索器发现候选，并先执行无候选、错人、冲突、过时请求、明确边界等 hard-off；
2. Step 1 只看回复前可见的 state、合格候选摘要和三个明确 background bits，输出
   `P(加入该组件会带来实质质量收益)`；
3. 依次应用 risk、概率阈值和 cost budget，组合成 16 个合法动作之一；
4. Step 2 generator 只接收被选资源并生成回复。

Step 1 看不到回复、judge 分数、未来 turn、gold action、数据集身份或外部 outcome，
所以修复方案没有引入 outcome leakage。它也不把“资源适用”重新冒充成收益标签：
适用性仍是 hard gate 和候选特征；收益仍由同状态开关 pair 定义。

## 8. 修复后的最小可学模型

RS/MS/ME primary heads 各用 5 个特征：

1. `candidate_state_match_score`；
2. `candidate_grounding_or_nonredundancy_score`；
3. 另外三个组件各自独立的 on/off bit。

MP head 再加入一个 `candidate_is_preference`，共 6 个特征。正式恢复数据达到至少
48 个独立开发用户后，6 仍低于 `floor(48/5)=9` 的容量上限。

两个分数由每类资源冻结的透明规则产生：

- RS：卡片与当前状态匹配度；required cues/permission/burden 的满足度；
- MP：当前相关度；稳定、非敏感、未重复的 grounding；
- MS：当前相关度；合格的 prior-session summary 是否真正补足跨 session 连续性；
- ME：当前相关度；时间、冲突、错人、旧请求和重复证据检查。

hard exclusion 在模型之前执行，不能指望 logistic 学会安全规则。cost 在质量概率之后
确定性处理，不能与质量相互抵消。

BGE-small 可以在未来只提供第一个 match scalar，或作为同样 5/6 特征预算下的 challenger；
它不负责定义标签、安全或 gold。D2 和覆盖门未过前，不比较 Qwen、更大 BGE 或 NLI，
否则只是在混杂设计上挑一个碰巧分数高的模型。

## 9. 下一步不是再做两天散装人评，而是一次有终点的 D2

D2 只回答一个问题：

> 当前 pair label 是否足够稳定，值得继续训练？

固定方案：

- 32 个 state，每组件 8 个；
- RS、MP、ME 共 24 个 state：尽量来自不同用户，每组件含 4 个原 on-benefit 与
  4 个原 nonpositive，每个 state 再独立生成 2 对；
- MS 的旧 pair 全部失效，因此新建 8 个 outcome-blind、合格 `MS_SESSION` state，
  每个 state 独立生成 3 对；
- 共 144 次 response API 调用；
- 主审评 72 个新 pair；
- 第二审只评固定的 32 个重叠 pair；
- 总人评决定 104 个，不再追加微包。

过门标准在生成前冻结：

- 至少 70% 的 state，原 pair + 两个新 pair 中有 2/3 给出同一非 uncertain 方向；
- 32 个共享 pair 的双人精确一致率至少 0.75；
- uncertain 不超过 0.10。

若通过，RS/MP/ME 用原 pair 加两个新 pair 的多数方向，MS 用三个全新 pair 的多数
方向，作为恢复开发标签。若不通过，不再把一次随机生成胜负当 hard label；改用重复
Bernoulli/soft expected-effect，或把 V1.5 主张缩到资源能力与路由可行性。这个判断
能真正区分“标签噪声”与“方法/覆盖问题”。

## 10. D2 通过后才扩数据，而且只扩缺口

现有 fit、已看过的 internal test 和 40 条 ESConv panel 全部降级为 development/
diagnostic evidence，不冒充 repaired confirmation。

D3 至少新增 32 位内容不重叠的开发用户，每位对四个组件各贡献一个 contrast，共
128 对。这样既能让四个 head 达到至少 48 个独立开发用户，也能为 MP 单独安排：

- 16 位 `MP_PREFERENCE` 用户：8 个 on-benefit、8 个 nonpositive；
- 16 位 `MP_PROFILE` 用户：8 个 on-benefit、8 个 nonpositive。

MS 的 D3 数据必须额外满足：

- 100% 来自合格的 strictly-prior session summary；
- last-message fallback 为 0；
- 内外 current-session visibility policy 完全相同；
- `MS_SESSION` 与 ME event chunk 不得只是同一文本换标签。

选择在 outcome 产生前
按 component、on/off blueprint、完整 background、match、grounding、count、age、
cost 和 domain shape 分层。

只有 repaired grouped OOF 与 calibration 过门后，才生成一次全新的 user-disjoint
internal confirmation；RS 外部确认使用未消费的 ESConv 对话。EvoEmo 由于已经做过
不看 outcome 的 schema/descriptor-support 检查，必须诚实称为
`development-informed external`，不能称为 untouched zero-shot；其 quality/risk
outcome 仍不得参与任何 feature、sample、threshold 或 model 选择。

## 11. 这次明确禁止再踩的坑

- 不因一次失败把某组件永久默认关闭；
- 不把“适用”重新偷换成“有收益”；
- 不把 48 行说成 48 个独立样本；
- 不用 component count 代替具体 background identity；
- 不把同名 MP 当作天然同构；
- 不把当前用户最后一句话冒充 prior-session summary；
- 不把 current-session 截断造成的缺失错误归功于 MS；
- 不用 outcome-free 外部结构审计后仍声称 untouched zero-shot；
- 不在标签稳定性未知时扩大几百条同类 pair；
- 不在修复透明 primary 之前搜索更大模型；
- 不再打开 Strategy Bank 或新增 RAG 人评来修 PM 数据问题；
- 不用已经看过的 ESConv/internal outcome 调 threshold 后再称为外部确认；
- 不降低 0.70 及格线来宣布学会。

## 12. 当前科学判断

当前最准确的表述不是“PM 已失败”，也不是“只差换个模型”，而是：

> 四组件资源效应存在可测的候选信号，但首轮 PM fit 不是对预注册主张的合格检验。
> 已确认的首要问题是特征容量、背景表示、独立组、MP 子类型覆盖与 MS 构念运输；
> 标签可重复性仍需一个 32-state 有界实验决定。MS 必须先完成零 outcome 构念修复，
> 之后才能运行该测量门；只有门通过，才值得继续补覆盖并重新训练。

这保住了研究的核心：不是要求 PM 全部选对，而是证明一个简单、透明、不作弊的
Step 1 在固定系统中确实学到高于基线、可复现的资源开关规律，并最终改善
quality–risk–cost 平衡。
