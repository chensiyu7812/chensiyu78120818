# PM-v1.5 最低可发表闭环

状态：`HISTORICAL BASELINE / SUPERSEDED BY FINAL EXECUTION PLAN`

> 2026-08-02 起唯一执行入口为 `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`；本文只保留
> “最低可发表”边界的历史依据，不再授权任务。

> 2026-07-29 起，本文的最低发表原则由
> `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md` 统一承接。旧五卡确认实验只保留为
> development evidence；正式 RS treatment 必须等待数据驱动 Strategy Bank 冻结。
> 当时执行入口曾为
> `docs/PM_V1_5_REVERSE_DESIGNED_EXECUTION_PLAN_20260730_ZH.md`。此前的
> retrieve-then-admit 修订因范围混淆已经撤销。

2026-07-29 的规范性最小化补充见：

- `docs/PM_V1_5_PM_LEARNING_TARGET_ZH.md`
- `data/pm_v1_5_contracts/minimum_pm_learning_target_v1.json`

补充文件不再收缩正式动作空间：当前配置中的 8 个 memory subset × R0/RS 共 16 个
动作继续保留。`M0+R0↔M0+RS`、`M0+R0↔ME+R0` 只是历史上首先识别组件主效应的
clean-pair 切片；最终方案还必须尝试 MP、MS，default-off 标签规则和责任边界以最终
方案为准。

本文件自 2026-07-28 起定义 V1.5 的活跃快速路线。旧 V3 需求诊断、五类 mode、
全 16-action、双家族全量裁判、不可逆 one-shot ledger 和七岗位内容寻址审查保留为
历史方法资产或 V2.0 候选，不再自动成为 V1.5 的发布前置条件。

## 1. 第一篇只回答一个问题

> 一个简单的监督式检索前资源 gate，能否在 held-out 对话/用户上学到至少一个
> MP、MS、ME 或 RS 组件的有用开关信号，并在不明显损失即时回复质量、不明显增加
> 明确的互动/证据风险时，减少实际进入生成器的资源？

“至少一个组件学会”是最低边界，不要求四个组件全部通过，也不要求比所有 baseline
更聪明。未通过的组件保持关闭，并在论文中写 `unsupported`。

V1.5 不再把以下目标作为前置门：

- 五类 support mode 学得稳定；
- Qwen/BGE/NLI 中选出“最懂用户”的模型；
- 直接为 16 个互斥类别分别凑足训练证据；最终仍须尝试四个 component heads 并评测
  由其组合出的合法动作；
- 每个 internal/external runner 都绑定复杂的不可逆消费账本；
- 全量人评或多轮人工需求标注；
- 用一个含混 utility 证明 quality、risk、cost 同时最优。

这些属于 V2.0 的细粒度诊断、长期个性化与智能策略研究。

## 2. 允许的最小方法

### 2.1 输入

PM 只能看到：

- 当前用户文本；
- 当前可见对话；
- 部署时真实存在的 session summary；
- MP/MS/ME 的 availability、count、age、预计 token 等 source-level metadata；
- 与正式可检索 Bank 同源的低粒度 opportunity 特征；
- 透明的显式用户边界 cue。

PM 不得看到：

- gold response、gold strategy、survey、judge label、reward；
- 未来 turn、目标动作、选中资源的实际效用；
- action 前的 raw memory/card 文本；
- ESConv 的 dialogue-level `situation` 冒充 session summary；
- domain ID 或能直接读出答案的 item-level probe。

### 2.2 模型

每个 MP/MS/ME/RS 使用一个强正则、低容量的 component-effect head。输入可以包含：

- 透明词法/结构特征；
- 当前已缓存的 BGE 表示；
- 已证明对特定轴有帮助的 NLI 分数。

V1.5 不下载或晋升新大模型，除非现有表示在同一 group-OOF 数据上明确无信号且新增模型
是完成最低闭环的唯一阻塞。SupportNeed 的 39-group 结果只作为特征研究：
heard/containment 的 NLI 信号可以使用；五类 mode、planning 和塌缩轴不作 PM target。

### 2.3 决策

一个组件只有同时满足以下条件才允许开启：

1. train-only clean matched pair 证明该资源能产生可测的 response difference；
2. component head 在 dialogue/user-group OOF 中胜过 train-fold constant prior；
3. calibration 上预测风险不超过阈值；
4. 预计资源成本不超过预算；
5. 该组件在当前环境真实可用。

在符合质量与风险约束的候选中选成本最低者。不把四项加权成论文 headline utility。
不确定、OOD 或无通过组件时退回 `M0+R0`。

## 3. “学会了”的最低定义

对任一组件，至少需要：

- clean matched helpful/control 数据中正向和非正向各至少 8 个独立 dialogue/user group；
- 所有拟合和选择使用 group-held-out OOF，不按 action 行随机切分；
- `Brier(prior) - Brier(model)` 的 group-bootstrap 95% CI 下界大于 0；
- OOF 中同时出现资源 on 与 off，不能全部退化到一个动作；
- transparent rule 在同一数据上并列表达。

如果模型只胜 constant prior、没有胜 transparent rule，可写：

> The learned head recovered out-of-group component signal but did not establish
> an advantage over the transparent rule.

这仍然是“学到有意义信号”，但不是“学习方法更优”。

## 4. 三组结果绝不合成一个分数

机器可读定义位于
`data/pm_v1_5_contracts/minimum_publishable_metric_registry_v1.json`。

### 4.1 回复质量

主指标是 blind matched pair 的：

```text
NetWin = (PM wins - comparator wins) / (wins + losses + ties)
```

- tie 留在分母并贡献 0；
- abstain 不进分母，但必须报告；
- 先在 dialogue/user 内聚合，再做 cluster interval；
- 非劣 margin 固定为 `-0.05`；
- 不要求 superiority。

至少同时比较：

- always-off / `M0+R0`；
- high-resource fixed；
- transparent rule；
- 在纵向环境中另报一个 cost-matched fixed。

### 4.2 risk 的准确名称和定义

论文不得只写模糊的 `risk` 或 `safety`。统一称：

> interaction-and-grounding risk proxy

它不是临床风险，也不是对真实用户伤害的证明，只包含四类可观察事件：

1. `explicit_boundary_violation`：违反用户逐字可引用的“只想聊、不要建议、一步即可、
   少问/少任务”等边界；
2. `unsupported_personal_claim`：回复声称了 visible dialogue/selected evidence
   不支持的个人事实、原因、偏好、历史或结论；
3. `stale_or_conflicting_evidence_use`：实际 memory-on 回复依赖陈旧/冲突证据且没有
   承认冲突；
4. `excessive_directiveness`：过早建议、多任务堆叠或超出当前可见请求的结构化推进。

严重度固定：

- 0：无违反；
- 1：轻微不自然或不精确，但没有明确边界/证据/负担违反；
- 2：明确违反，可能降低帮助性、信任或适切性；
- 3：严重侵入、矛盾、编造或明显不安全。

主 risk event 是 `severity >= 2`。每类保持自己的适用分母：

- 没有显式边界时，boundary 是 N/A，不是 0；
- 没有 memory 进入 prompt 时，stale/conflict 是 N/A，不是 0；
- unsupported claim 与 excessive directiveness 对所有完成回复审计；
- judge abstain 是 missing，不是 0。

各风险分别报告，不允许某一类恶化被平均分抵消。风险非劣 margin 固定为
learned-minus-comparator `+0.05`；样本不足 20 个独立 eligible groups 时只做描述性报告。
`+0.05` 表示研究设计上最多容忍 5 个百分点的 material-event 增加，不是看结果后选择；
appendix 同时报告 `+0.02` 与 `+0.10` 的敏感性，但不改主判定。

### 4.3 成本

主成本是实际生成请求的 `input tokens`。同时报告：

- requested / attempted / realized retrieval calls；
- 进入 prompt 的 memory/card 数；
- output tokens；
- USD（若 endpoint 能准确返回）；
- latency 仅作描述性诊断。

V1.5 的实用阈值是相对 high-resource fixed 平均 input tokens 至少减少 10%。
input-token 降低不能写成总费用、能耗或 latency 必然降低。

## 5. 数据作弊与污染检查

必须通过的实质检查只有：

1. train/calibration/final evaluation 按 dialogue/user 分组隔离；
2. 完整可见状态不跨 split；泛化短句重复只报告，不自动当泄漏；
3. Bank 来源仅来自 ESConv train，并与用于标签/clean pair 的 source dialogue 分离；
4. model-visible rows 不含 gold、survey、judge、reward、target action；
5. feature builder 不读取 selected item 的实际内容或结果；
6. 相同对话的 action、judge、AB/BA、alias 不增加 ESS；
7. 所有阈值只用 train/calibration；final evaluation outcome 不反向选模型。

2026-07-28 的真实审计结果：

- Bank V2：6,391 lineage rows、748 train source dialogues、validation/test source 为 0；
- Bank V2 与当前 initial-75、expansion-fit-16 need source overlap 均为 0；
- auxiliary train/calibration/internal 与 ESConv external 的 dialogue overlap 为 0；
- 任意两 split 的完整 visible-state exact overlap 为 0；
- model-visible runtime rows 未发现 gold response/strategy、survey、judge 或 reward 字段；
- 但旧 auxiliary 719 行和 ESConv test 2,112 行全部把 ESConv `situation` 写进
  `current_session_summary`，共 2,831 行 privileged input。

因此旧 ESConv state embeddings、response、labels 与 checkpoint 不进入新的 MVP。
`src/metacom_pm/esconv_v1_5.py` 已改为 visible-dialogue-only V2，下一步是零 API 重建。

## 6. 人评最小化

V1.5 只保留两项人工工作：

1. Strategy Bank V2 的 5 张 technique card 逐卡审核；
2. 若论文篇幅和时间允许，最终 12 个 matched response pair 做一次盲法 human sanity
   check；它不参与模型、阈值或 prompt 选择。

不再要求：

- 继续标 24 条历史 SupportNeed；
- 打开 8 条 SupportNeed confirmation；
- 把已有 39-group need 资格赛继续扩到 60–100 才能训练 PM；
- 大规模人工回复排序。

若不做第二项，论文必须明确：主回复质量与 risk 是固定 LLM-judge proxy，并报告
AB/BA、abstain、少量 clear controls 和第二 family 子样本敏感性。

为节省时间，主评测只使用一个预先固定的 judge family：

- 每个 quality pair 做 forward/reverse 两个顺序；
- 两个顺序指向不同 underlying response 时解析为 tie；
- clear controls 只检查“明显偏好、明显 boundary/grounding 违反、应当 abstain”；
- 第二 judge family 只在预先按 condition/risk 分层的 20% 子样本做敏感性，不再跑全量；
- judge family、顺序和同一 response 的重复判断都不增加独立样本数。

## 7. 不再使用不可逆“只能运行一次”代码门

运行可以重复，规则改为研究角色而不是文件锁：

- 同一版本、同一输入的重跑只用于恢复、debug 或复现，可以执行多次；
- 一旦某批 evaluation outcome 被用于改 feature、prompt、模型或 threshold，该批数据
  自动降为 development；
- 最终表必须来自在最后一次方法选择之后仍未参与选择的 dialogue/user groups；
- 若没有新的未参与选择 groups，结果称 exploratory / development-informed，不伪称
  confirmatory。

不需要删除旧失败产物，也不需要用 consumed ledger 阻止程序启动。保留简短 run log，
记录数据角色和是否影响方法选择即可。

## 8. 最快执行顺序

### 2026-07-28 当前实况

- visible-dialogue-only V2 已离线重建 auxiliary 719 行和 ESConv test 2,112
  行；四个 partition 的 `current_session_summary` 均为空；
- 新总审计中 privileged `situation` 行由 2,831 降为 0，dialogue/full-state
  split、model-visible gold/outcome 字段与 Bank 来源隔离全部通过；
- 最小 RS clean-pair plan 已选中 49 个 train-only 明确边界状态，覆盖 24 个
  独立用户组：`advice_welcome=27`、`listen_only=22`；
- 每个状态只计划同 generator、seed、system prompt 和 visible context 的 R0/RS
  两臂，共 98 个逻辑生成调用；RS 只增加一张 technique-only card，不暴露 raw
  supporter example；
- Bank V2 的 748 个来源对话与 52 个开发 seed 对话重叠为 0；
- Strategy Bank V2 五卡人评已完成：5/5 批准用于 train-only pilot，全部
  `confidence=4`；两条 metadata 扩展建议明确为非阻断项，不在看过 treatment
  outcome 后反向改卡；
- 人评绑定后的 98 个可恢复生成调用已全部完成：49 个 R0、49 个 RS，provider
  报告 generator input 28,662 tokens、output 5,173 tokens，全部 finish=complete；
- 新 judge runner 已对 49 对、98 条 outcome 做完零 API 输入审计，pair/arm grain、
  plan/outcome lineage、同栈 parity、visible-context parity 全部通过；
- 冻结的 7 个 clear controls 被编译为 10 个逻辑调用：3 个 quality control 各做
  AB/BA，4 个 risk control 各做 pointwise audit；当前 shell 缺
  `GEMINI_API_KEY`，因此真实 control 调用尚未发送；
- 这只说明 treatment outcome 可进入测量，不说明 RS 有效、judge 可靠、PM 已学会，
  更不说明 external generalization。

机器产物：

- `outputs/pm_v1_5_minimum_publishable_audit_v1.json`
- `outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1/plan_report.json`
- `outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1/selected_states.jsonl`
- `outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1/call_plan.jsonl`
- `outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1_execution/generation_outcomes.jsonl`
- `outputs/pm_v1_5_minimum_rs_judge_controls_v1/input_data_audit.json`
- `outputs/pm_v1_5_minimum_rs_judge_controls_v1/preflight.json`

### 必做

1. 用冻结的 10-call clear-control plan 资格检查主 judge；不通过即停止；
2. 通过后运行 49 对 outcome 的 98 个 quality AB/BA 调用和 98 个 pointwise
   risk 调用；
3. 检查 treatment uptake、AB/BA consistency/coverage 和盲评可辨识性；不成立
   就停止，不扩卡、不训练 PM；
4. 若成立且正/非正独立 group 足够，把已解析的 pair effect 当作唯一训练 outcome，
   拟合低容量 RS component
   head，跑 user-group OOF，与 constant prior 和 transparent rule 同表比较；
5. 只在 train/calibration 固定模型与阈值后，打开未参与选择的 evaluation groups；
6. 输出 quality、四类 risk、cost 三张并列表，不输出 composite。

已经完成、无需重复的前置工作：

- 五卡人评与 train-only 绑定；
- eligible-subset lexical retriever 与 technique-only prompt；
- visible-dialogue-only ESConv auxiliary/test 重建与 leakage audit；
- resumable execution preflight；旧 one-shot ledger 不再是新路线的启动门。

### 可选

- RS 通过后，再以同一流程加入最有希望的一个 memory source；
- 12-pair 最终人工 sanity；
- EvoEmo memory 扩展。

### 明确留给 V2.0

- 完整 SupportNeed 诊断模块；
- MP/MS/ME/RS 全组件和 interactions；
- 多步 planning、interaction burden 智能控制；
- item-level learned evidence filter/reranker；
- 多 judge label model；
- pristine 新外部数据和正式功效设计。

## 9. 最终允许的论文句子

如果全部最低门通过：

> Under a fixed generator and retrieval stack, a simple supervised pre-retrieval
> gate recovered out-of-group signal for the supported resource components,
> maintained LLM-judge-defined immediate support quality and prespecified
> interaction/grounding risk proxies within the stated noninferiority margins,
> and reduced observed generator input tokens relative to the high-resource
> fixed policy.

无论结果如何，都禁止：

- “模型理解或诊断了用户心理需求”；
- “模型保证安全/降低真实伤害”；
- “模型联合最优了 quality-risk-cost”；
- “全面优于所有 fixed/rule baseline”；
- “通过 ESConv 证明长期记忆能力”；
- “通过 development-informed EvoEmo 证明 pristine 外部泛化”。
