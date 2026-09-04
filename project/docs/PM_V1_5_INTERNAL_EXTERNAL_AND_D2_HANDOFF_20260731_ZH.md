# PM V1.5 内部超集—外部子域与 D2 执行交接

> **2026-08-01 修订：** 本文第“研究主张没有改变”节记录的是 D2 当时的
> paired-effect 目标。当前 Step 1 已按
> `resource_routing_measurement_layers_v1.json` 改为 outcome-blind resource-opportunity
> routing；D2 soft effect 只保留为 generator/system-effect 测量证据，不再逐状态生成 PM
> hard gold。内部超集—外部子域、同 compiler/retriever/prompt/generator 与外部 outcome
> 不回流的约束继续有效。

状态：`D2 GENERATION COMPLETE / HUMAN QUALITY REVIEW PENDING`

日期：2026-07-31

## 结论

当前路线已经冻结为：

- **外部能直接测的能力，由外部数据测。**
- **外部数据没有提供的能力，由内容不重叠的内部受控环境补测。**
- 内部不复制 ESConv/EvoEmo 的具体对话、用户、记忆或 outcome，而是覆盖相同资源
  ontology、同一运行机制和外部会遇到的候选特征范围。
- 外部只替换成当前真实测试用户自己的私有记忆内容。

这不是“两套 PM”。PM、动作空间、候选描述、检索器、过滤器、prompt 和 generator
都相同；不同的只是不同用户本来就应当不同的私有记忆内容。

## 各数据域负责什么

| 能力 | 内部受控环境 | ESConv | EvoEmo |
|---|---|---|---|
| RS 策略资源开关 | 测 | 测 | 可报告但不是唯一主证据 |
| memory-off | 测 | 测 | 测 |
| MP_PROFILE | 测 | 不测 | 测 |
| MP_PREFERENCE | 主证据 | 不测 | 数据不支持，不能声称测过 |
| MS_SESSION | 主证据 | 不测 | 测 |
| MS_PATTERN | V2 | 不测 | 当前不测 |
| ME 历史事件 | 主证据 | 不测 | 测 |
| 完整 16 动作 | 主证据 | 不要求覆盖 | 覆盖到哪里报告到哪里 |

由于项目已经在不查看 quality/risk outcome 的条件下检查过 EvoEmo schema、历史深度和
候选 descriptor support，最终必须写成 `development-informed external`，不能写成
完全 untouched zero-shot。外部 quality/risk outcome 仍不得用于选样、调特征、阈值
或模型。

## MS 修复

V1.5 只保留：

> `MS_SESSION`：一个严格完成的 prior session 的合格摘要。

当前 session dialogue 是所有动作共同可见的 baseline，不是 MS。跨多个 session
归纳出的 pattern/evolution 暂留 V2。

旧 64 个 MS pair 全部降为诊断，原因是其中 59 个注入了至少一条
last-user-message/current-user-text fallback；只有 4 个同时满足 supplied-only summary
与空 current-session summary，不能在看过 outcome 后事后挑这 4 个。

新 D2 的 8 个 MS state：

- 全部没有进入旧 256 个 first-fit contrasts；
- selected MS item 全部来自 supplied prior-session summary；
- fallback 为 0；
- current-session summary 为空；
- 覆盖 MS 的 8 种其他组件 background。

## D2 已完成的执行

冻结蓝图：

- 32 个 state，每组件 8 个；
- RS/MP/ME 各 8 个旧 state：4 个历史 on、4 个历史 nonpositive；
- 这些旧 state 均使用空 current-session summary，保持原 pair 与新 replicate 的状态
  视图一致；
- 每个 RS/MP/ME state 新生成 2 个独立 pair；
- 8 个 fresh MS state 各生成 3 个独立 pair；
- 合计 72 个新 pair、144 个 response calls。

实际执行：

- 144/144 调用完成；
- 72/72 pair 均有 control/treatment 两臂；
- effect labels 在生成期间始终 unknown；
- generator、prompt compiler、资源栈与计划 hash 全部通过 preflight；
- 预计成本上限约 0.0384 美元。

机器证据：

- `outputs/pm_v1_5_d2_measurement_blueprint_v1/blueprint_report.json`
- `outputs/pm_v1_5_d2_measurement_generation_v1/plan_report.json`
- `outputs/pm_v1_5_d2_measurement_generation_v1_execution/execution_summary.json`

## 现在只剩一次有终点的人评

主审核：

- 72 个新 pair；
- 只评可见对话下 A/B/tie/uncertain 和决定性质量标准；
- 不评组件、资源、risk 或 cost；
- 轻微风格偏好必须记 tie。

第二位独立审核者：

- 固定 32 个 pair；
- 每组件 8 个；
- 用于测量 inter-rater exact agreement，不再追加第三轮散装人评。

页面：

- 主审核：
  `outputs/pm_v1_5_d2_measurement_blind_v1/primary_72/human_blind_review.html`
- 第二审核者：
  `outputs/pm_v1_5_d2_measurement_blind_v1/independent_overlap_32/human_blind_review.html`

冻结过门标准：

- 至少 70% state 得到 2/3 相同的非 uncertain 方向；
- 32 个共享 pair 的双人 exact agreement 至少 0.75；
- uncertain 比例不超过 0.10。

通过后才进入 D3 覆盖补齐和正式重训；不通过则不再把单次生成胜负当 hard gold，
改用重复 Bernoulli/soft effect，或收窄 V1.5 的证据边界。

## 研究主张没有改变

Step 1 仍学习：

> 在当前可见 state、实际合格候选和固定其他组件 background 下，加入当前组件带来
> 实质质量收益的概率。

随后独立应用：

1. atomic interaction-and-grounding risk guard；
2. 冻结概率阈值；
3. deterministic token/cost budget；
4. 形成 16 个合法动作之一。

Step 2 generator 只接收最终选中的资源并生成回复。Step 1 看不到生成回复、judge、
未来 turn、gold action、数据集 ID 或外部 outcome，因此没有通过 outcome 泄漏作弊。
