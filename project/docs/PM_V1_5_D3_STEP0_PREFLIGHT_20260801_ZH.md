# PM V1.5 D3 Step 0 冻结报告（2026-08-01）

## 结论

D3 的零成本、无结果阶段已经通过并冻结。当前不是“PM 已学会”，而是终于得到一套可以开始生成训练结果、且不会重演旧版混杂与作弊问题的设计：32 个全新用户、128 个单组件同状态对照、32 个预先选定的重复对，共 160 对、320 次回复生成。

320 次生成现已全部完成；人工标签尚未产生，外部结果没有读取。

## PM 到底会看到什么

每个 component head 只看到：

- 同一运行时检索器给出的候选—状态词法匹配分数；
- 六项可审计检查汇成的 grounding/nonredundancy 分数；
- 另外三个组件是否已在背景动作中开启；
- MP 额外看到候选是 preference 还是 profile。

RS/MP/MS/ME 分别只有 5/6/5/5 个特征。用户、数据集、state/card/memory ID、回复、judge、人工结果、未来轮次和 topic/template 标记都不进入模型。

## 硬规则与学习目标的边界

以下情况不作为有噪声的训练题，而是在 PM 之前直接关闭：错用户、当前或未来记忆、显式冲突、过期回合请求、来源不合格、RS 的寒暄/结束/高风险/无合格卡。

通过硬门后仍可能“相关但重复”或“可用但没有实际增益”。这部分才由 PM 学习加入组件后产生实质质量收益的概率。因此没有把“资源适用”偷换成“资源有益”。

## 数据覆盖

- 32 个用户全部是新构造内容；与 ESConv、EvoEmo 做构造后精确文本审计，重叠均为 0。
- 私人记忆仍由同一个 `build_evo_memory`/bounded-history compiler 生成；没有把外部用户内容混进训练库。
- 所有状态均有 MP、MS、ME 与 RS 候选，所以运行时 16 个动作全部保留。
- 每个 head 的其余三位背景动作共 8 种，每种恰好 4 个用户。
- MP 为 16 个 preference、16 个 profile；preference 至少覆盖 8 种表达家庭。
- MS 全部来自人工提供的严格先前 session summary，未使用 last-message fallback；当前 session summary 统一为空。
- 32 个硬门压力项（错用户、未来、过期请求、显式冲突各 8 个）全部 fail closed，且不消耗生成调用。

## 发现并修掉的一次新捷径

第一版 Step 0 为了构造“重复证据”，把记忆原句直接放进当前上下文。这会自动抬高 lexical match，导致“高匹配”和“低非重复性”绑定，模型无法识别两个不同概念。该版本没有进入结果生成。

最终版把四个组件的 match 与 grounding 分层分别安排。四个 head 的 match×grounding 四格均有覆盖，每格至少 4 个用户；RS 为严格 8/8/8/8。这个检查已写入 fail-closed preflight，后续重建若复发会直接报错。

## 已冻结规模与成本

- Primary：128 对，每用户每组件一对；
- Repeat：32 对，每组件 8 对，在任何结果产生前选定；
- 总计：160 对、320 次生成；
- 每对 control/treatment 使用同一 seed；
- 冻结价格代理下的上界约 0.078141 美元。

## 当前状态与下一步

320/320 调用全部 `complete`，160 个 pair 均恰好具有 control/treatment 两臂，call ID 无重复、回复无空值。唯一完整质量任务已经生成：

`outputs/pm_v1_5_d3_blind_review_v1/human_blind_review.html`

页面共 160 对，A/B 在 component×primary/repeat 层内严格平衡；组件、开关、重复身份和 private key 均不进入页面。完成后只对“组件开启且质量胜出”的条目做一次更小的 grounding-risk 筛查，不再进行 RAG/card 审核或第二轮全量质量人评。随后将重复对合并成 state soft target，以 user-grouped OOF 一次性训练四个 L2 logistic heads，并逐 head 应用预先冻结的及格门。

## 主要证据

- `outputs/pm_v1_5_d3_step0_blueprint_v1/preflight_report.json`
- `outputs/pm_v1_5_d3_step0_blueprint_v1/d3_contrast_blueprint.jsonl`
- `outputs/pm_v1_5_d3_step0_blueprint_v1/hard_gate_controls.jsonl`
- `outputs/pm_v1_5_d3_generation_v1/generation_plan_report.json`
- `outputs/pm_v1_5_d3_generation_v1_execution/execution_summary.json`
- `outputs/pm_v1_5_d3_blind_review_v1/manifest.json`
- `data/pm_v1_5_contracts/d3_minimum_stochastic_effect_training_v1.json`
