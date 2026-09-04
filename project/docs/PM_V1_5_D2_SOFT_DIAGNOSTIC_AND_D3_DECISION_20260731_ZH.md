# PM V1.5 D2 软目标诊断与最小 D3 决策

状态：`D2 SOFT DIAGNOSTIC COMPLETE / D3 DESIGN FROZEN / D3 DATA NOT BUILT`

## 这次实际推进了什么

两份 D2 人评已按正确统计粒度合并：

- 96 个独立生成 pair；
- 32 个 state，每组件 8 个；
- 每个 state 恰好 3 个独立 pair；
- 同一 pair 有两位评审时先在 pair 内平均，不能把两位评审冒充两次生成；
- treatment-win=1，control-win/tie=0；
- 每个 state 再对三个 pair 等权平均，得到 soft expected-benefit；
- D2 仍是严格 FAIL，soft target 只作恢复诊断，不是新 hard gold。

32 个 state 的 soft target 分布为：13 个 `0`、7 个 `1`，其余 12 个位于两者之间。
这说明资源效应既没有退化成全关，也没有退化成全开；同时约三分之一 state 的结果会随
生成或评审变化，正是不能继续用单次胜负冒充确定真值的原因。

## 候选相似度本身够不够

每组件只有 8 个独立用户，按 `floor(groups/5)` 容量规则最多只能检查一个特征。本轮固定
使用运行时已有的自然 Top-1 词法候选匹配分数，不试 BGE、不换模型、不搜特征。

| 组件 | soft target 均值 | Spearman rho | 留一用户 MSE 相对均值增益 | 双诊断同为正向 |
|---|---:|---:|---:|---|
| RS | .333 | .411 | +.0181 | 是 |
| MP | .479 | -.074 | -.0798 | 否 |
| MS | .208 | .013 | -.0090 | 否 |
| ME | .562 | .356 | -.0812 | 否 |

结论不是“记忆无效”，而是“候选相似度不足以预测记忆注入收益”。ME 的排序相关为正，
但留一用户误差更差，说明 8 个点下关系不稳定；MP/MS 连方向都没有。RS 则第一次表现出
透明候选分数与软收益的一致正向关系，但 8 个 state 仍不能晋级模型。

这也解释了为什么现在再换 BAAI 不对路：当前缺的不是另一种相似度，而是候选是否有据、
是否已经在当前对话出现、是否冲突或陈旧，以及另外三个组件到底谁已经开启。BGE 只能
改变“像不像”，不能自动补齐这些变量。

## 最小 D3

D3 不再沿用“128 个单 pair 就是 128 个 state gold”的说法。冻结方案为：

- 32 位内容不重叠的新开发用户；
- 每位用户分别贡献 RS、MP、MS、ME 一个 contrast，共 128 个 state；
- 每个 state 一个 primary pair；在 outcome 前固定 32 个 state 再加一个 repeat pair；
- 总计 160 pair、320 次 response API 调用、160 个质量盲评决定；
- 每个 state 的总训练权重固定为 1，repeat 只改进 soft target，不扩大 ESS；
- D2 只作测量证据，不进入 D3 训练；
- 不再做一轮完整第二人复核，论文直接披露 D2 测得的二分类 inter-rater `.8125`；
- D3 训练使用 state 等权的 fractional binary cross-entropy 加 L2，而不是把每条当确定 gold。

32 个用户使每个 head 的容量上限为 6 个特征，恰好容纳：

- RS/MS/ME：match、grounding/nonredundancy、三个具体 background bits，共 5 个；
- MP：再加 `candidate_is_preference`，共 6 个。

MP 固定为 16 个 preference、16 个 profile；MS 只允许严格过去、具有 supplied summary 的
`MS_SESSION`；ME 加入 conflict/stale/wrong-person 的硬门负对照；RS 保持同一冻结 Bank、
自然 Top-1、最多注入一张卡。

## 接下来唯一正确的工程顺序

1. 实现 grounding/nonredundancy 和三个显式 background bits；
2. 先测试特征无 outcome、无 ID、无外部结果、无 topic/template 捷径；
3. 再构造 32 个新用户与 128 个 contrast；
4. 所有覆盖和容量预检通过后才允许生成 320 次 response；
5. 完成最后一份有上限的 160-pair 质量盲评；
6. 四个 L2 logistic head 只训练一次；
7. 哪个 head 过 `.70 BA` 等全部门，哪个才进入新用户 internal confirmation；
8. 冻结后才跑 ESConv/EvoEmo，外部结果不得反向修模型。

因此下一步已经不再是 RAG、人评或 encoder 岔路，而是一个明确的 D3 Step 0：把两个缺失
的透明候选特征做实并通过 outcome-blind preflight。

机器结果：

- `outputs/pm_v1_5_d2_measurement_review_analysis_v1/state_soft_targets.jsonl`
- `outputs/pm_v1_5_d2_soft_target_diagnostic_v1/diagnostic.json`
- `data/pm_v1_5_contracts/d3_minimum_stochastic_effect_training_v1.json`

