# PM V1.5 四组件边际效应执行定稿

> **2026-08-01 构念优先修订：** 本文原先把单次 paired response 的 material winner
> 直接作为 Step 1 PM gold。新证据表明这会把 routing、retrieval、execution、generator
> uptake 与主观单次质量偏好混成一个标签。当前最高优先合同改为
> `data/pm_v1_5_contracts/resource_routing_measurement_layers_v1.json`：Step 1 学习
> outcome-blind resource opportunity；retrieval、execution、grounded use 与 aggregate
> quality-risk-cost 分开评测。本文以下 paired-effect 设计仍是组件与整套系统的效果验证，
> 但不再逐条定义 Step 1 hard gold。适用/机会通过仍不自动等于系统有收益。

## 唯一主张

V1.5 的 PM 不是“资源适用性分类器”，也不是直接 16 分类器。它在候选发现后、
注入和生成前，分别估计 MP、MS、ME、RS 在当前状态、当前候选和其余组件背景下
带来实质回复收益的概率。四个二元决定组合成 16 个合法动作。

适用性、许可和候选安全只作前置硬门，不能产生 on gold。固定资源是否值得打开，
必须由同状态、同生成栈、仅改变一个组件 bit 的成对回复产生质量证据；component-on
实质胜且无 material risk 才是正标签。R0 胜、tie 或 risk 失败均为 off；无法确认
treatment 或无法裁定为 unknown。

正式机器可读合同为
`data/pm_v1_5_contracts/canonical_component_marginal_effect_v1.json`。

## RAG 在 V1.5 到此冻结

当前 40-core/80-card Bank、透明 Top-1 treatment、32 对 train-only RS effect 和
40 对 ESConv 冻结评测保留。它们已经证明 RS 有真实条件效应，也证明 family-only
模型不足。V1.5 不再通过新增 H2 或卡片小包把一个固定 treatment 修到“完美”。

`repair_v2` 是看过 H2v1 错例后的新 treatment；接入会改变已完成实验的候选，因此
只能作为未来版本诊断。现有 H2v2 页面还错误使用空 query，严禁人评或作为资格证据。

## 256 个 clean contrasts

每个 head 使用 64 个 component contrasts。对某一组件，另外三个 bit 构成 8 种
背景，每种背景 8 组；train/calibration/internal-test 分别为 4/2/2。每对只改变
被测组件一个 bit：

`response(state, background + component)` 对
`response(state, background)`。

这样既保留 16 动作，也能测量 MP/MS/ME/RS 在其他资源存在时的干扰，而不是假设四个
head 完全可加。

旧 7,488 action sweep 曾用于减少第一版调用，但 memory transport 审计后来确认其
每 source 固定两条的 backend 与外部构造不一致。用户选择强 transport 修复后，旧
192 个 memory contrasts 只保留为诊断；64 个 current-stack RS contrasts 继续正式
使用。正式 MP/MS/ME 必须在同 `build_evo_memory` 编译器重建的 backend 上重新生成。

## 唯一剩余人评

所有替代生成和 shortcut 检查完成后，只制作一个 256 对质量盲评包；20% 分层重复审核。
只有 component-on material wins 追加一次最小 atomic-risk 审核。不再追加 RAG
卡片、H2 或零散 confirmation 小包。

## 模型与通过标准

主模型为四个 L2 logistic heads，使用 user-grouped OOF 和 group-weighted binary
cross-entropy。输入必须是推理时可见的 state × candidate × background 信息；禁止
读取 outcome、judge、人评、regime、needed_memory_sources、split 或外部结果。

每个 head 的最低资格线为 balanced accuracy 0.70、positive recall 0.60、Brier
优于 prevalence prior、on/off 各至少 20%，并要求五个 seed 的 BA 标准差不超过
0.03。BGE-small 只可作为同一 state-candidate 标签上的 challenger，不再作为前置。

## 2026-07-30 实际执行状态

第一版四组件执行已经完成，但 memory 部分因 transport 修订而暂停：

- 旧 7,488 action sweep 通过了 R0 memory contrast 结构复用审计；
- 第一版冻结了 256 个独立 state 的 clean contrast 蓝图，四组件各 64 对、每个背景 8 对；
- 512 个 response arms 中，256 个旧 R0 arm 合格复用，256 个 current-stack arm
  已用同一 `meta/llama-3.1-8b-instruct` supporter treatment 完成生成；
- 生成执行为 256/256 complete，所有 effect label 在生成阶段保持 unknown；
- 已生成 256 个独立质量题加 52 个冻结重复题的盲评包，共 308 次呈现，但该页面已经
  明确标记暂停，不再审核；
- 盲评公开面不含 component、split、control/treatment、action、memory、strategy、
  retrieval score 或生成来源。

当前不应完成旧 308 次盲评。正式 adapter/backend 已落地为 416 个同编译器、严格过去
时状态；替代蓝图已冻结 192 个 MP/MS/ME pairs，并保留原 64 个 RS pairs。384 个
arms 的冻结计划与可恢复 runner 已准备完毕、尚未调用 API。当前下一步是执行该计划，
再制作唯一 replacement 256-pair packet。

## 2026-07-30 个性化 memory transport 修订

“同一套 RS Bank”不能机械地解释成“所有数据集使用相同 memory 文本”。个性化记忆必须
来自各用户自己的既往历史；真正需要统一的是 memory compiler 的语义、retriever、
filter、candidate descriptor、注入和生成合同。

现有内部 catalog 每个 source 固定两条，而 EvoEmo 的 MP/MS/ME 中位 catalog 数量为
7/22/68；内部与外部 ME item 中位长度为 37/102 tokens。这对强外部泛化主张构成
高严重度 transport mismatch。弱主张下旧 pairs 仍可描述给定小 catalog 的候选效应，
但强 transport 修复下只保留 64 个 RS formal；旧 192 个 memory pairs 降为诊断，
不能证明 PM 会从任意大型 catalog 中找对具体记忆。

正式 PM 输入因此必须落实为 `state × realized candidate descriptor × background`，
而不是 source-only 或全 catalog count。外部 candidate 超出 train/calibration 冻结支持
时必须 abstain/off。详见 `docs/PM_V1_5_MEMORY_TRANSPORT_REPAIR_ZH.md` 和
`data/pm_v1_5_contracts/memory_transport_candidate_contract_v1.json`。

已完成的强修复证据：

- 416 个状态，train/calibration/internal=`192/96/128`；
- MP/MS/ME catalog 中位数 `3/4.5/5`，三源均 416 个状态有非空自然候选；
- 严格 user-group split 零重叠、严格只用 prior sessions、零 EvoEmo 内容/outcome；
- 当前具体 memory item 的正式 ranker 是 lexical score，Top-k=`2/2/3`；BGE 仅提供
  source/candidate-state similarity challenger，不谎称 cosine 在做正式 item retrieval；
- 新 192 memory contrasts 各 head 64、各 background 8、split=`96/48/48`，需 384
  个新 response calls；冻结计划预检通过、预计生成费用上界约 0.104 美元；加保留的
  64 RS 后最终仍为 256 对。
