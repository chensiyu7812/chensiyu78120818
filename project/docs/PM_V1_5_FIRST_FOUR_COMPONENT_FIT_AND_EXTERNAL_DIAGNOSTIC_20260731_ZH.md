# PM V1.5 四组件首次正式训练与外部诊断结论

日期：2026-07-31  
状态：`FIRST_FIT_COMPLETE / NO_HEAD_PROMOTED / CURRENT_LEARNED_PM_NOT_USABLE`

## 1. 结论

当前这版 V1.5 已经解决了“能不能形成不作弊、同一资源栈、可解释的四组件训练
数据”问题，但还没有解决“PM 能不能从这些数据稳定学会开关”问题。

因此现在不能声称：

- PM 已学会 RS、MP、MS 或 ME 的开关；
- BAAI 已解决语义判断；
- 当前 learned PM 可以进入正式 EvoEmo 质量实验；
- internal test 中某两个较高点估计证明了泛化。

可以声称并有实际 artifact 支持的是：

1. 四个组件各有 64 个同状态 one-bit contrasts，共 256 个独立语义对；
2. 四组件使用同一 repaired longitudinal memory/strategy stack；
3. 人工质量裁决完成，52/52 位置反转复测一致；
4. 109 个 component-on quality wins 完成 interaction-and-grounding risk proxy，
   106 个安全采用、3 个因 material risk 改为 off；
5. 最终标签为 106 on、150 off、0 unknown；
6. user-grouped OOF、冻结 internal test、透明 baselines、shortcut diagnostics 和
   BGE-small challenger 均已执行；
7. 结果诚实地显示四个 learned heads 都未达到预冻结“及格”门。

## 2. 第一轮训练结果

主模型是四个互相独立的 L2 logistic heads。输入只含当前可见状态、已发现但尚未注入的
候选资源描述和其他组件背景，不含回复、judge、质量标签、风险结果、用户 ID、数据集
身份或未来信息。

| Head | development grouped-OOF BA | Brier gain vs prior | positive recall | seed BA std | internal-test BA | 结论 |
|---|---:|---:|---:|---:|---:|---|
| RS | 0.6223 | +0.0297 | 0.4030 | 0.0544 | 0.4272 | 未过 |
| MP | 0.5027 | +0.0102 | 0.6417 | 0.0381 | 0.7250 | 未过 |
| MS | 0.4678 | -0.0078 | 0.2466 | 0.0351 | 0.7308 | 未过 |
| ME | 0.4452 | -0.0360 | 0.1351 | 0.0305 | 0.6126 | 未过 |

冻结门要求 BA≥0.70、positive recall≥0.60、Brier 胜 prevalence prior、on/off 各至少
20%、五 seed BA std≤0.03。四个 head 没有一个同时满足。

MP/MS 的 16 条 internal-test BA 看起来较高，但开发 OOF 接近或低于随机，而且每个
internal head 只有 16 条。它们只能视为小样本波动，不能越过模型选择门。

## 3. BAAI 的实际结论

按合同只运行一个预注册 challenger：`BAAI/bge-small-en-v1.5`。

| Head | BGE development OOF BA | BGE Brier gain | internal BA | 是否晋级 |
|---|---:|---:|---:|---|
| RS | 0.6423 | +0.0208 | 0.3475 | 否 |
| MP | 0.5254 | -0.00003 | 0.7750 | 否 |
| MS | 0.5719 | +0.0019 | 0.6581 | 否 |
| ME | 0.4815 | -0.0112 | 0.6126 | 否 |

BGE 对 RS/MS 有一点点点估计改善，但没有达到 0.70，稳定性不合格，而且没有形成
development 与 internal 一致的提升。结论不是“BAAI 很差”，而是：

> 通用 embedding 能表达文本或候选的语义相似性，却不能从当前每状态一次生成、
> 一次人评所得的边际回复胜负中恢复稳定开关规律。

继续在同一 256 组上换 BGE-M3、Qwen、PCA 维数、C、threshold 或 seed，会变成模型
赛马和结果后选择，不再是可信修复。

## 4. 已执行的外部诊断

### 4.1 ESConv

失败的冻结 RS head 被原样投到既有 40-dialogue family-balanced ESConv panel。该结果
只作 posthoc descriptive diagnostic，不能晋级模型。

- feature support：31/40；
- balanced accuracy：0.4583；
- positive recall：0.25；
- Brier：0.2983，差于冻结 development prior 的 0.2400；
- 所选回复：17 个 material winner、16 个 loser、7 个 tie；
- 相对 always-off 多 1,489 tokens；
- 相对 always-on 少 3,445 tokens。

这表明它确实减少了一部分 RS 调用和成本，但没有可靠地把资源保留在更可能获益的
状态上，当前不能用。

### 4.2 EvoEmo

没有读取任何 EvoEmo 回复质量、风险或 future outcome，也没有 API 调用。只把冻结
memory heads 投到 204 个外部状态，检查候选存在、development-range support、开关率和
预计注入 token。

| Head | 有候选状态 | 候选落入该 head development support | predicted on |
|---|---:|---:|---:|
| MP | 114/204 | 0/114 = 0% | 0 |
| MS | 204/204 | 177/204 = 86.8% | 28 |
| ME | 204/204 | 122/204 = 59.8% | 0 |

完整付费 EvoEmo 质量外测现在只会测一个 MP/ME 退化关机 policy，不能回答“PM 是否学会
quality-risk-cost 平衡”。因此没有执行付费调用，避免把已知不合格模型的结果误写成
正式泛化证据。

## 5. 为什么总体 transport audit 通过，实际 head 却 OOD

此前的 shared-compiler audit 检查的是全部 416 个可用内部纵向状态。正式训练实际只从
中选择每 head 64 条，其中 development 只有 48 条。二者不是同一个支持集合。

零 outcome 复审得到：

| Head | 完整 416 内部状态对外部 exact candidate pattern 覆盖 | 48 条 labeled development 覆盖 | labeled development 联合范围支持 |
|---|---:|---:|---:|
| MP | 70.2% | 0% | 0% |
| MS | 62.3% | 30.4% | 86.8% |
| ME | 48.5% | 22.1% | 59.8% |

MP 最直观：外部 candidate token bucket 全部为 1，并常见 retrieved fraction=0.5；
当前 MP development 全部是 token bucket=2、retrieved fraction=1.0。因此 shared
compiler 没坏，是产生 labels 的蓝图抽样漏掉了外部会出现的输入格子。

此外，小样本仍很稀疏：

- MP 的 48 条 development 只有 27 个完整特征 pattern，其中 8 个 pattern 同时出现
  on/off 冲突；
- MS/ME 各有 44 个 pattern，其中 40 个都是 singleton；
- 这说明当前每 head 48 条对于 8–9 个特征过于稀疏，模型几乎没有机会从同类状态中
  学到可重复频率。

## 6. 当前失败的科学含义

这次结果排除了几个旧的模糊解释：

- 不是因为没有四组件数据：256 对已经完整；
- 不是因为风险标签缺失：109 个 on-win 已审核；
- 不是因为 train/test 用户泄漏：group split 已强制零重叠；
- 不是因为仍然使用两套 memory compiler：13/13 机制合同已通过；
- 不是因为只缺一个 BAAI embedding：注册 challenger 也未过门；
- 不是因为 RAG 从来无效：RS direct-effect 数据中确实同时存在 win/off/tie。

当前最有证据的两个瓶颈是：

1. **训练子集选择失败。** 完整内部池有可迁移候选，但 labeled subset 没覆盖；
2. **单次 realized win 标签可能噪声过大。** 每个状态只生成一次 R0/on 回复，标签混合了
   状态、候选资源真实效应和 generator sampling variation。PM 要预测的是期望收益，
   一次抽样却只提供一个高方差 Bernoulli 观测。

第二点目前是有力假设，不是已经证明的结论，因为尚未对同一状态重复生成独立 pairs。

## 7. 下一步：一次有界修复，不再回到无限 RAG 人评

### R1：冻结本轮失败

当前 256 对、primary、BGE、ESConv 和 EvoEmo diagnostics 全部保留，不能删掉、换 seed
或只挑有利 head。它们是 V1.5 的 first-fit baseline。

### R2：先做小型标签可重复性试验

从四个 head 各取 8 个状态，共 32 个：

- 一半来自当前 on、一半来自 off；
- memory head 同时覆盖外部常见 candidate covariate strata；
- 每状态额外生成两组独立的同栈 control/treatment pairs；
- 只做 64 个新的质量盲评决定；仅当 component-on 胜且质量复现时，再审很少量 risk。

预先冻结判定：

- 若原标签与两次重复的多数方向一致率低于 70%，停止训练单次硬标签；
- 改用三次重复估计 expected material-benefit probability，或把该局限明确留给 V2；
- 若一致率达到 70%，说明主要是覆盖/容量问题，可以进入 R3。

这一步比再做几百条 Bank 人评更直接，因为它检验“PM 要学的目标本身是否稳定”。

### R3：只在 R2 通过后重建 coverage-aware blueprint

不读取 external outcome，只用已经允许的 outcome-free candidate covariates，从 416 个内部
状态分层选择新 train/calibration：

- MP 必须覆盖 token bucket 1/2、retrieved fraction 0.5/1.0、relevance 1–4；
- MS/ME 按 relevance、margin、token、relative-age 主要格子分层；
- 每 head 特征压缩到 3–5 个预先解释清楚的量，避免 48 条配 8–9 特征；
- 仍用 user-grouped OOF、固定阈值和 independent internal/external groups；
- 所有四 head 都尝试，只有通过哪个才主张哪个，不能把未过的 head 删除。

### R4：模型通过后才运行正式外部质量实验

顺序固定为：

`grouped-OOF pass → frozen internal check → outcome-blind external support pass →
ESConv/EvoEmo response generation → blind quality → minimal risk → cost/utility`

外部数据不能救回 development 失败模型，也不能用于调 threshold 或选择新训练样本。

## 8. 对论文的现实判断

当前数据足以写“资源直接效应、风险代理和 learned gating 的一次严格失败”，但不足以写
“PM 已学会四组件 quality-risk-cost 平衡”。

最快仍有机会得到正向 V1.5 结论的路线，不是继续修 RAG 或换 embedding，而是用 R2
回答一个更根本的问题：

> 同一个 state × candidate 的边际收益标签，是否在独立生成下足够稳定，值得让小模型
> 学习？

如果答案是肯定的，再做一次 coverage-aware 小规模补充；如果是否定的，就应停止把
generator 随机性当 PM gold，并把“期望效应需要重复 treatment”作为 V1.5 的明确结论和
V2 的方法改进。

## 9. 关键 artifact

- 最终标签：
  `outputs/pm_v1_5_transport_repaired_final_effect_labels_v1/component_effect_labels.jsonl`
- primary：
  `outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1/training_report.json`
- BGE challenger：
  `outputs/pm_v1_5_transport_repaired_bge_challenger_v1/bge_challenger_report.json`
- ESConv diagnostic：
  `outputs/pm_v1_5_new_rs_head_esconv_frozen_diagnostic_v1/external_diagnostic_report.json`
- EvoEmo outcome-blind projection：
  `outputs/pm_v1_5_frozen_memory_heads_evoemo_diagnostic_v1/external_support_report.json`
- labeled-subset coverage audit：
  `outputs/pm_v1_5_component_label_coverage_audit_v1/coverage_audit.json`

