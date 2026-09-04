# PM-v1.5 快速会议版：GPT Pro 审查入口

更新时间：2026-07-17

审查对象：当前分支中的 **PM-v1.5 快速会议版**。这是一个尚未产出正式实验结果的
方法与执行链审查包，不是结果仓库，也不是论文结论已经成立的证明。

## 0. 首先避免版本混淆

本仓库曾把 `pm-v1.5-supplemental` 用来表示“冻结 PM-v1 后、不重训的 post-hoc
补充诊断”。那套历史材料仍保留用于解释 PM-v1 为什么失败，但它不是本次审查对象。

当前 PM-v1.5 的定义不同：

> 基于 PM-v2.2 的 fail-closed 基础设施，使用干净且一致的训练/测试机制重新生成开发
> 数据、重新训练 Policy Manager、重新冻结并重新做外部评测的快速会议版；它明确省略
> 人工评测与 V2.2 的部分昂贵复核层。

因此：

- 旧 `PM_V1_5_SUPPLEMENTAL_ANALYSIS_ZH.md` 只能视为 legacy PM-v1 诊断；
- 旧 V1/V1.5 数字不能成为当前 PM-v1.5 的训练依据或正式结果；
- 当前 PM-v1.5 只有在本文件所述完整链路执行并通过后，才可能支持其条件性主张；
- 当前代码状态下还没有训练好的 PM-v1.5 checkpoint、study freeze 或外部主结果。

## 1. 研究目的

研究问题不是“情感支持是否在临床上有效”，而是一个系统与资源分配问题：

> 一个检索前 Policy Manager 能否根据当前支持场景，选择是否以及调用哪些长期记忆来源
> 和 Strategy RAG，并在外部固定输入评测中维持回复质量，同时减少生成器实际接收的输入
> tokens？

动作空间为 8 个记忆子集 × 2 个策略模式，共 16 个动作。PM 只能看到检索前状态特征，
不能看到实际 memory 文本、检索 item ID、生成回复或 judge label。

## 2. 唯一允许的主要外部主张

主张不是预先保证成立；代码只能在下列两个冻结条件同时满足时输出 `SUPPORTED`：

1. 相对 `best_fixed`，即固定 `MPMSME+RS` structured-high-resource comparator，
   PM-v1.5 的六维 `quality_composite` 配对差值按 `user_id` 聚类 bootstrap 的 95% CI
   下界不低于 `-0.02`（归一化 [0,1] 尺度，约等于原始 1–5 量表的 0.08 分）；
2. 同一比较中，PM-v1.5 的 `observed_input_tokens` 配对差值 95% CI 上界严格小于 0。

任一条件不满足，正式输出必须是 `NOT_SUPPORTED`。外部流程跑通、平均值方向有利或
某个 judge 给出高分，都不能替代这个判定。

这里有一个需要审查者重点判断的方法选择：主要 comparator 是高资源固定策略，而不是
calibration 选出的 same-token `cost_matched_fixed`。后者仍是七条件外部矩阵和内部路由
诊断的重要 baseline，但当前冻结代码没有把“PM 外部优于 same-token fixed policy”设为
主要论文判定条件。请审查这是否足以支撑论文想表达的资源路由价值。

## 3. “回复质量”的定义

主分析不请求 LLM `overall` 字段，也不只使用 emotional support 与 personalization。
它冻结为六个维度的确定性加权 composite：

| 维度 | 权重 |
|---|---:|
| emotional support | 0.30 |
| personalization | 0.20 |
| memory appropriateness | 0.15 |
| factual grounding | 0.15 |
| temporal consistency | 0.10 |
| non-intrusiveness | 0.10 |

GPT-4o 对 192 个正式 unit 做主质量判断；Claude 只对预冻结的 54-unit 分层样本做
敏感性分析，不与 GPT 分数混合，也不决定主要 verdict。七个匿名 condition 在同一个
batched prompt 中评分并做位置平衡；这降低调用成本，但无法完全消除候选间 contrast
bias。

## 4. 路由层证据

内部 `internal_test` 只在训练和 calibration 完成并冻结后查看一次。报告比较 PM、
same-token cost-matched fixed policy 与 16-action oracle，包括：

- quality regret 与 routing regret；
- memory / Strategy RAG 开关判断；
- quality-acceptable rate；
- excess observed cost。

这些只能描述“在本研究 synthetic 状态与模型 judge 标签下的决策质量”，不能写成
人工确认的资源正确性、真实用户获益或临床改善。

## 5. Risk 与 latency 的严格边界

- Risk 不进入主要质量—成本 verdict。两家 judge 只对 36-unit 预冻结分层样本检查
  evidence/resource-use problems。只有用户聚类 CI 上界不超过 [0,1] 风险 margin 0.05
  时，才允许写“预注册分层审计未发现 PM 增加证据误用问题”。这不是总体安全性、
  临床风险、人口风险或部署风险结论。
- Latency 没有 interleaved confirmatory design，只能报告同次运行中真实记录的
  mean/median/P95 和配对点差。代码硬编码为 `DESCRIPTIVE_ONLY_NOT_CONFIRMATORY`，
  不能写“显著降低延迟”。

## 6. 明确不主张

本研究不主张：

- 临床疗效、心理健康改善、危机干预能力或真实部署安全性；
- 当前 PM 已经学会人类意义上的“正确资源调用”；
- 对所有用户、数据集、语言、模型或数字人系统泛化；
- 全量人工验证或全量双 judge-family 复核；
- latency 的确认性优势；
- PM 外部显著优于所有 baseline；
- dry-run、transport pilot、旧 PM-v1 诊断或脚手架测试是正式实验结果。

## 7. 相比旧 PM-v1，当前链路试图修复什么

1. 训练、sweep、外部生成使用同一 supporter treatment、同一 Strategy Bank 和同一
   retrieval/Evidence Filter 机制；V1.5 全链路关闭 Evidence Filter，不再训练/测试错配。
2. Strategy Bank 额外使用 `escN -> esconv_N` 确定性来源映射排除 EvoEmo 重合：
   84 个 ESConv 来源被排除，bank 从 12,429 降为 12,403 cards。
3. train / calibration / internal_test 用户与 semantic family 分离；超参数只由 calibration
   选择，internal_test 冻结后只查看一次。
4. 每个 468-state development matrix 必须覆盖全部 16 actions，不能用筛选后的 pilot
   冒充完整 sweep。
5. 外部七条件使用相同 fixed seeker tracks、相同 supporter treatment 和内容寻址的
   policy/freeze lineage。
6. 付费阶段均先 dry-run、接受精确 cost hash，再以 append-only physical-attempt ledger
   执行；确定性失败跨进程不可偷偷重试。

## 8. 截至当前提交的真实执行状态

| 阶段 | 状态 | 能否视为论文结果 |
|---|---|---|
| clean Strategy Bank / overlap audit | 已完成 | 只能证明数据血缘与已知 overlap 处理 |
| 1-call generation compatibility pilot | 曾结构性 PASS，但绑定旧 config hash | 否；当前 config 下必须重跑 |
| automated semantic review | 当前 66-call dry-run 已完成；三次旧执行因传输故障中止，没有 PASS gate report | 否 |
| 52-user development generation | 未执行 | 否 |
| 468 × 16 full action sweep / judging | 未执行 | 否 |
| PM-v1.5 training / internal_test | 未执行 | 否 |
| fixed seeker / study freeze | 未执行 | 否 |
| seven-condition external generation / judging | 未执行 | 否 |

特别说明：历史 1-call pilot 的 attestation 绑定
`pm_v1_5.yaml` SHA-256 `372c95dd...579156`，当前 config 为
`81c1d120...07762`。虽然改动主要发生在后续外部合同，现有 fail-closed 代码仍会把它
视为 lineage mismatch；因此当前入口不把该 pilot 写成仍然有效的上游 PASS gate。

## 9. 建议 GPT Pro 优先审查的问题

1. 主张是否与 comparator 一致：对高资源 fixed policy 的“质量非劣 + tokens 更少”是否
   足以证明 Policy Manager 的价值，还是必须增加对 same-token fixed policy 的外部优势
   判定？
2. 没有人评时，六维 LLM composite、GPT 主分析加 Claude 分层敏感性是否足够支撑当前
   非临床、非真实改善的有限主张？
3. 七候选 batched judging 的位置平衡和 order pilot 是否足以控制 contrast/context bias？
4. synthetic development、EvoEmo development-informed external evaluation 和
   Strategy Bank 的数据边界是否仍存在可利用的泄漏或隐性调参通道？
5. internal_test 是否在所有代码路径中真正只查看一次，且 freeze 后没有再反向调参？
6. V1.5 wrappers 是否在训练、生成、baseline 与外部评测间保持同一 generation/retrieval
   contract，尤其是 Strategy RAG 与关闭的 Evidence Filter？
7. forced-swap canary 是否只承担 judge sensitivity，且没有被错误解释成 PM efficacy？
8. 当前 risk 与 latency 的论文措辞是否严格受代码合同限制？
9. 传输层有限重试是否可能因重启、缺字段或 schema 失败产生选择性重试？
10. 在没有正式结果时，是否还有任何代码或文档提前暗示 `SUPPORTED`？

## 10. 推荐阅读顺序

1. 本文件；
2. `project/docs/PM_V1_5_CORE_CHAIN_PLAN_ZH.md`；
3. `project/docs/PM_V1_5_REVIEW_ARTIFACT_INDEX.json`；
4. `project/configs/pm_v1_5.yaml`；
5. `project/src/metacom_pm/v1_5_external_claims.py`；
6. `project/src/metacom_pm/v1_5_external_batched.py`；
7. `project/scripts/v1_5_create_freeze.py`；
8. `project/scripts/v1_5/` 与 `project/scripts/v1_5_run_automated_semantic_review.py`；
9. `project/tests/test_v1_5_*.py`、`test_bounded_retry.py` 和
   `test_api_retry_classification.py`；
10. 历史问题背景：`project/docs/PM_V1_FAILURE_LIMITATION_POSTMORTEM_ZH.md` 与 legacy
    `PM_V1_5_SUPPLEMENTAL_ANALYSIS_ZH.md`，但不要把其中结果当作当前 V1.5 结果。

## 11. 可直接交给 GPT Pro 的审查任务

> 请把当前 PM-v1.5 当作“尚未执行正式实验的预注册实现”审查。逐项核对研究目的、主要
> comparator、claim 判定、数据分割、bank overlap、训练/测试 mechanism parity、freeze、
> external batched judging、risk/latency 边界和重试记账。请区分代码完整性、设计有效性与
> 结果证据；不要把 dry-run、transport pilot 或 legacy PM-v1/V1.5 supplemental 数字视为
> 当前结果。最后给出：P0/P1/P2 问题、必须在付费执行前修复的项目、可以留作 limitation
> 的项目，以及当前主张在什么条件下才站得住。
