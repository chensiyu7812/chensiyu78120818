# MetaCom V3.3 EvoEmo Pairwise 评价失败报告

> **ARCHIVED / 只读单次事故报告。** 本文不是当前问题清单或执行路线。可迁移问题统一由
> `PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md` 维护；本文只保留旧 V3.3 pairwise gate 的
> 原始失败事实与证据。

更新时间：2026-06-30
状态：API 调用已完整完成，但 confirmatory judge gate 失败。该结果只能作为诊断，不能作为论文确认性外部结果。

## 1. 发生了什么

EvoEmo selective generation 已成功完成：

- `outputs/evoemo_selective/dialogues.jsonl`: 816 dialogues
- `outputs/evoemo_selective/turns.jsonl`: 8160 turns
- `outputs/evoemo_selective/artifact_attestation.json`: 已生成

随后运行 EvoEmo pairwise-only response evaluation：

- evaluation freeze: `outputs/evoemo_pairwise_eval_freeze.json`
- judge: GPT-4o
- pairs-only: true
- expected pairwise rows: 816
- actual pairwise rows: 816

但是最终没有生成：

- `outputs/evoemo_selective_metrics/selective_summary.json`
- `outputs/evoemo_selective_metrics/artifact_attestation.json`

原因是预设的 AB/BA orientation consistency gate 未通过。

## 2. 失败门禁

预设门槛：

```text
min_orientation_consistency = 0.80
```

实际结果：

| Comparison | Rows | Units | Orientation Consistency | Raw PM Score | Disagreement-as-tie PM Score |
|---|---:|---:|---:|---:|---:|
| pm_vs_best_fixed | 204 | 102 | 0.647 | 0.488 | 0.490 |
| pm_vs_full_history_rs | 204 | 102 | 0.745 | 0.853 | 0.853 |
| pm_vs_session_rag_rs | 204 | 102 | 0.402 | 0.461 | 0.466 |
| pm_vs_strong_rule | 204 | 102 | 0.569 | 0.458 | 0.461 |

全部 comparison 都低于 0.80，因此本轮 pairwise evaluation 被 fail-closed 拦截。

## 3. 费用与 token

实际 raw judge usage：

- usage rows: 816
- prompt tokens: 12,267,429
- completion tokens: 90,529
- 估算 GPT-4o 费用：约 31.57 USD，按 input 2.50 USD / MTok、output 10 USD / MTok 粗算

高费用的主要原因：

- 每次 pairwise prompt 比较 10 个 fixed-context cases；
- 每次包含 authorized ground truth；
- 每个 case 包含完整 `context_before_turn`、当前 seeker message、两个 candidate responses；
- 每个 comparison 又做 AB 与 BA 双顺序。

这不是无限循环或重复跑同一 key，而是当前评价协议本身成本高。

## 4. 诊断性读法

这批结果不能作为正式论文结论，但可作为诊断信号：

- PM 对 `full_history_rs` 的诊断优势明显；
- PM 与 `best_fixed` 接近；
- PM 与 `session_rag_rs` 接近或略低；
- PM 对 `strong_rule` 略低；
- judge 对候选位置敏感，尤其 `pm_vs_session_rag_rs` 的 AB/BA 一致性很低。

因此当前不能写：

> EvoEmo pairwise 证明 PM response quality 非劣或更优。

只能写内部记录：

> 当前 EvoEmo pairwise bundle judge protocol 未通过稳定性门禁，需要重设计评价协议。

## 5. 当前 artifact

已上传到 review package：

- `outputs/evoemo_selective_metrics/freeze_verification.json`
- `outputs/evoemo_selective_metrics/selective_dialogue_pairs.jsonl`
- `outputs/evoemo_selective_metrics/raw_selective_judge_calls.jsonl`
- `outputs/evoemo_selective_metrics/pairwise_gate_failure_diagnostic.json`

其中 `pairwise_gate_failure_diagnostic.json` 是从完整 816 行 pairwise judgments 离线汇总得到的失败诊断。

## 6. 下一版评价协议要求

下一版 API 评价必须先过 pilot，不允许直接全量跑：

1. no-API dry-run
   - 统计调用数；
   - 统计 token；
   - 估算费用；
   - 超过预算直接拒跑。

2. 小样本 pilot
   - 先跑少量 turns；
   - 检查 JSON 成功率；
   - 检查候选顺序稳定性；
   - 检查 position bias；
   - pilot 不过，不进入正式跑。

3. 正式评价改用更短、更稳的设计
   - 单 turn 为评价单位；
   - 同一 turn 内多候选打分，而不是 10-turn bundle pairwise；
   - 候选顺序固定 seed 随机并平衡；
   - 预注册抽样规模；
   - 保留 budget gate。

4. memory / strategy audit 禁止全量直接跑
   - 必须使用 stratified sampled audit；
   - memory omission audit 只在少量关键 turn 上使用完整 memory；
   - misuse audit 尽量只给 selected memory；
   - strategy audit 单独轻量 prompt。
