# MetaCom V3.3 Cost-matched Fixed Baseline 诊断

更新时间：2026-07-14

这份文件记录 `selection_stable.json` 中 `budget_matched_fixed_action` 在 EvoEmo external states 上的 no-API 成本诊断。诊断不重训模型、不调用 generator / judge API，只估计固定动作在现有 EvoEmo fixed tracks 上的检索后 prompt 成本，并与 PM 的已观测 generation logs 对齐。

复现入口：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/tokkio/miniconda3/envs/sim_eval/bin/python \
  scripts/17p_analyze_cost_matched_fixed_baseline.py
```

输出：

- `outputs/evoemo_response_v4/cost_matched_fixed_baseline_diagnostic.json`
- `outputs/evoemo_response_v4/cost_matched_fixed_baseline_diagnostic.md`

## 1. 诊断回答什么问题

该诊断用于回答：

> 是否存在一个和 PM token 成本接近的 fixed-action baseline？

这是当前最关键的剩余对照之一。因为 PM 在 EvoEmo 上的 `epsilon=0` selection rule 使 cost tie-breaker 很少发挥作用，审稿人自然会问：

> 如果固定选择一个和 PM 成本相近的动作，是否也能达到类似回复支持评分？

本诊断只回答成本是否匹配；不回答回复质量。

## 2. 主要结果

`selection_stable.json` 中的 budget-matched fixed action 是：

```text
ME+R0
```

成本诊断结果：

| Scope | Method | n | Observed total tokens | Local component tokens | Memory tokens | Strategy tokens | Selected memories |
|---|---|---:|---:|---:|---:|---:|---:|
| All 1020 turns | PM observed | 1020 | 1291.8 | 1335.7 | 677.3 | 250.7 | 4.16 |
| All 1020 turns | ME+R0 local estimate | 1020 | n/a | 1342.3 | 934.6 | 0.0 | 3.00 |
| V4 204 sample | PM observed | 204 | 1294.6 | 1340.5 | 683.3 | 245.4 | 4.20 |
| V4 204 sample | ME+R0 local estimate | 204 | n/a | 1346.5 | 934.7 | 0.0 | 3.00 |

用 PM observed/local ratio 校准后：

| Scope | PM observed tokens | ME+R0 calibrated observed tokens |
|---|---:|---:|
| All 1020 turns | 1291.8 | 1298.2 |
| V4 204 sample | 1294.6 | 1302.3 |

结论：

> `ME+R0` 在 EvoEmo 上确实是一个接近 PM token 成本的 fixed-action baseline 候选。

## 3. 对下一步实验的影响

如果要检验 learned routing 的外部增益，当前最值得补的 API baseline 是：

```text
ME+R0 on the V4 204-unit sample
```

已新增专用脚本：

- generation: `scripts/17q_run_cost_matched_fixed_baseline.py`
- response scoring: `scripts/17r_eval_cost_matched_fixed_response.py`

最小实验链路：

1. 生成 `ME+R0` 在 V4 sample 上的回复；
2. 使用同一 V4 pointwise judge schema 评分；
3. 与 PM 做 paired delta / bootstrap CI；
4. 报告 `Support Rating`、子维度、input tokens。

可能解释：

- 如果 `ME+R0` 支持评分接近 PM，则 learned routing 的外部增益需要降级表述，PM 更像一种低成本可部署 router，而不是外部质量优势明显的 adaptive policy。
- 如果 PM明显优于 `ME+R0`，则可以说明即使当前 abstention / strategy-off calibration 不完善，状态条件化 routing 仍优于同预算常量动作。

## 4. API 运行记录

Generation 已完成：

| Item | Value |
|---|---:|
| condition | `cost_matched_me_r0` |
| action | `ME+R0` |
| completed turns | 204 / 204 |
| estimated cost | 0.929 USD |
| mean input tokens | 1420.9 |
| max input tokens | 1796 |
| cost estimate sha256 | `bd103c93315730f8e6ed2b72346047477db35e17f6976be7f3815d5ae5b45b6d` |

Response scoring 已完成：

| Item | Value |
|---|---:|
| conditions | `pm`, `cost_matched_me_r0` |
| expected calls | 204 |
| successful calls | 204 |
| estimated cost | 4.329 USD |
| mean input tokens | 6687.9 |
| p95 input tokens | 10480 |
| max input tokens | 10933 |
| order variants | `0` |
| cost estimate sha256 | `1eaf647f6442f9dfa991e1432aaeb799b319d9d3a477cbe17e38b97591bbe44a` |

## 5. Response scoring 结果

`ME+R0` generation 和 GPT-4o V4 pointwise scoring 已完成。

输出：

- generation: `outputs/evoemo_cost_matched_me_r0_generation/`
- response scoring: `outputs/evoemo_cost_matched_me_r0_response/`
- bootstrap summary: `outputs/evoemo_cost_matched_me_r0_response/paired_bootstrap_summary.json`

完整性检查：

| Check | Value |
|---|---:|
| generated turns | 204 / 204 |
| response judgment calls | 204 / 204 |
| score rows | 408 / 408 |
| raw rows | 204 |
| failures | 0 |

实际 generation 成本：

| Method | n | Mean input tokens | Median input tokens | Mean output tokens | Mean latency |
|---|---:|---:|---:|---:|---:|
| PM | 204 | 1294.6 | 1537.5 | 68.1 | 1.92s |
| ME+R0 | 204 | 1283.8 | 1287.0 | 77.8 | 1.21s |

因此，`ME+R0` 是真正的同预算固定动作对照。

Response scoring summary:

| Method | n | Support Rating ↑ | Personalization ↑ | Memory Approp. ↑ | Factual ↑ | Temporal ↑ | Non Intrusive ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| PM | 204 | 3.755 | 3.495 | 4.255 | 4.500 | 4.525 | 4.480 |
| ME+R0 | 204 | 3.926 | 3.696 | 4.289 | 4.505 | 4.529 | 4.451 |

Paired PM minus ME+R0 deltas:

| Metric | PM - ME+R0 mean | 95% cluster bootstrap CI | P(delta < 0) | W/T/L |
|---|---:|---:|---:|---:|
| Support Rating | -0.172 | [-0.250, -0.093] | 1.000 | 22/126/56 |
| Personalization | -0.201 | [-0.275, -0.123] | 1.000 | 16/132/56 |
| Memory appropriateness | -0.034 | [-0.069, 0.000] | 0.969 | 3/191/10 |
| Factual grounding | -0.005 | [-0.025, 0.015] | 0.585 | 3/197/4 |
| Temporal consistency | -0.005 | [-0.025, 0.010] | 0.607 | 2/199/3 |
| Non-intrusiveness | 0.029 | [-0.029, 0.088] | 0.149 | 22/166/16 |

解释：

- `ME+R0` 在同预算下显著高于 PM 的 `Support Rating` 和 personalization；
- factual grounding、temporal consistency、non-intrusiveness 基本相近；
- 这说明当前 supervised PM 的外部 learned routing advantage 没有被该 V4 结果支持；
- 负面结果主要集中在支持性表达和个性化，而不是事实性或时间一致性。

## 6. 对论文主张的影响

这个结果必须进入论文边界。当前最稳结论应改为：

> The supervised PM provides a deployable source-level routing mechanism and reduces cost relative to high-resource baselines, but it does not outperform a cost-matched fixed action on EvoEmo V4. This suggests that external resource routing remains under-calibrated and that fixed medium-resource policies can be strong baselines.

中文表述：

> 当前监督式 PM 相对全历史和高资源结构化基线能降低成本，但在 EvoEmo V4 上没有超过同预算固定动作 `ME+R0`。这说明当前 PM 的外部 learned routing advantage 尚不成立，资源 abstention、strategy-off 和固定预算内的动作校准仍是核心局限。

可以写：

- `ME+R0` is a strong cost-matched fixed-action baseline.
- The current supervised PM does not beat this cost-matched fixed baseline on support rating.
- This strengthens the diagnosis that coarse supervised routing is insufficient for robust external calibration.

不应写：

- PM outperforms all same-budget fixed policies.
- Learned routing is validated as superior to cost-matched fixed routing.
- PM has solved resource calibration on EvoEmo.

## 7. Forced-swap blind probe

因为完整 V4 cost-matched response scoring 使用单一正式 order setting，而此前 pairwise / pointwise judge 都暴露过候选顺序敏感问题，所以新增一个最小 forced-swap blind probe：

- sample units: 40
- judge calls: 80
- order 0: PM first, ME+R0 second
- order 1: ME+R0 first, PM second
- metrics: Support, Personalization, Preference
- rule: 若双顺序 preference 不一致，则 resolved preference 记为 tie

复现入口：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  /home/tokkio/miniconda3/envs/sim_eval/bin/python \
  scripts/17s_eval_cost_matched_forced_swap_probe.py \
  --run \
  --sample-units 40 \
  --accept-cost-estimate-sha256 e4c3b6b2a2aa0de99d3de13757b25c2555e20d65c98a6fe7839ce76ea74a9f27 \
  --max-api-calls 100 \
  --max-estimated-usd 2.5 \
  --max-input-tokens-per-call 12000
```

输出：

- `outputs/evoemo_cost_matched_forced_swap_probe/response_summary.json`
- `outputs/evoemo_cost_matched_forced_swap_probe/response_scores.jsonl`
- `outputs/evoemo_cost_matched_forced_swap_probe/response_judgments.jsonl`
- `outputs/evoemo_cost_matched_forced_swap_probe/response_raw_calls.jsonl`

完整性检查：

| Check | Value |
|---|---:|
| sample units | 40 |
| expected calls | 80 |
| judgment rows | 80 |
| score rows | 160 |
| raw rows | 80 |
| successful raw calls | 80 |

Forced-swap score summary:

| Method | Support ↑ | Personalization ↑ |
|---|---:|---:|
| PM | 3.863 | 3.038 |
| ME+R0 | 4.013 | 3.175 |

Dual-order PM minus ME+R0 deltas:

| Metric | PM - ME+R0 mean | 95% bootstrap CI |
|---|---:|---:|
| Support | -0.150 | [-0.250, -0.050] |
| Personalization | -0.138 | [-0.238, -0.050] |

Dual-order resolved preference:

| Resolved preference | Count |
|---|---:|
| PM | 2 |
| ME+R0 | 11 |
| tie | 27 |
| order disagreement | 1 |

解释：

- forced-swap 之后，ME+R0 仍小幅高于 PM；
- 但多数样本为 tie，说明 ME+R0 不是“碾压” PM；
- 关键结论应写为：当前监督式 PM 没有展示出优于同预算固定 `ME+R0` 的稳定优势；
- 该结果不能推出 `ME+R0` 是一般最优策略，只能说明当前 LLM-judge 设置下 fixed medium-resource policy 是强 baseline。

## 8. 是否需要重训

不需要。

该诊断和后续 `ME+R0` fixed-action baseline 都不需要重训 PM。若跑 API，只需要新增一个 fixed condition 的 generation + V4 response scoring。旧的 PM / baseline 结果不需要重跑。

如果要把主方法改成能击败 `ME+R0` 的版本，则需要重新设计训练目标、selection rule 或加入 RL / user-feedback calibration，并重新冻结与重跑外部 generation / response evaluation。这不建议作为当前 V3.3 收口版本的补丁。

## 9. 论文边界

可以写：

> A cost-matched fixed action (`ME+R0`) achieved higher support rating than PM at nearly identical input-token cost, showing that the current supervised PM does not yet provide a robust external advantage over same-budget fixed routing.

> A forced-swap blind probe showed the same direction with many ties, suggesting that the cost-matched baseline is competitive rather than universally dominant.

不应写：

- `ME+R0` 和 PM 质量相同；
- PM 已经优于同预算 fixed baseline；
- cost matched baseline 可以替代 response evaluation。

最稳中文表述：

> 同预算 `ME+R0` 在 V4 pointwise scoring 与 forced-swap probe 中均小幅高于当前 PM，但 forced-swap 中多数样本为平局。因此该诊断应被解释为：当前监督式 PM 尚未证明自己优于强同预算固定动作，而不是 `ME+R0` 已经成为长期情感支持中的通用最优策略。
