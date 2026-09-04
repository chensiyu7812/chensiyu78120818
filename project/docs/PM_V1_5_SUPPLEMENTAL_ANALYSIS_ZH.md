# Legacy PM-v1.5 supplemental：补充诊断与论文边界说明

更新时间：2026-07-15

> **历史命名警告：** 本文中的 “PM-v1.5” 只指 GitHub 分支
> `pm-v1.5-supplemental`，即冻结 PM-v1 后、不重训的 post-hoc 诊断。它不是当前
> `agent/pm-v1-5-conference-review` 分支中的“PM-v1.5 快速会议版”。当前版本会重新生成
> development 数据、重新训练并重新做外部评测，且截至 2026-07-17 尚无正式结果。审查
> 当前版本请从仓库根目录 `README_PM_V1_5_REVIEW_ZH.md` 开始；本文数字不得作为当前
> 版本的结果。

文档性质：PM-v1 冻结之后的补充诊断。本文不修改原 checkpoint，不重新训练 PM，也不将外部结果用于反向调参。

---

## 1. 一句话定位

PM-v1.5 不是新的 Policy Manager，而是：

> 在保持 PM-v1 方法与 checkpoint 不变的前提下，补充分析其外部动作集中、abstention 失败、selection-rule 敏感性，以及同预算固定动作 `ME+R0` 的竞争力。

它用于修正论文解释，不用于把 post-hoc 结果包装成新的 confirmatory method。

---

## 2. PM-v1 原始主结果仍然保留

原 EvoEmo V4 六条件比较显示：

| Method | Support Rating | Mean input tokens |
|---|---:|---:|
| Context Only | 3.745 | 393 |
| Learned PM | 3.657 | 1292 |
| Rule Source Selector | 3.662 | 1624 |
| Structured Memory Top-k + Strategy | 3.662 | 1651 |
| Raw Session Top-4 + Strategy | 3.750 | 3237 |
| All Raw Sessions + Strategy | 3.618 | 14076 |

这里原始数据字段 `overall` 与 `emotional_support` 完全相同，因此 3.745、3.657 等数值只能称为：

```text
Support Rating / Emotional Support Rating
```

不能继续称为独立的 holistic Overall Quality。

原始结果仍然支持：

1. Context Only 是强局部回复和 abstention baseline；
2. focused raw-session retrieval 可能提高支持评分，但成本更高；
3. full-history injection 成本极高且没有带来更高支持评分；
4. Learned PM 与 Rule / Structured high-resource baselines 的支持评分接近，同时输入 tokens 更少；
5. Learned PM 不是回复质量全面最优策略。

---

## 3. 外部动作分布：PM 没有学会可靠 abstention

EvoEmo 1020 个 fixed-input states 中：

```text
MSE+RS      521
MPMS+RS     165
MP+RS       128
MS+RS        72
MPE+RS       62
ME+RS        34
MSE+R0       27
other R0     10
```

聚合后：

- M0 rate：0.0%；
- R0 rate：3.6%；
- RS rate：96.4%；
- MSE+RS：51.1%；
- action entropy：2.208 bits。

Synthetic reference 上则为：

- M0 rate：24.4%；
- R0 rate：25.0%；
- RS rate：75.0%；
- action entropy：3.426 bits。

正确解释：

> PM-v1 在 synthetic development distribution 上能够输出较多动作，但这种动作多样性没有迁移到 EvoEmo。外部分布下，它仍有一定 source-level variation，却没有可靠学会 no-memory abstention 和 Strategy RAG off。

不能写成：

- PM learned when to turn off all external resources；
- PM robustly adapts across longitudinal users；
- PM uses all 16 actions according to state。

---

## 4. TF-IDF 与外部分布偏移

补充 no-API 诊断显示：

| Split | Zero-vector rate | Mean TF-IDF nnz | Token vocabulary coverage | Unique vocabulary coverage |
|---|---:|---:|---:|---:|
| Synthetic reference | 0.0% | 57.922 | 100.0% | 100.0% |
| EvoEmo | 0.0% | 47.327 | 26.7% | 16.3% |

因此：

- 不能说 TF-IDF 在 EvoEmo 上完全 collapse；
- 但存在明显 lexical distribution shift；
- synthetic 的 100% coverage 部分来自 vocabulary 就是在该语料上拟合；
- external natural language 的低覆盖削弱了细粒度状态区分能力。

Stable caps 使 severe scalar/catalog numeric OOD 为 0，但 raw inventory 规模差异仍然很大：

| Source | Synthetic mean count | EvoEmo mean count | Synthetic mean tokens | EvoEmo mean tokens |
|---|---:|---:|---:|---:|
| MP | 0.852 | 7.000 | 28.218 | 35.324 |
| MS | 1.850 | 22.118 | 41.155 | 891.912 |
| ME | 0.556 | 22.118 | 11.394 | 6159.529 |

所以 stable caps 只能解释为：

> 防止数值越界和不受控外推。

不能解释为：

> synthetic 与 EvoEmo 分布已经匹配，或外部 calibration 问题已经解决。

---

## 5. Semantic-family split：原 user-fold 对回复排序偏乐观

使用既有 synthetic action outcomes 和 silver labels，在 semantic-family split 上重新训练诊断模型：

| Metric | User-fold reference | Semantic-family split |
|---|---:|---:|
| Pairwise accuracy | 0.6734 | 0.6368 |
| Pairwise log loss | 0.5792 | 0.6435 |
| Misuse MAE | 0.0795 | 0.0852 |
| Memory omission MAE | 0.2062 | 0.2149 |
| Memory decision MAE | 0.2357 | 0.2428 |
| Strategy risk MAE | 0.0464 | 0.0390 |
| Strategy decision MAE | 0.0518 | 0.0464 |

五折 pairwise accuracy 约为：

```text
0.637 ± 0.045
```

这说明：

1. user-fold 不能被解释为完全未见语义场景泛化；
2. response ranking 在 semantic holdout 下出现中等程度下降；
3. 风险回归头没有整体崩溃，但仅凭 MAE 也不能证明强风险泛化；
4. EvoEmo 应被视为更强的综合外部压力测试。

该诊断不替换 frozen PM-v1，也不用于重新选 checkpoint。

---

## 6. 为什么 M0 / R0 没有被选择

关键诊断：

| Diagnostic | Synthetic reference | EvoEmo |
|---|---:|---:|
| Best RS score − best R0 score | 0.119 | 0.083 |
| Best memory score − best M0 score | 0.068 | 0.070 |
| Safe M0 exists | 100.0% | 100.0% |
| Safe R0 exists | 100.0% | 100.0% |
| Mean candidates after epsilon | 1.000 | 1.000 |

这排除了“风险门禁把 M0/R0 全部过滤掉”的解释。

更准确的机制是：

1. response scorer 平均偏好 memory-on；
2. response scorer 平均偏好 RS；
3. frozen selection 使用 `epsilon=0.0`；
4. 因此通常只有最高 predicted response score 动作进入候选；
5. cost tie-breaker 几乎没有实际发挥机会。

PM-v1 的真实选择机制应描述为：

> quality-first, risk-constrained selection with cost-aware tie-breaking。

不能描述为：

> 已经完整联合优化 quality、risk 和 cost。

---

## 7. Top-1 / Top-2 score margin

EvoEmo 上：

- mean(top-1 − top-2 predicted response score) = 0.012；
- 53.9% states 的差距 < 0.01；
- 100% states 的差距 < 0.05。

这说明：

> PM 对很多具体高分动作之间并没有强置信度。`epsilon=0` 把很小的预测分差转化为 winner-take-all 决策。

但也必须同时看到：

- best memory − best M0 = 0.070；
- best RS − best R0 = 0.083。

因此不能把全部问题归因于 epsilon。完整解释是：

1. scorer 对 resource-on 有系统偏好；
2. 多个高资源动作之间的分差又很小；
3. zero-margin selection 放大了该偏好。

---

## 8. Epsilon sensitivity 只作为诊断

| Epsilon | M0 rate | R0 rate | RS rate | Action entropy | Mean estimated action cost | Mean candidate count |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0% | 3.6% | 96.4% | 2.208 | 911.9 | 1.000 |
| 0.01 | 0.1% | 6.3% | 93.7% | 2.519 | 750.9 | 1.919 |
| 0.02 | 1.3% | 8.0% | 92.0% | 2.498 | 587.9 | 3.176 |
| 0.05 | 25.8% | 15.4% | 84.6% | 2.326 | 306.5 | 6.514 |
| 0.10 | 48.0% | 56.5% | 43.5% | 2.257 | 159.1 | 9.914 |

可以得出：

> Frozen PM 的外部动作分布对 selection margin 高度敏感；M0 和 R0 并非完全被 scorer 排除。

不能得出：

- epsilon=0.05 已被证明是更优正式 policy；
- estimated action cost 等同于 observed generator tokens；
- 提高 epsilon 后支持评分仍然非劣。

任何正式修改 epsilon 的版本都必须重新冻结、生成和评价，因此不属于 PM-v1.5。

---

## 9. 同预算固定动作 `ME+R0`

`selection_stable.json` 中定义的 budget-matched fixed action 为：

```text
ME+R0
```

在 204 个 V4 sampled states 上，实际生成成本几乎相同：

| Method | Mean input tokens | Median input tokens | Mean output tokens | Mean latency |
|---|---:|---:|---:|---:|
| PM | 1294.6 | 1537.5 | 68.1 | 1.92s |
| ME+R0 | 1283.8 | 1287.0 | 77.8 | 1.21s |

同一 cost-matched evaluation prompt 内：

| Method | Support Rating | Personalization | Memory appropriateness | Factual | Temporal | Non-intrusiveness |
|---|---:|---:|---:|---:|---:|---:|
| PM | 3.755 | 3.495 | 4.255 | 4.500 | 4.525 | 4.480 |
| ME+R0 | 3.926 | 3.696 | 4.289 | 4.505 | 4.529 | 4.451 |

Paired PM − ME+R0：

| Metric | Mean delta | 95% cluster bootstrap CI |
|---|---:|---:|
| Support Rating | -0.172 | [-0.250, -0.093] |
| Personalization | -0.201 | [-0.275, -0.123] |
| Memory appropriateness | -0.034 | [-0.069, 0.000] |
| Factual grounding | -0.005 | [-0.025, 0.015] |
| Temporal consistency | -0.005 | [-0.025, 0.010] |
| Non-intrusiveness | +0.029 | [-0.029, 0.088] |

注意：该 run 中的 PM 绝对分不能直接替换原六条件 V4 主表中的 PM=3.657，因为两次 judge prompt 的候选集合不同。可靠结论是同一次 prompt 内的 paired delta。

---

## 10. Forced-swap blind probe

40 个 matched units 使用双顺序：

- order 0：PM first；
- order 1：ME+R0 first；
- 双顺序 preference 不一致时 resolved 为 tie。

双顺序平均结果：

| Method | Support | Personalization |
|---|---:|---:|
| PM | 3.863 | 3.038 |
| ME+R0 | 4.013 | 3.175 |

Paired delta：

| Metric | PM − ME+R0 | 95% bootstrap CI |
|---|---:|---:|
| Support | -0.150 | [-0.250, -0.050] |
| Personalization | -0.138 | [-0.238, -0.050] |

Resolved preference：

```text
PM wins:       2
ME+R0 wins:   11
Tie:          27
Order disagree: 1
```

正确解释：

- ME+R0 的优势不是单纯由候选顺序造成；
- 多数样本仍然为 tie，不应描述成 PM 被全面碾压；
- PM-v1 没有建立优于同预算固定 Event Memory policy 的稳定优势；
- 不能据此推断 ME+R0 对所有数据、用户或长期目标普遍最优。

---

## 11. 为什么 `ME+R0` 在 EvoEmo 上强

合理机制包括：

1. EvoEmo subsequent topics 与具体历史事件天然对齐，Event Memory 容易提供直接个性化；
2. R0 避免 Strategy RAG 带来的模板化、过度结构化和过早建议；
3. 固定中等资源动作没有额外 routing error；
4. PM 常选 MS+ME+RS，可能引入摘要噪声和策略过度；
5. 当前 fixed-input judge 更重视当前轮承接、自然性和非侵入性。

这些是与结果一致的机制解释，不是已完成的独立因果验证。

---

## 12. PM-v1.5 对论文的影响

### 可以继续保留的主线

- 长期情感支持中的资源使用是一个独立 pre-retrieval allocation 问题；
- memory / strategy 资源不是越多越好；
- Context Only 是必要 abstention baseline；
- focused raw-session retrieval 质量较高但成本高；
- full history 成本极高且没有更高支持评分；
- PM 相对 structured/full-history high-resource baselines 能减少输入资源；
- source-level routing 与 item-level filtering 是不同层次的问题。

### 必须收紧的主张

不能写：

- Learned PM 已证明优于同预算 fixed routing；
- PM 已学会何时不使用记忆；
- PM 已学会何时关闭 Strategy RAG；
- PM 已完成 quality-risk-cost 联合优化；
- PM 是外部自适应长期 support policy。

推荐写：

> PM-v1 is a coarse pre-retrieval source router that reduces context use relative to several high-resource baselines. Supplemental diagnostics show that it remains under-calibrated for abstention and strategy-off decisions and does not establish a robust advantage over a cost-matched fixed Event Memory policy.

---

## 13. 主表与补充材料的建议关系

PM-v1 的原六条件表可以继续作为主结果图景，用于展示资源非单调关系。

`ME+R0` 可以作为：

- Discussion 中的关键补充对照；
- Limitations 中的 same-budget negative result；
- Appendix / Supplement 中的 paired table；
- PM-v2 的直接设计动机。

即使不把 `ME+R0` 放进原六条件主表，正文也不能继续让读者产生“PM 已证明优于同预算固定动作”的印象。

---

## 14. V1.5 与 V2 的界线

PM-v1.5：

- 同一个 frozen PM；
- 不重新训练；
- 不改变 selection；
- 只补诊断和强 baseline；
- 用于解释旧模型边界。

PM-v2：

- 新数据；
- 新标签；
- 新模型；
- 新 utility；
- 新 M0/R0 gates；
- 新 human calibration；
- 新 internal cost-matched gate；
- 必须重新冻结和外部评价。

因此两者必须分支、文档和实验结果完全分开。
