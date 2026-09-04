# PM V1.5：Strategy open coding 双模型结果与最小人工门

状态：`TWO-PASS COMPLETE / TWO HUMAN SETS COMPLETE / CODER A GROUNDING FAILED`

两份 72 条归并的逐项对照与最终使用边界见
`docs/PM_V1_5_DUAL_STRATEGY_ANNOTATION_DECISION_ZH.md`。Coder A 在全部 160 条中
有 81 条目标 evidence 失配，因此不再参与投票或全量弱回标。

## 1. 做了什么

对 train-only、label-blind 的 160 条 ESConv supporter responses 做了两次独立开放编码：

- Coder A：Llama 3.1 8B Instruct；
- Coder B：DeepSeek V4 Flash；
- 两者均通过项目已有 NVIDIA credit-backed 开发端点；
- 两者都看不到 ESConv 原生八类、现有五族、50 个 submove、problem/emotion 和
  private lineage；
- 每条输出 meaningful action、自由 primary/secondary action、information、
  self-disclosure、reusability、risk 和 response evidence；
- 输出不是 gold，也没有多数投票。

Llama 20/20 batches、DeepSeek 20/20 batches 均完整覆盖 160 条。完整调用的 provider
usage 为：

- Llama：37,776 prompt + 21,835 completion = 59,611 tokens；
- DeepSeek：48,944 prompt + 21,240 completion = 70,184 tokens。

另有：

- Llama 最初一次因非逐字 evidence 被拒，未写入结果；
- DeepSeek 一次连续 503、一次 ID 不完整，恢复后补齐，失败批次不进数据；
- GLM 完成 16 条后因吞吐过慢停止，只作不完整 sensitivity，不进主聚合；
- Mistral 已被 NVIDIA 下线并返回 410，没有产生编码数据。

## 2. 两模型一致性

| 字段 | A 正例 | B 正例 | Exact agreement | Cohen κ |
|---|---:|---:|---:|---:|
| meaningful support action | 134 | 153 | 85.6% | 0.251 |
| mainly information | 15 | 15 | 86.3% | 0.191 |
| mainly self-disclosure | 22 | 34 | 81.3% | 0.357 |
| reusable general technique | 52 | 137 | 39.4% | 0.030 |
| clear risk/boundary problem | 12 | 5 | 93.1% | 0.323 |

Exact agreement 不能单独解释，因为多数类别高度不平衡。κ 显示：

- 两模型对“这是什么动作”有可用但有限的候选信号；
- 对 information/self-disclosure/risk 的边界仍不稳定；
- 对“是否值得成为通用卡”几乎没有一致标准，DeepSeek 明显比 Llama 宽松；
- 因此不能让 LLM 自动决定 card eligibility，也不能把两者多数票称为归纳 taxonomy。

两模型 primary action 短语的 BGE cosine 中位数为 0.648，p10/p90 为
0.514/0.809。该数只反映短语相似度，不用于自动裁决。

## 3. 为什么还需要人，但不需要重标 160 条

人工选择分三层：

### Tier 1：72 条，当前必要

包含：

- meaningful action 分歧；
- information 分歧；
- self-disclosure 分歧；
- risk 分歧或任一模型报 risk。

这 72 条直接决定哪些 source 能支持通用、安全卡，是无法用模型投票替代的研究核心。

### Tier 2：20 条，codebook 稳定后再做

从 55 条仅 reusability 分歧中固定哈希抽样 20 条，用来校准“可复用”的边界。它不阻塞
先从 Tier 1 归纳 codebook。

### Tier 3：15 条，可选校准

从其余一致池固定抽取约 20%，检查“两模型一致”是否仍可能共同犯错。

所以现在不要求一次完成人工 107 条，更不要求 160 条全审。活跃人工任务只有 Tier 1
的 72 条；Tier 2/3 等第一版 codebook 出来后再决定是否需要。

## 4. 当前证据意味着什么

双模型一致认为：

- 132/160 条存在 meaningful action；
- 46/160 条可复用；
- 44/160 条同时 meaningful、reusable 且两者均未报 risk。

这 44 条仍不是 gold，但说明 ESConv train 中存在足够形成紧凑技术库的候选行为，不是
“清洗完没有可用内容”。真正阻塞是动作边界和可复用性定义，而不是 source volume。

人工 Tier 1 完成后：

1. 合并同义 primary actions，形成第一版 atomic-move codebook；
2. 用明确 codebook 重跑/回标 9,148 条全量来源；
3. 每卡检查 20 个独立 dialogue、topic/emotion 分布和风险边界；
4. 与现有 50 张 top-down reference 对齐；
5. 只有通过来源门的卡进入 retrieval qualification。

## 5. 产物

- 双模型完整配对：
  `outputs/pm_v1_5_esconv_strategy_open_coding_aggregation_v1/paired_model_codes.jsonl`
- 当前必要的 72 条人工页：
  `outputs/pm_v1_5_esconv_strategy_open_coding_aggregation_v1/human_review_core.html`
- 可后做的 35 条校准页：
  `outputs/pm_v1_5_esconv_strategy_open_coding_aggregation_v1/human_review_optional_calibration.html`
- 聚合报告：
  `outputs/pm_v1_5_esconv_strategy_open_coding_aggregation_v1/aggregation_report.json`
- 可复跑 runner：
  `scripts/v1_5/23m_run_esconv_strategy_open_coding_v1_5.py`
- 可复跑聚合器：
  `scripts/v1_5/23n_aggregate_esconv_strategy_open_coding_v1_5.py`
