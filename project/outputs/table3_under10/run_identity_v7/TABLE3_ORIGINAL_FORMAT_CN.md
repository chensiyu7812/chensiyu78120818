# Table III：按原表列重排本次六组结果

沿用原 Source-set fit / Context misuse / Sufficiency / Major issues 列，六组使用本次相同的 204 个固定状态。主 judge 为 GPT-4.1-mini-2025-04-14；没有混入历史表格或 GPT-4o pilot 的分数。

| Method | Filter | n | Source-set fit ↑ | Context misuse ↓ | Sufficiency ↑ | Major issues ↓ |
|---|---|---:|---:|---:|---:|---:|
| Learned PM | OFF | 204 | 4.382 | 0.010 | 4.525 | 0 |
| Learned PM | ON | 204 | 4.123 | 0.025 | 4.436 | 0 |
| Rule Source Selector | OFF | 204 | 4.412 | 0.015 | 4.642 | 0 |
| Rule Source Selector | ON | 204 | 4.348 | 0.015 | 4.456 | 0 |
| Structured Fixed | OFF | 204 | 4.539 | 0.005 | 4.667 | 0 |
| Structured Fixed | ON | 204 | 4.353 | 0.029 | 4.471 | 0 |

Source-set fit 与 Sufficiency 为 selected-context risk 审计的 1–5 分；Context misuse 为 0–3 严重度均值，不是百分比。Major issues 为 verdict=major_issue 的样本数。Sufficiency 沿用原审计列含义，取 risk 阶段 response_support_sufficiency；独立 omission 阶段同名字段是另一上下文下的判断，没有混用。

## 质量与资源补充

| Method | Filter | Quality (Overall) ↑ | Retrieval calls ↓ | Memory tokens candidate → kept ↓ | Generator input tokens ↓ |
|---|---|---:|---:|---:|---:|
| Learned PM | OFF | 3.446 | 2.716 | 683.3 → 683.3 | 1274.6 |
| Learned PM | ON | 3.265 | 2.716 | 683.3 → 71.5 | 693.3 |
| Rule Source Selector | OFF | 3.436 | 3.000 | 1024.6 → 1024.6 | 1607.4 |
| Rule Source Selector | ON | 3.270 | 3.000 | 1024.6 → 91.5 | 729.8 |
| Structured Fixed | OFF | 3.480 | 4.000 | 1037.8 → 1037.8 | 1634.8 |
| Structured Fixed | ON | 3.284 | 4.000 | 1037.8 → 93.0 | 732.2 |

Quality 为本次预先固定的 overall（1–5），与 emotional_support 是不同字段。Retrieval calls 为实际请求的逻辑来源数（含 Strategy），不是 HTTP 请求数。Memory tokens 是 v1 chars/4 估计；Generator input 为实际生成记录中 Llama tokenizer 的完整输入 token 数，二者口径不同。自然结束补生成沿用完全相同输入，所以不改变输入 token 数。

当前可比较的主张是：在相同 Filter 设置下，PM 与 Rule／Fixed 的质量均值接近，同时使用更少的检索来源和生成输入。已有 Overall 配对区间包含 0，支持如实报告差异与不确定性，尚不构成正式等效证明。

当前记录没有完整、统一口径的端到端延迟测量。生成计时未包含实际在线 PM、检索与 Filter 全流程，而且补生成与原生成为不同执行批次；因此端到端延迟留空，不能用逻辑来源数或 token 减少比例代替时间减少比例，也不将旧 NVIDIA 延迟并入本次本地重跑。

本表保留已授权的 fixed_off 单条 risk 辅助 omission 校验例外；样本和最早原始评分均保留，详见[最终核验](FINAL_AUDIT_CN.md)。

[本格式 CSV](table3_original_format.csv) · [完整原始导出](table3.csv) · [配对置信区间](paired_intervals.json)
