**表 III：总 API 预算低于 10 美元的执行方案（2026-09-10）**

采用 NVIDIA 免费 Llama 生成、OpenAI 官方 GPT-4.1 mini Batch 全量评分，并用 GPT-4o 对固定的小样本交叉检查。预计 $6.53，加 25% 预留为 **$8.17**；执行预算预留上限设为 **$9**。本方案替代上一版全量 GPT-4o 的费用建议。NVIDIA 按用户确认的免费条件计为 $0。

保留原 204 个状态、六组 `PM / Rule / Structured Fixed × Filter OFF / ON`，全量生成 1,224 条回复。PM checkpoint、fixed seeker tracks、生成参数、Filter 参数、三个评分模板和配对统计设计均沿用此前离线方案。更换的是主评分模型与计费渠道；最终结果应注明 `gpt-4.1-mini-2025-04-14`，不能称为完全相同 judge 的原数字复现。

| 工作 | 规模与渠道 | 预计 API 费用 |
|---|---|---:|
| Llama 生成 | NVIDIA 免费；包含 pilot 生成 | $0 |
| 全量质量评分 | 204 次，GPT-4.1 mini Batch | $0.45 |
| 全量误用审计 | 1,224 次，GPT-4.1 mini Batch | $1.13 |
| 全量遗漏检查 | 1,224 次，GPT-4.1 mini Batch | $1.76 |
| mini 技术 pilot | 固定 12 个状态、168 次同步评分 | $0.44 |
| GPT-4o 交叉检查 | 同一批状态、168 次同步评分 | $2.75 |
| **预计合计** | 不计缓存折扣 | **$6.53** |
| **含 25% 余量** | 基于未四舍五入金额计算 | **$8.17** |

GPT-4.1 mini 官方普通文本价格为每百万输入 $0.40、输出 $1.60；Batch 按普通价的 50% 计算，因此本方案全量评分的有效单价为输入 $0.20、输出 $0.80。GPT-4o 交叉检查使用每百万输入 $2.50、输出 $10.00。Batch 使用 24 小时 completion window，实际总工期还取决于账户排队额度及批次数。[mini 官方价格](https://developers.openai.com/api/docs/models/gpt-4.1-mini)、[GPT-4o 官方价格](https://developers.openai.com/api/docs/models/gpt-4o)、[Batch 官方说明](https://developers.openai.com/api/docs/guides/batch)。

**执行顺序与评分检查。**

1. 使用已冻结的输入先生成 pilot 的 72 条回复。12 个状态按 seed × turn 的六个组合各选两个，覆盖 12 个不同用户；候选排序由固定 seed 的 SHA256 决定，在看到回复和评分前固定。完整名单已落盘，主实验仍包含全部 204 个状态。
2. 两个 judge 分别对相同 pilot 评分：每个状态做两种候选顺序的质量评分，以及六条回复各自的误用和遗漏审计。检查 JSON 完整性、候选 ID、截断、换序后的评分波动，逐项查看两个 judge 的分歧和理由。严重的解析或 rubric 理解问题应先修复，再决定是否提交全量评分。继续条件不得依赖 PM 是否占优。
3. pilot 可用后完成全部生成，将三类评分送入 mini Batch。按 `custom_id` 对齐结果，保持六组使用同一主 judge；GPT-4o 小样本分数单独报告，不混入主表均值。12 个状态只能支持初步检查，不能证明 mini 与 GPT-4o 等效；GPT-4o 也不是人工金标准。
4. 收齐配对结果后生成表 III，保留质量、实际误用、遗漏和资源指标，并计算用户聚类的配对置信区间。结果方向由实际观测决定。

**如何落实少于 10 美元的约束。**

配置中的执行预留上限为 $9。联网执行器必须在每次提交或重试前核算：已结算费用＋已排队／用量未明请求的预留＋本次请求按输出上限的预留，合计不得超过 $9。生成完成后，应重新统计真实回复的输入 tokens。失败且无 usage 的请求不能自动当作零成本，也不能自动升级到更贵模型。若余量不够，保存断点并停止新提交。

目前的 $8.17 是计划预算，不是已被运行时强制执行的账单上限。API 执行器及其预算控制尚未接入；这份离线脚本不读 API key，也不提交请求。执行限额可以防止继续提交超预算任务，但不能保证任意故障和重试情况下仍完成全部实验。

除了预期输出估价，还计算了所有评分输出达到配置上限、输入额外增加 10% 的场景，合计 **$8.03**。该场景与 25% 余量是两种估算方式，不相加；它仍是规划值。Llama 与 GPT 的 tokenizer 不同，尚未生成的回复每条按 100 个 GPT tokens 预留，最终需按真实文本复核。

**落盘产物与验证。**

- [预算明细](../outputs/table3_under10/plan/budget.json)：分阶段调用数、实际输入模板 token 估算、预期输出及输出上限两种费用计算。
- [执行配置](../outputs/table3_under10/plan/config.json)：固定模型 snapshot、免费生成假设、Batch 价格、$9 限额要求和 pilot 规则。
- [固定 pilot 名单](../outputs/table3_under10/plan/pilot_units.json)、[来源与文件 hash](../outputs/table3_under10/plan/manifest.json)。
- [离线预算脚本](../scripts/22_budget_table3_under10.py)：先验证此前 plan 的 manifest、数据、代码和基线 hash，再基于实际冻结输入重新估价；不覆盖旧方案。

重读规范化 JSONL 会改变质量评分 payload 内嵌字典的键顺序，因此 BPE 边界相对原内存估价少了 14,478 个 tokens，费用影响很小；信息内容不变，风险与遗漏输入 token 数完全一致。预算 JSON 单独记录了这个差异。

```bash
cd /home/tokkio/chensiyu78120818-table3/project
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/tokkio/audits/pm-v1-table3-20260910/.venv/bin/python \
  scripts/22_budget_table3_under10.py --out outputs/table3_under10/plan_new
```

本轮付费推理调用为 **0**。这是可核算、可审查的低成本重跑方案，尚未产出新的表 III 实验分数。
