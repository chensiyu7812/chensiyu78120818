# PM-v1.5 拆职裁判人工盲评说明

这份人工标注只用于验证新的测量工具，不能直接充当自动训练金标签，也不能单独决定哪个模型可以批量标注。

## 标注对象

`human_blind_packet.jsonl` 共 12 条，来自两类 **train-only** 新鲜状态：

- 6 条纵向合成训练状态；
- 6 条 ESConv auxiliary 训练状态。

它们均未出现在此前失败的 12 条 longitudinal bakeoff 或 24 条 ESConv pairwise pilot 中。文件不展示 domain、state/user/action ID、regime、模型身份或预期方向。

## 质量比较

对每条的 A/B 回复填写：

- `overall_preference`：A、B、tie 或 insufficient；
- `support_quality_preference`：A、B、tie 或 insufficient；
- `quality_confidence`：1–5；
- `quality_notes`：简短说明。

这里只比较回复质量和情绪支持质量。不要在这两个字段里另行综合记忆、策略、安全或费用；这些由下面的逐回复审计独立处理。

## 逐回复证据/风险审计

模板只列出编译器判定对该候选动作适用的风险维度。每一行填写：

- `verdict`：`no_violation`、`violation` 或 `insufficient_evidence`；
- `severity`：0–3；`no_violation` 和 `insufficient_evidence` 必须为 0，`violation` 必须为 1–3；
- `response_excerpt`：回复中的精确原文；若审计的是遗漏，写 `[omission]`；
- `evidence_excerpt`：支持判断的可见对话或 selected context 精确原文；不存在则写 `[none]`；
- `reason`：简短解释。

不要因为系统选择了更多记忆或策略就自动判风险。只有最终回复真实误用、暴露、冲突、虚构、遗漏、过度套用或遗漏必要策略时才判 `violation`。无法从所给证据可靠判断时选择 `insufficient_evidence`，不要猜。

## 提交纪律

1. 只编辑 `human_annotation_template.jsonl` 的空白标注字段。
2. 不修改 `blind_item_id`、维度名或列表结构。
3. 人工标注必须在任何新裁判 API 调用前完成并冻结哈希。
4. 这 12 条只用于资格赛参照，不用于对同一批结果继续调 prompt、阈值或 schema。
