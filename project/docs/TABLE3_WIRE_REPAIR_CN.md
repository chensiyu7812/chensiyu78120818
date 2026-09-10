**表 III Batch 字段顺序修复，2026-09-10**

执行到主评分 456 个有效请求时，一条质量请求两次输出说明后持续空白，耗尽 1,600 tokens。所有旧请求与费用均保留，累计费用上界 $3.4175511，无未知费用或仍在排队的请求。

排查确认执行脚本用 `canonical_json(sort_keys=True)` 写 Batch 输入，重排了 JSON Schema 的 `properties`；同步 pilot 用原始 Pydantic 字段声明顺序。质量模型本来最后输出 `reason`，Batch 却把 `temporal_consistency` 排到了 `reason` 后。四个质量失败响应均在 `reason` 之后输出大量空白。字段排序不一致已经确认，不能据此断言它是空白循环的唯一原因。[OpenAI 文档](https://developers.openai.com/api/docs/guides/structured-outputs#key-ordering)明确结构化输出使用 schema 的字段顺序。

这是原实验执行的一处序列化修复：保留原 schema、字段约束、模型、rubric、seed、采样、生成回复、pilot 门槛、统计定义和最大两次尝试；不增加旧请求的重试额度。每个主评分请求重建原始字段顺序，按实际有序 wire body 计算 hash 和新 ID，Batch 使用不排序的 JSON 序列化。全部 2,652 个主评分统一重做，旧的主评分不混入表格；同步 pilot 本来就是正确字段顺序，完全相同的请求可复用。所有 1,224 条自然结束的生成结果复用，无需再生成。

`wire_amendment_v4/parent_ledger.sqlite` 是停止后的父账本快照；28 条补生成 token 记录也逐个固定 hash。新账本继承所有旧费用，包括无效评分，费用不归零。提交前再次核算全量剩余预留，总额不得超过 $9。新格式的未知费用失败仍停止，明确已计费的无效结果最多两次同请求尝试。技术修复不授权改变评分门槛或按评分选择重跑样本。

入口 `scripts/28_run_table3_wire.py freeze|verify|run`。输出 `run_wire_v4` 包括实际 Batch wire 文件、有序 hash 校验、完整 CSV、全部配对区间与中文版分析。旧冻结、运行、评分和 incident 全部保留。
