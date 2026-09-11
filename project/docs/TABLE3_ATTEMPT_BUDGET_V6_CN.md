**表 III 重试预算修订，2026-09-10**

`run_identity_v5` 在正式评分推进到 1070/2652 时，第 4 个批次里剩下的最后一条 risk 评分（`unit` 的 `fixed_off` 臂）连续两次都在 `maximum_attempts_per_job=2` 的上限内返回了完全相同（逐字节一致）的响应：schema 合法、自然结束、`audit_call_id` 与预期一致，但 `omission_severity=1`（伴随 `strategy_omission=1`）。协议规定该评分调用必须把 `omission_severity` 固定为 0（遗漏由另一个独立 judge 专门测量），因此两次都被判为无效，触发了程序原有的硬停止 `Frozen ordered-request attempt limit exhausted`：不做隐式重试，也不悄悄接受不合规的响应。

两次失败完全相同，一度怀疑是确定性问题——如果确实如此，再多重试也没有意义。为了在不触碰账本、不产生正式评分的前提下验证，另外用同一条冻结请求体（同模型、`temperature=0`、同一 `seed`）向 OpenAI 同步接口单独发起了一次域外诊断调用，不写入 `ledger.sqlite`，不计入任何评分。结果这次返回 `omission_severity=0`，完全合规。说明这条输入在 API 层确实存在真实的样本间差异，`temperature=0` 加固定 `seed` 并不保证逐次完全确定，两次官方尝试只是恰好都落在了不合规的那一侧。

据此，本修订只做一件事：把 `execution.maximum_attempts_per_job` 从 2 提高到 5，对整条运行统一生效（不是只针对这一条请求开后门），其余一切原样不变——`parse_response`/`record_response` 里的全部校验规则（包括 selected-only omission 必须为 0 的这条硬约束）、模型、温度、seed、prompt、schema、pilot 门槛、bootstrap 设置和 9 美元预算上限都逐字节沿用 `identity_v5`。任何响应仍然必须独立通过现有全部检查才会被接受；不因为分数好坏挑选响应，也不删除、不追溯修改此前两次失败的记录与费用。

是否在此修订重试预算而非直接接受该响应或放宽校验，由用户在看到根因分析和诊断结果后明确选择："扩容重试预算（推荐）"，而非另一个选项"暂停等待人工复核"。

父账本、已提交的全部 Batch 实际请求与原始返回文件、此前的辅助编号修复记录都原样保存在 `identity_amendment_v6` 并固定 hash。新运行 `run_identity_v6` 保存新账本、原始响应、完整表格和分析；入口 `scripts/30_run_table3_attempt_budget_v6.py freeze|verify|run`。`run_identity_v5` 与更早的运行、冻结全部原样保留，不做任何修改。
