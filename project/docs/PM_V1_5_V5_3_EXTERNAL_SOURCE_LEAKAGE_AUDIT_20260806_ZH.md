# V5.3 EvoEmo / ES-MemEval 同源防泄漏机械审计（2026-08-06）

状态：**response/QA 字段投影与禁用字段 canary 已完成；内部—外部内容重叠门 pending。**

本审计是零 API、零生成回复、零 quality/risk outcome 的输入面检查，不是 PM、检索器、generator
或论文效果的 PASS。机器报告位于：

`outputs/pm_v1_5_v5_3_external_source_leakage_audit_v1/report.json`

## 已完成的机械检查

- 对真实 EvoEmo 18 个用户的全部因果切点检查 419 个 response source projection；投影顶层字段严格
  只有 `basic_info` 与严格过去的同用户 `dialog_history`。
- current/future session 暴露为 0，跨用户 session 暴露为 0。
- 对 response source 中所有 evaluator-only `answer/evidence/capability` 等字段做 canary 改写；419 个
  projection 的序列化输入均不变，失败为 0。
- 对 `questions` 与 `summaries` 两个 released evaluator 容器的 1,552 条 QA row 做 canary；改变
  `answer/evidence/capability/theme/group` 后，retrieval query 与 generation messages 均逐字节不变，
  失败为 0。
- QA session documents 全部来自同一 user，跨用户暴露为 0。

一个需要冻结的细节：真实文件中有一个裸 `session_id` 被两个不同用户共享。因此 session 身份必须使用
`(user_id, session_id)`，不能假定裸 ID 全局唯一。当前审计器按复合身份检查，没有把这一合法碰撞误报成
跨用户泄漏。

## 尚未完成，不能伪造为 PASS

当前没有提供正式、冻结的 V5.3 internal superdomain model-visible text artifact，因此以下门保持：

`PENDING_FORMAL_V5_3_SUPERDOMAIN_SURFACE`

审计器已经实现可复用接口，可在正式 artifact 冻结后检查：

1. internal model-visible fields 与 external question 的全文 exact overlap；
2. 与 external evaluator-only answer 的全文 exact overlap；
3. 与 external same-user session text 的全文 exact overlap；
4. Unicode NFKC + casefold + word-token normalization 后的冻结 n-gram overlap（当前默认 8-gram）。

碰撞报告只保存 internal/external surface ID 和内容 SHA，不把 external answer/session 原文复制进报告。
如果正式 superdomain schema 使用不同字段名，必须先显式绑定 model-visible 字段，不能递归抓取整份 JSON，
否则可能把 label、gold 或 private audit 文本自身误当成训练输入。

## 正式接线前仍需完成

- 将 `project_evoemo_response_source()` 或等价的逐字段投影接进正式 V5.3 response runner；当前报告明确
  记录 `formal_v5_3_generation_runner_wired_to_this_projection=false`。
- 正式 superdomain 冻结后，以 `--internal-superdomain <path>` 运行脚本 69；出现任一 exact 或冻结
  n-gram overlap 时 fail-closed，并人工确认模板性公共短语是否需要更长 n-gram 敏感性分析。
- 将最终 superdomain path/SHA、EvoEmo source SHA、n-gram 规格与审计报告 SHA 绑定进 P2 plan。
- 本门只防直接同源文本/字段泄漏，不证明“语义上没有模仿 EvoEmo”，也不证明外部评测未被开发过程暴露；
  后两项仍由暴露账本和研究报告边界处理。

## 复现

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python \
scripts/v1_5/69_audit_v5_3_external_source_leakage_v1_5.py
```

正式 superdomain 存在后：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python \
scripts/v1_5/69_audit_v5_3_external_source_leakage_v1_5.py \
  --internal-superdomain <frozen-json-or-jsonl>
```
