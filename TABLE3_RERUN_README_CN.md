# Table III：v1 六组重跑与完整结果

本次重跑已完成：204 个固定状态 × 6 组，共 1,224 条自然结束回复；2,652 次正式评分请求均已处理，包含一条已授权且记录的辅助字段校验例外。研究基础为 `pm-v1-frozen`，提交 `c16608343fe60e92e57c622fdc24738efe57d08c`。本分支为 `experiment/pm-v1-table3`。

## 直接查看结果

- [全部结果大表：77 行汇总项](project/outputs/table3_under10/run_identity_v7/all_results/ALL_RESULTS_CN.md)
- [Excel 工作簿](project/outputs/table3_under10/run_identity_v7/all_results/Table3_All_Results.xlsx)：总表、1,224 条逐条数据、238 条配对区间、评分与来源分布、576 条 pilot 观测、运行费用及全部尝试账目。
- [总表 CSV](project/outputs/table3_under10/run_identity_v7/all_results/all_metrics.csv) · [逐条数据 CSV](project/outputs/table3_under10/run_identity_v7/all_results/per_response.csv)
- [原表列格式](project/outputs/table3_under10/run_identity_v7/TABLE3_ORIGINAL_FORMAT_CN.md) · [中文分析](project/outputs/table3_under10/run_identity_v7/TABLE3_CN.md)
- [最终核验与例外更正](project/outputs/table3_under10/run_identity_v7/FINAL_AUDIT_CN.md)

主评分模型为 `gpt-4.1-mini-2025-04-14`，GPT-4o 仅用于固定 12 状态 pilot。生成模型为本地 A6000 上的 Llama 3.1 8B Instruct，使用已记录的官方一致权重与固定 tokenizer。沿用 v1 PM checkpoint、Rule／Fixed 配置、fixed seeker tracks 和相同六组输入。全部方法使用同一个 memory-only post-retrieval Filter，Strategy 保持原样。

在相同 Filter 设置下，PM 与 Rule／Fixed 的 Overall 均值接近，PM 的来源数和生成输入更少；质量等效尚未被正式证明。生成环节耗时与离线策略／Filter 耗时已有记录，但没有完整、统一的端到端延迟测量。总表完整保留所有已测量维度，论文选列另行讨论。

## 原始证据与还原

原始执行目录、历次请求／响应、旧结果、v6／v7 父账本快照及最终账本保存在 [压缩证据包目录](project/outputs/table3_under10/published_archive) 中。[清单](project/outputs/table3_under10/published_archive/archive_manifest.json) 记录每个原始文件、压缩包及分卷的 SHA-256。最终 SQLite 账本原始大小超过 100 MiB，因此使用可无损还原的压缩包保存，并按最多 40 MiB 分卷；还原脚本会自动拼接和校验。页面直接展示的分析文件也原样包含在包中。

从仓库根目录执行，无需 API 密钥：

```bash
# 核验压缩包、逐个成员和当前直接展示的结果。
python project/scripts/32_verify_published_table3.py

# 如需查看 SQLite 或复算，将压缩成员还原至原相对目录。
# 已有且 hash 相同的文件会复用；遇到内容不同的已有文件会停止。
python project/scripts/32_verify_published_table3.py --restore
```

还原不会调用生成或评分 API。压缩包不含密钥、环境文件、Python 环境、模型权重、锁文件或进程日志。完整请求、响应、费用、生成回复及补生成记录均保留。生成权重的文件 hash、来源和 tokenizer 信息见 [模型清单](project/outputs/table3_under10/local_model_v2/model_manifest.json)。

## 复算大表

```bash
# 先执行上面的 --restore；建议在独立 Python 环境中安装分析依赖。
python -m pip install -r project/analysis/table3/requirements.txt
python project/analysis/table3/export_all_results.py --output /tmp/table3-all-results-rebuilt
```

复算入口仅读取已完成的请求与评分，重建 Excel／CSV／Markdown，不调用 API。原始导出脚本及修改前中文报告保存在 [provenance](project/analysis/table3/provenance)，其旧服务器路径是历史记录；可移植复算请使用上面的入口。原分析 QA 清单中的绝对来源路径映射到本 checkout 的相同 `project/` 相对路径。

## 冻结方案与验证

最终冻结 hash：`fcf404e9d1d35038d803c58cde50c40dbda474f0e5b6fe7d48cb074b4f3ad765`。

还原证据后，在具有冻结依赖的环境中，可从 `project/` 运行：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/31_run_table3_risk_omission_exception_v7.py verify
```

运行代码与每次修订均保留：

- [初始冻结执行方案](project/docs/TABLE3_FROZEN_EXECUTION_CN.md) · [低于 $10 的预算方案](project/docs/TABLE3_UNDER10_CN.md)
- [本地生成修订](project/docs/TABLE3_LOCAL_AMENDMENT_CN.md) · [自然结束修订](project/docs/TABLE3_EOS_AMENDMENT_CN.md)
- [Batch 字段顺序修复](project/docs/TABLE3_WIRE_REPAIR_CN.md) · [辅助编号关联修复](project/docs/TABLE3_IDENTITY_REPAIR_CN.md)
- [重试预算修订](project/docs/TABLE3_ATTEMPT_BUDGET_V6_CN.md) · [单条校验例外](project/docs/TABLE3_RISK_OMISSION_EXCEPTION_V7_CN.md)

原 28 条触及 100-token 上限的回复全部补至 EOS，前 100 tokens 完全匹配，24 条文本改变。最终表格未混入旧主评分，所有旧费用保留。重复失败的 `fixed_off` risk 请求采用最早原始响应，只豁免它不参与表格遗漏指标的辅助字段检查；原修订说明中“五次 reason 逐字一致”的说法已由最终核验更正，冻结原文保留。

费用账本上界为 $6.3861351，另有一次约 $0.0013 的未入账诊断估计，合计约 $6.39。诊断响应没有进入正式评分，合计不是精确官方账单金额。

本次发布以本页、最终冻结清单、结果清单、证据包清单和最终核验为入口；仓库早期全局 `release_manifest.json` 属于历史打包记录。
