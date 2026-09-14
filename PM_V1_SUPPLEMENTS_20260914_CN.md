# PM v1：2026-09-14 补充实验与全部证据

研究基础为 `pm-v1-frozen`，原六组重跑见 [Table III 入口](TABLE3_RERUN_README_CN.md)。本页对应后来补充的诊断，原模型、阈值和 204 状态主表均保持不变。

## 直接查看

| 材料 | 入口 |
|---|---|
| 最新动作置换与时延结果 | [完整中文报告](project/outputs/targeted_checks_20260914/README_CN.md) |
| 最新实验执行前协议 | [固定方案](project/analysis/targeted_checks_20260914/PROTOCOL_CN.md) |
| 审稿补充的全部结果与限制 | [补充报告](project/outputs/reviewer_supplement_20260914/SUPPLEMENT_REPORT_CN.md) |
| 真实方法定义和论文措辞 | [方法说明](project/analysis/reviewer_supplement_20260914/METHODS_AND_CLAIMS_CN.md) |
| 英文论文增补段落 | [LaTeX](project/analysis/reviewer_supplement_20260914/PAPER_INSERTS_EN.tex) · [三页 PDF 预览](project/outputs/reviewer_supplement_20260914/paper_supplement_preview.pdf) |
| 发布文件清单与校验 | [清单](project/outputs/supplements_20260914_publication/manifest.json) · [离线核验脚本](project/scripts/33_verify_published_supplements.py) |

## 最新结果

**开发动作置换诊断：** 1,728 cards、16 合成用户、5 个留出 fold。相同 fold 与合法动作集合内置换 PM 动作，保留动作组合的使用频率，用已有 judge 标签评价。PM 与置换期望的回复偏好为 0.57612 / 0.51981，差值 +0.05631，用户聚类 95% CI [+0.04893, +0.06363]；平均输入为 555.30 / 555.98 tokens。五个 fold 的回复偏好差值均为正。

这是已有合成开发数据上的条件性机制诊断：使用 OOF 模型和最终 consensus 阈值，不是新的独立测试或 nested validation；[0,1] 回复偏好量纲不能与外部 1–5 Support 混合。

**本地时延试跑：** 48 个状态、18 用户、34 scenarios；五方法、各三次重复，共 720 次测量。原 BF16 Llama 3.1 8B、A6000、自然 EOS；随机交错顺序，计入策略、检索、prompt、生成和解码开销。

| 方法 | 平均首个可见输出 ms | 平均完整回复 ms | 平均输入 tokens |
|---|---:|---:|---:|
| PM OFF | 433.76 | 2,323.61 | 1,277.52 |
| Rule OFF | 502.12 | 2,353.75 | 1,609.38 |
| Fixed OFF | 491.61 | 2,319.48 | 1,636.83 |
| Context Only | 81.02 | 1,842.81 | 375.13 |
| ME+R0 | 216.88 | 2,269.47 | 1,267.94 |

PM 相对 Rule / Fixed 的平均首输出等待减少 13.61% / 11.77%，减少量的用户聚类 95% CI 为 [43.15, 93.39] / [31.08, 85.13] ms。完整回复耗时差值区间跨零；首输出 P95 未优于 Fixed；Context Only 和 ME+R0 更早开始输出。不能写成 PM 全面最快。

计时是常驻模型与预构建索引下的本地处理链路，不包含远程网络、数据库或 UI。所有方法使用相同 PreparedStrategyRetriever 文档特征预计算，各请求清空查询缓存。原三组的 432 次生成回复均与旧结果相同；新 Context Only / ME+R0 回复没有新 judge 质量分。该试跑不替换 204 状态主表，也不证明质量等效。

## 原始数据、统计与冻结

- 时延：[720 条原始请求](project/outputs/targeted_checks_20260914/latency/requests.jsonl)、[逐请求 CSV](project/outputs/targeted_checks_20260914/latency/per_request.csv)、[均值和分项](project/outputs/targeted_checks_20260914/latency/arm_means.csv)、[所有配对区间](project/outputs/targeted_checks_20260914/latency/paired_intervals.csv)、[三次重复](project/outputs/targeted_checks_20260914/latency/by_repeat.csv)。
- 动作置换：[逐 card 对照](project/outputs/targeted_checks_20260914/shuffle/per_card.csv)、[统计区间](project/outputs/targeted_checks_20260914/shuffle/contrasts.csv)、[固定参照全部指标](project/outputs/targeted_checks_20260914/shuffle/policy_means.csv)、[五个 fold](project/outputs/targeted_checks_20260914/shuffle/by_fold.csv)、[库存分解](project/outputs/targeted_checks_20260914/shuffle/by_inventory.csv)。
- 新实验：[执行冻结](project/outputs/targeted_checks_20260914/freeze.json)、[全部请求顺序](project/outputs/targeted_checks_20260914/schedule.jsonl)、[验证记录](project/outputs/targeted_checks_20260914/verification.json)、[代码](project/analysis/targeted_checks_20260914)。
- 审稿补充：[576 条双序逐候选评分](project/outputs/reviewer_supplement_20260914/results/quality_scores.csv)、[全部顺序区间](project/outputs/reviewer_supplement_20260914/results/order_intervals.csv)、[历史需求标签及核验标志](project/outputs/reviewer_supplement_20260914/results/history_need_labels.csv)、[正式运行及账本](project/outputs/reviewer_supplement_20260914/run_v2)、[冻结请求](project/outputs/reviewer_supplement_20260914/frozen_v2)。

审稿补充也保留对 PM 不利或未解决的证据：Context Only 的旧逐项比较、ME+R0 的已有报告及缺失原始文件说明、候选顺序敏感性、历史引用核验失败，以及风险/成本机制的实际作用。首次补充实施有序列化和可见窗口错误，已隔离在 `frozen/` 和 `run/`，原始尝试与费用保留，不混入正式结果。历史需求标签是探索性 LLM 标注，首次 35 条引用核验失败，重试后仍有 25 条未通过。

## 费用

动作置换和本地时延新增 API 费用为 **$0**。审稿补充含全部尝试的计价上界为 **$1.1634704**，连同此前记录累计约 **$7.5509055**，不是官方发票金额。[补充费用记录](project/outputs/reviewer_supplement_20260914/results/receipt.json)。本次上传不发起生成或评分请求。

## 离线核验与复算

新材料均直接包含在 Git 中，保留原始字节和 SHA-256。日志中包含复现所需的进程/设备记录及编译日志；不包含 API 密钥、环境文件或模型权重。Python 字节码缓存未发布。

从仓库根目录，仅验证本次新增文件，无第三方依赖、无 API：

```bash
python project/scripts/33_verify_published_supplements.py
```

如需验证原冻结依赖或复算，先还原此前 Table III 证据包，再核验：

```bash
python project/scripts/32_verify_published_table3.py --restore
python project/scripts/33_verify_published_supplements.py --check-source-dependencies
```

在兼容的 Python 环境中，从 `project/` 可进一步运行：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python analysis/reviewer_supplement_20260914/verify_supplement.py
PYTHONNOUSERSITE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python analysis/targeted_checks_20260914/analyze_results.py
```

准确的生成环境、包版本及模型来源由执行冻结和模型清单记录。分析脚本可以重算现有记录，无需 GPU 或 API；不要为查看结果启动 `run_checks*.py run` 或 `latency.py run`，它们是原实验执行入口。旧的 `release_manifest.json` 为历史打包记录，本次材料由专用发布清单覆盖。
