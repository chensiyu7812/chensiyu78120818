# PM v1 定向补充结果

日期：2026-09-14。本轮两项补充已完成并通过离线校验，新增 API 费用 **$0**。未修改 PM、阈值、原主表或此前补充。

[执行前协议](../../analysis/targeted_checks_20260914/PROTOCOL_CN.md)；执行 freeze：`391eb62b5e02e9d7fc1641bfc3dc5e5a596f8ea76135eaa3fe1f3da77edf1b62`。

## 1. 动作置换：受控开发中存在状态匹配收益

使用 1,728 cards、16 合成用户、5 个 user-held-out folds。按 fold 和合法动作集合分为 40 层，在层内置换 PM 动作，精确保留来源/RS 组合频率，用已有逐动作 judge 标签及 sweep 输入记录评价。PM 自身预测分未用于效果评分。

| 指标 | PM | 置换期望 | PM−置换 | 95% 用户聚类 CI |
|---|---:|---:|---:|---:|
| response_preference | 0.57612 | 0.51981 | +0.05631 | [+0.04893, +0.06363] |
| misuse | 0.07523 | 0.06992 | +0.00531 | [-0.00247, +0.01227] |
| omission | 0.28530 | 0.29433 | -0.00903 | [-0.01860, +0.00037] |
| strategy_risk | 0.24277 | 0.25150 | -0.00873 | [-0.03815, +0.02070] |
| input_tokens | 555.30324 | 555.97872 | -0.67548 | [-13.85297, +12.14605] |
| source_invocations | 1.73090 | 1.73090 | +0.00000 | 总体频率由设计精确保持 |

回复偏好增益 +0.05631，五个 fold 均为正。两者平均输入约 555.30 / 555.98 tokens。该结果支持：在这些合成开发状态上，状态与动作的对应关系提供了额外偏好收益，不能仅由平均资源调用频率解释。它没有显示所有风险同时改善。

**范围限制：** OOF 模型应用最终 consensus 阈值，不是新的独立或 nested validation；合成用户留出不等于语义留出。用户 CI 条件于此数据估得的动作频率。开发 response-preference 为 [0,1] 重构量纲，不等于外部 Support，也不推翻既有外部 Context Only / ME+R0 结果。输入预算近似相同，非逐状态精确匹配。

[逐 card](shuffle/per_card.csv)、[差值与 CI](shuffle/contrasts.csv)、[固定参照](shuffle/policy_means.csv)、[各 fold](shuffle/by_fold.csv)、[各库存](shuffle/by_inventory.csv)、[独立验证](shuffle/verification.json)。

## 2. 本地时延试跑：全部 720 次完成

48 状态、18 用户、34 scenarios，五组各重复 3 次。NVIDIA RTX A6000，冻结 Llama 3.1 8B BF16 / SDPA / greedy，batch size=1，随机交错顺序。全部回复自然 EOS，无实验输出 token 上限。

| 方法 | 首个可见文本 ms | 完整回复 ms | 首文本 P95 ms | 完整回复 P95 ms | 输入 tokens | 输出 tokens |
|---|---:|---:|---:|---:|---:|---:|
| PM OFF | 433.76 | 2323.61 | 637.30 | 3189.17 | 1277.52 | 69.58 |
| Rule OFF | 502.12 | 2353.75 | 667.87 | 3076.90 | 1609.38 | 67.67 |
| Fixed OFF | 491.61 | 2319.48 | 593.35 | 2958.07 | 1636.83 | 66.81 |
| Context Only | 81.02 | 1842.81 | 89.88 | 2435.04 | 375.12 | 66.12 |
| ME+R0 | 216.88 | 2269.47 | 251.85 | 3065.44 | 1267.94 | 75.54 |

先在状态内平均三次，再按用户聚类 10,000 次。以下为 PM−baseline；负数代表 PM 较快。所有区间都是探索性逐项区间，没有多重比较联合覆盖保证。

| 比较 | 指标 | 差值 ms | 95% 用户 CI | 均值减少 |
|---|---|---:|---:|---:|
| PM−Rule OFF | 首个可见文本 | -68.36 | [-93.39, -43.15] | +13.61% |
| PM−Rule OFF | 完整回复 | -30.14 | [-126.45, +84.97] | +1.28% |
| PM−Fixed OFF | 首个可见文本 | -57.85 | [-85.13, -31.08] | +11.77% |
| PM−Fixed OFF | 完整回复 | +4.12 | [-117.35, +133.98] | -0.18% |
| PM−Context Only | 首个可见文本 | +352.74 | [+318.78, +387.00] | -435.35% |
| PM−Context Only | 完整回复 | +480.80 | [+330.07, +624.56] | -26.09% |
| PM−ME+R0 | 首个可见文本 | +216.88 | [+186.16, +248.68] | -100.00% |
| PM−ME+R0 | 完整回复 | +54.14 | [-72.71, +174.57] | -2.39% |

**计时边界：** 常驻模型和预构建 inventory/静态索引下，从 runtime state 反序列化到输出；包含策略选择、查询、实际检索、prompt 构造、tokenization、生成和解码。未包含远程网络、数据库或 UI，所以这是本地处理链路时延，不能直接写成线上用户端实测加速。

各方法共享相同 PreparedStrategyRetriever 文档特征预计算；请求间清空查询缓存，只有同请求内可以复用。该共同实现与旧未优化检索日志不同，不跨运行拼接时延。P95 是各方法 144 次请求的描述性统计；三次重复不是新的独立用户样本。

原三组回复对照：{'matched': 432}；同状态同方法跨重复出现不同回复的组数：0。Context Only / ME+R0 仅新增时延和回复记录，没有新 judge 质量分。

[全部逐请求](latency/per_request.csv)、[状态内均值](latency/per_state_means.csv)、[均值与分项](latency/arm_means.csv)、[全部配对 CI](latency/paired_intervals.csv)、[各重复](latency/by_repeat.csv)、[完整汇总](latency/summary.json)。

## 3. 对论文呈现的含义

动作置换结果可以作为内部机制证据：在固定边际动作频率、平均输入近似相同的条件下，匹配状态带来更高开发回复偏好。正文须同时标明 synthetic development 的范围。外部资源优势仍单独用原 204 状态结果呈现，不混合两种质量量纲。

- 相对 Rule OFF 的首个可见文本：差值区间完全低于零，支持本次本地试跑中 PM 均值较低。

- 相对 Rule OFF 的完整回复：差值区间跨零，不能确立时延优势。

- 相对 Fixed OFF 的首个可见文本：差值区间完全低于零，支持本次本地试跑中 PM 均值较低。

- 相对 Fixed OFF 的完整回复：差值区间跨零，不能确立时延优势。

48 状态计时和动作置换都是已有结果后的补充诊断，不能升级成质量等效或普遍最优的确认性证明。Context Only 和 ME+R0 参照保留，不能仅因 PM 优于较高资源方法就称其全局最划算。

## 执行与复现

首次启动 CUDA 数字编号映射到了 A4500，设备断言在推理前停止。随后按 A6000 UUID 锁定设备；没有计时样本被排除。原日志及 [运行环境](execution_environment.json)保留。

在 project 目录使用 sim_eval Python，设置 `PYTHONNOUSERSITE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1`。无需 source API 密钥。

- 验证执行冻结：`python analysis/targeted_checks_20260914/latency.py verify`。
- 汇总与独立验证：`python analysis/targeted_checks_20260914/analyze_results.py`。
- 原始请求结果按 request_id 落盘；同一冻结运行可续跑，已完成请求不重跑。

分析脚本按照冻结指标在运行期编写，哈希单列于 analysis manifest。独立验证包括所有原始成本映射、留出动作对应、40 张代表卡的另一路最小二乘分数重构、720 次请求与执行顺序对应、自然停止、时间分项和完整计时相加一致。

[完整验证记录](verification.json)、[分析文件哈希](analysis_manifest.json)。
