**表 III 冻结执行记录，2026-09-10**

用户已授权：固定低于 $10 的方案，然后开始运行；密钥通过用户指定的本地环境文件加载，不进入代码、输出或 Git。

固定输入与方法：204 states、六组、v1 PM checkpoint 和 fixed seeker tracks、NVIDIA Llama 3.1 8B 生成；GPT-4.1 mini 全量 Batch 评分、固定 12 个状态的 mini／GPT-4o 同步检查。生成的 pilot 回复按相同请求 hash 复用于主实验；主表评分仍使用完整 204 个状态的 mini Batch 结果。

冻结包位于 `outputs/table3_under10/frozen_v1/`。manifest 记录完整配置、pilot 名单、代码、原始计划、checkpoint、数据及依赖版本的 SHA256。启动、生成、Batch 提交和统计导出前核验。旧冻结文件不覆盖；若确需修复代码或变更参数，必须另建带来源说明的新版本，保留原结果及账本。

pilot 继续条件在生成前固定：全部 72 条回复及两个 judge 的 336 次评分可解析；两个 judge 分别通过 v1 默认的换序平均绝对差 ≤0.50、位置均值最大偏移 ≤0.40。两个 judge 在 order 0 上的 overall 平均绝对差 ≤0.75；明确误用与明确遗漏（severity ≥2）的分类一致率分别 ≥80%。后两项是本次预先指定的运行筛查门槛，不能解释成已验证的模型等效标准。所有分歧及理由均保存，门槛不涉及任何方法是否胜出。门槛失败即暂停全量评分。

费用控制使用 SQLite 整数账本。每个请求先预留输入估算×1.25＋64 tokens、输出达到上限时的费用；已结算费用、失败尝试及队列中请求均纳入 $9 限制。没有 usage 的调用保留预留，不自动视作免费或重新提交。实际用量超过单次预留也会停止。每个逻辑请求最多两次尝试；没有自动换模型、去掉 schema 或扩大 token 上限的回退。

主评分按约 100 万输入 tokens 分批，最多同时一个 Batch；每 45 秒查询状态。单批最长按服务的 24 小时 completion window 处理，完整耗时取决于账户额度和队列速度。输出按 custom_id 合并；进程中断后已成功请求复用，待结算请求先恢复查询。未知提交结果不会盲目重发。

全量完成后输出 `table3.csv`、`TABLE3.md`、逐条 `observations.jsonl` 和 `paired_intervals.json`。主指标为 overall、明确误用比例、明确遗漏比例；同时导出 ≥1 的任意误用、严重度及其他质量维度。置信区间以全部状态均值为估计量，用户聚类重采样为主，scenario 为敏感性分析，10,000 次，seed=20260910。缺失任何状态／组／评分任务都不导出完整表。

运行状态、原始响应、请求、费用账本均写到 `outputs/table3_under10/run_v1/`，与冻结包分开。该目录被 Git 忽略。没有自动推送仓库或对外发布。

```bash
cd /home/tokkio/chensiyu78120818-table3/project
set -a
source /home/tokkio/metacom5_3.env
set +a
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
/home/tokkio/audits/pm-v1-table3-20260910/.venv/bin/python scripts/23_run_table3.py verify
/home/tokkio/audits/pm-v1-table3-20260910/.venv/bin/python scripts/23_run_table3.py run
```

同一条 `run` 命令用于断点恢复；`status` 查看状态及费用，`export` 只重新导出已完整收齐的结果。冻结命令 `freeze` 仅在输出目录不存在时运行，不能用它覆盖既有冻结。
