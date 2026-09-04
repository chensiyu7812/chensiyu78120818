# Policy Manager 版本索引

更新时间：2026-07-18

本仓库中的 PM 研究按四条明确区分的轨道管理。尤其注意：历史
`pm-v1.5-supplemental` 和当前“PM-v1.5 快速会议版”曾共用 V1.5 这个简称，但方法与结果
血缘完全不同，不能混用。

## PM-v1：原始冻结研究版本

分支：[`pm-v1-frozen`](https://github.com/chensiyu7812/metacom-v33-review/tree/pm-v1-frozen)

冻结提交：

```text
c16608343fe60e92e57c622fdc24738efe57d08c
```

用途：

- 保存原 V3.3 / PM-v1 方法、结果和六条件 EvoEmo 主实验；
- 不包含后补 `ME+R0`、forced-swap 或 PM-v2；
- 该分支应保持只读。

## Legacy PM-v1.5 supplemental：PM-v1 的补充诊断版本

分支：[`pm-v1.5-supplemental`](https://github.com/chensiyu7812/metacom-v33-review/tree/pm-v1.5-supplemental)

主要文档：

- `project/docs/PM_VERSIONING_AND_BRANCHES_ZH.md`
- `project/docs/PM_V1_5_SUPPLEMENTAL_ANALYSIS_ZH.md`

用途：

- 保持 PM-v1 checkpoint 和选择规则不变；
- 汇总 OOD、semantic split、margin/epsilon sensitivity；
- 汇总同预算 `ME+R0` 和 forced-swap 结果；
- 修正 Overall、claim boundaries、Discussion 和 Limitations；
- 不构成新训练模型或重新选择后的 confirmatory method。

## 当前 PM-v1.5_1：审查修复后的重新训练候选版

版本分支：[`pm-v1.5_1`](https://github.com/chensiyu7812/metacom-v33-review/tree/pm-v1.5_1)

首要入口：仓库根目录 `README_PM_V1_5_REVIEW_ZH.md`

全局历史失效模式与不可回归检查入口：
`project/docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md`。任何后续局部修复都应先按该文档的
因果链、改动影响矩阵和阶段停止门检查，不能只验证当前报错点。

`PM-v1.5_1` 是 `PM-v1.5` 方法族的协议修复发布版本；方法族内部稳定标识仍为
`pm-v1.5`，配置中的 `release_revision` 明确记录 `pm-v1.5_1`，二者不得与 legacy
supplemental 混用。

2026-07-18 审查后的下一次运行合同：
`project/docs/PM_V1_5_PROTOCOL_REPAIR_CONTRACT_ZH.md`。该合同当前为
`IMPLEMENTED_NOT_EXECUTED / PAID_RUN_BLOCKED_PENDING_FRESH_POST_REPAIR_PILOT`；V8.2 已消费
失败，旧 V8.3 dry-run 也已因后续 runtime/输入合同修复而失效；后续配置解锁不能替代全新
dry-run identity 的独立审查与逐阶段 approval manifest。原快速
会议版文档只作为历史设计背景。

用途：

- 基于 PM-v2.2 的 fail-closed 基础设施重新生成 development 数据并重新训练 PM；
- 统一 development、sweep 与 external 的 Strategy Bank、retrieval、supporter generation
  和关闭的 Evidence Filter 合同；
- 以“相对高资源 fixed comparator 的六维回复质量非劣，同时实际输入 tokens 更少”为
  条件性主要外部主张；
- 以 synthetic internal-test 的完整 16-action 反事实矩阵提供有边界的路由证据；
- 不做人评，不主张临床或真实用户改善，risk 仅作分层证据误用审计，latency 仅作描述性
  实测。

截至 2026-07-17，该轨道发布的是付费正式执行前的代码与方法审查包：没有训练好的
PM-v1.5 checkpoint、study freeze 或外部主结果。Legacy supplemental 的任何数字都不是
当前版本的结果。

## PM-v2：实质性重设计

分支：[`pm-v2-redesign`](https://github.com/chensiyu7812/metacom-v33-review/tree/pm-v2-redesign)

用途：

- 新数据、标签、模型和 multi-objective selection；
- explicit M0/R0 gates；
- human label calibration；
- internal cost-matched reportability gate；
- 必须重新训练、冻结和外部评价。

## Main 分支

`main` 当前作为整合与审查入口，包含多个版本的文档和代码。论文复现或引用时不要只写 `main`，应明确指定：

- `pm-v1-frozen`；
- `pm-v1.5-supplemental`（legacy post-hoc）；
- `pm-v1.5_1`（当前协议修复后的重新训练候选版）；
- 或 `pm-v2-redesign`。
