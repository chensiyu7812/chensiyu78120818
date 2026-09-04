# Policy Manager 版本索引

更新时间：2026-07-17

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

## 当前 PM-v1.5：重新训练的快速会议版

审查分支：[`agent/pm-v1-5-conference-review`](https://github.com/chensiyu7812/metacom-v33-review/tree/agent/pm-v1-5-conference-review)

首要入口：仓库根目录 `README_PM_V1_5_REVIEW_ZH.md`

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
- `agent/pm-v1-5-conference-review`（当前重新训练的快速会议版，审查阶段）；
- 或 `pm-v2-redesign`。
