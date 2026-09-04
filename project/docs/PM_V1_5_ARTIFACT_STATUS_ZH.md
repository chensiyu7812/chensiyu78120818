# Legacy PM-v1.5 supplemental Artifact 完整性状态

更新时间：2026-07-15

> **历史文档：** 本文只记录 `pm-v1.5-supplemental` 分支（冻结 PM-v1 后、不重训的
> post-hoc 诊断）的 artifact 缺口，不描述当前重新训练的 PM-v1.5 快速会议版。当前版本
> 的代码、主张边界与真实执行状态见仓库根目录 `README_PM_V1_5_REVIEW_ZH.md` 和
> `project/docs/PM_V1_5_REVIEW_ARTIFACT_INDEX.json`。

## 已提交到 `pm-v1.5-supplemental`

- PM-v1 原始代码与结果基线（继承自 `c16608343fe60e92e57c622fdc24738efe57d08c`）；
- `PM_V1_5_SUPPLEMENTAL_ANALYSIS_ZH.md`；
- `PM_V1_FAILURE_LIMITATION_POSTMORTEM_ZH.md`；
- cost-matched baseline 诊断文档；
- 更新后的表格与 Figure 2 说明；
- forced-swap 脚本 `scripts/17s_eval_cost_matched_forced_swap_probe.py`。

## 当前 GitHub 中仍缺少的本地 artifact

现有诊断文档引用了下列本地文件，但它们目前不在可访问的 GitHub tree 中：

```text
scripts/17p_analyze_cost_matched_fixed_baseline.py
scripts/17q_run_cost_matched_fixed_baseline.py
scripts/17r_eval_cost_matched_fixed_response.py
outputs/evoemo_cost_matched_me_r0_generation/
outputs/evoemo_cost_matched_me_r0_response/
outputs/evoemo_cost_matched_forced_swap_probe/
```

因此，当前 V1.5 分支已经完成**版本和论文口径分离**，但 ME+R0 完整 API 复现链仍需把上述本地 scripts、summary、scores、judgments、raw-call manifest、cost hash 和 attestation 提交后才能称为完整可复现 artifact。

在这些文件提交前：

- 文档中的 ME+R0 数值可作为已有本地运行记录；
- 不应声称 GitHub 已经包含全部原始输出；
- 不应从文档数字反向重建或修改结果；
- `pm-v1-frozen` 不受此缺口影响，因为它本来不包含 V1.5 补充实验。
