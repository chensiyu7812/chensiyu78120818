# PM V1.5 D2 最终双人盲评结果

状态：`STRICT D2 FAIL / ANNOTATIONS VALID / SOFT-TARGET RECOVERY ONLY`

本文件取代先前“第二附件错配、D2 尚未完成”的临时结论。正确的
`independent_overlap_32` 已收到并通过机器校验。

## 结论

两份标注都能正式使用：72 条 primary 与 32 条 independent overlap 的协议、ID、枚举、
理由和评审者身份均合格，且两位评审者的 annotator ID 不重叠。

使用后的正式结论不是 D2 通过，而是预冻结的严格三分类 inter-rater 门失败：

- 同一 state 的独立生成方向复现：`30/32 = .9375`，通过 `.70` 门；
- primary uncertain：`0/72 = 0`，通过 `.10` 上限；
- 两位评审者对同一 32 pair 的 treatment/control/tie 精确一致：
  `21/32 = .65625`，低于 `.75` 门；
- 严格三分类 Cohen's kappa：`.4854`；
- 若只作诊断，把 treatment 映射为 on、control/tie 映射为 off，则一致率
  `26/32 = .8125`，kappa `.6175`。

最后一项不能事后替代原先冻结的三分类门，因此 formal D2 仍为 FAIL。

## 标注质量

第二份文件本身没有发现不可用问题：

- `32/32` 行齐全、blind ID 与冻结 overlap packet 完全一致；
- A/B/tie 分布为 `16/11/5`，无 uncertain；
- decisive criterion 与理由均非空；
- annotator 为 `clinician_reviewer_01`，与 primary 的
  `independent_psychology_support_reviewer` 不重叠；
- 页面 A/B 位置反转的一致性说明也与填写内容相符。

因此不能把 formal FAIL 解释成“第二份标注格式错误”或“评审没有认真做”。

## 分歧在哪里

| 组件 | 精确一致 | 总数 | 一致率 |
|---|---:|---:|---:|
| RS | 8 | 8 | 1.000 |
| MP | 5 | 8 | .625 |
| MS | 5 | 8 | .625 |
| ME | 3 | 8 | .375 |

11 条严格分歧中，大量是 tie 与某一侧实质更好的阈值差异；但其中仍有 6 条会改变最终
on/off 映射，不能全部当作无关紧要的文风偏好。ME 是最明显的测量薄弱点，RS 在本批
反而达到 8/8。

`2deea586` 的失眠项正是直接方向分歧：primary 判 control，第二评审判 treatment。
第二评审已明确报告低把握，因此保留为真实分歧；不得在解盲后翻票或单独裁决来提高门值。

## 三个协议问题

1. manifest 使用 `request_dialogue_fit`、
   `clarity_naturalness_not_overloaded`，页面导出使用
   `request_and_dialogue_fit`、`clarity_naturalness`。这是 schema alias，不是构念变化；
   分析器已统一到页面名称，后续 packet 应只保留一套枚举。
2. 32 条 `current_session_summary` 全为空是冻结设计，不是漏包。当前完整可见对话是固定
   baseline；MS treatment 是生成器私有资源，盲评者不应看到组件或记忆证据身份。
3. `1e5d8eb9` 与 `a1484c8e` 的共同源对话含截断文本 `make it a a`。两臂和两位评审者
   同受影响，且 RS 两人一致为 8/8，因此不事后删除；任何 D3 packet 必须增加破损句尾
   preflight。

## 对训练的含义

本结果支持“资源的边际质量效应不是稳定的单次硬标签”，不支持“资源无效”或“PM
不可能学习”。生成重复本身很稳定，而不同评审者的 materiality threshold 仍有明显差异。

按预冻结合同，当前禁止把每个单次 pair 直接转成 hard on/off gold。最快且不继续堆人评的
恢复路径是：

1. 到此停止 D2 人评，保留严格 FAIL；
2. 将每个独立生成 pair 的 treatment-win 记为 1，control-win/tie 记为 0；同一 pair 的
   两位评审者先平均，再对同一 state 的独立生成取均值，形成 soft expected-benefit
   diagnostic；
3. D2 只有每组件 8 个 state，只能检验软目标与特征是否有可学习方向，不能单独晋级 PM；
4. 若该零新增成本诊断仍无方向，V1.5 应缩窄主张，而不是继续换 encoder、搜 seed 或补人评；
5. 若有方向，任何 D3 扩充必须在生成 outcome 前冻结抽样、重复次数、judge/human 复核比例、
   user-group split 与停止门，不能再用单 realization 胜负作 hard gold。

机器事实源：

`outputs/pm_v1_5_d2_measurement_review_analysis_v1/analysis.json`

