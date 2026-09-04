# PM V1.5 首个 RS 开关训练结果

状态：`MINIMUM DEVELOPMENT SIGNAL LEARNED / FORMAL GATE NOT PASSED`

日期：2026-07-30

## 1. 数据与标签

- 32 个独立 ESConv-train dialogue/user groups；
- 同状态、同 generator、同 system、同 seed，仅 R0/固定 Top-1 RS 不同；
- 人盲质量：RS 13 胜、R0 13 胜、tie 6、uncertain 0；
- tie 因质量等价而 RS 成本更高，映射为 off；
- 13 个 RS 胜例的 material-risk 最小审核全部为 no；
- 最终训练标签：RS-on 13、RS-off 19、unknown 0。

这些标签是固定生成协议下的人评 effect proxy，不是客观偏好或个体因果效应。

## 2. 三个模型的同一 grouped-OOF 结果

| 模型 | 可见输入 | Brier gain 对 train-fold prior | BA | 正例召回 | 开/关 | 判断 |
|---|---|---:|---:|---:|---:|---|
| 六个透明 state 特征 | 用户 state | -0.02774 | 0.4028 | 0.3846 | 16/16 | 未学会 |
| BGE-small PCA4 | 可见对话 state | -0.00713 | 0.4595 | 0.0769 | 4/28 | 未学会 |
| coarse candidate family L2 logistic | 候选发现后的五类 family one-hot | +0.00476 | 0.5972 | 0.6154 | 16/16 | 最低开发信号通过 |

最后一个模型的 family 系数方向：

- Restatement `+0.2795`
- Question `+0.1789`
- Affirmation `+0.0725`
- Reflection `-0.3461`
- Suggestion `-0.4189`

这与人评观察一致：基础模型常已能给出普通建议或情感反映，而聚焦提问、忠实复述更可能
补足当下回复。

## 3. 为什么修改决策时点

旧定义要求 PM 在 retrieval 前决定是否开 RS，却又要求它预测“实际 Top-1 卡是否有
边际收益”。模型只知道用户 state，不知道准备提供什么资源，目标信息不足。

当前定义：

1. 确定性 scope/boundary gate；
2. 廉价 candidate discovery，只形成 eligible Top-1 及 coarse family；
3. PM_RS 只读取用户 state 与 coarse family，决定是否允许注入；
4. on 才把卡片 guidance 放入 generator prompt；
5. candidate lookup 成本与实际 RS 注入成本分别报告。

PM 不读取 card ID、完整卡文、retrieval score、生成回复、judge 或 outcome。

## 4. 允许和禁止的结论

允许：

> 在固定 Bank 与检索器下，粗粒度资源类型提供了一个弱但正的 grouped-OOF
> RS 开关信号；模型能同时开关，proper score 优于先验。

禁止：

- PM 已稳定学会 RS；
- BA `.597` 足以证明正式泛化；
- BAAI 是最终最优表示；
- 该结果代表客观人类偏好、临床收益或外部测试成功。

正式门未过的两项是 BA `<.70` 和五 seed BA 标准差 `.0667>.03`。当前 32 组停止方法
搜索，模型冻结为 development candidate，下一证据必须来自未见 groups 或外部评测。

## 5. 可复算产物

- 最终标签：
  `outputs/pm_v1_5_rs_effect_final_labels_v1/pm_rs_effect_labels.jsonl`
- state-only 透明模型：
  `outputs/pm_v1_5_minimum_pm_rs_effect_v1/pm_rs_training_report.json`
- BGE challenger：
  `outputs/pm_v1_5_bge_pm_rs_challenger_v1/bge_pm_rs_report.json`
- candidate-aware 模型：
  `outputs/pm_v1_5_candidate_aware_pm_rs_development_v1/candidate_aware_pm_rs_report.json`
