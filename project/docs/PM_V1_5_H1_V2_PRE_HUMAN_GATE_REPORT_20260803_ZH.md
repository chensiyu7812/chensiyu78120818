# PM V1.5 H1 v2 人评前数据资格报告

状态：`PASS / READY_FOR_FINAL_H1_V2_REVIEW`

## 结论

H1 v1 的人评结论不能训练：公开样本跨 split 重复，MP 近乎恒关，RS 近乎恒开，而且 RS
只标了卡片“看起来合适”，没有标相对 R0 的非冗余增量。

H1 v2 已在**不读取旧人评答案、回复结果或外部 lockbox**的条件下重建并通过人评前静态门。
它现在可以进入一次完整 H1 审核，但尚不是 gold、尚不可训练，也不代表 PM 已学会。

## 修复后的科学合同

每个 exact Rank-1 候选分别审核两门：

1. `applicability_safe`：owner、时间、当前目标、边界与安全是否允许使用；
2. `incremental_over_r0`：候选是否相对当前可见上下文与 resource-free R0 提供具体、非冗余用途。

机械派生规则为：双 `yes` 才 ON；任一 `no` 为 OFF；其他情况为 ABSTAIN。单次生成的实际
赢家不进入 H1 gold；资源是否真正做功、是否误用以及最终 quality–risk–cost 由 H2/H3 独立验证。

RS 检索器只保留状态级硬排除、可观察的 technique-family 检索与内容排序。它不使用完整
`when_to_use/when_not_to_use`、非冗余或收益规则替 PM 提前决定开关。明确“一点式”请求只限定
负担；如果卡片只是重复用户已说出的动作或上一轮已经执行的动作，第二门应为 `no`。

## 人评前静态门结果

| 检查 | 结果 |
|---|---:|
| 独立 state / user | 256 / 256 |
| FIT / fresh confirmation / sealed internal | 128 / 64 / 64 |
| model feature rows | 1,024 |
| MP exact Rank-1 present / absent | 209 / 47 |
| MS exact Rank-1 present / absent | 236 / 20 |
| ME exact Rank-1 present / absent | 236 / 20 |
| RS exact Rank-1 present / absent | 224 / 32 |
| 人工需裁决 / 无候选自动OFF | 905 / 119 |
| exact公开payload重复组 | 0 |
| topic-normalized跨split重复组 | 0 |
| candidate-present构造代理最大多数占比 | 0.6214（门槛 `<0.70`） |
| 预期正候选漏检 | 0 |
| RS technique-family realization | 224/224 |
| API / 人评标签 / 回复结果 / 外部lockbox读取 | 0 / 0 / 0 / 0 |

这里的“构造代理”只证明题目预先覆盖了正负条件，**不是人评 gold，也不进入模型**。最终真实
标签分布只能由 H1 双门审核得出；如果真实标签仍然退化、IAA不足或 ABSTAIN过高，就停止训练
并回到定义/候选层，而不是用构造 bit 覆盖人评。

## 现在需要完成的唯一人评

- 主审：256 个 state bundle；同一状态内同时看 MP/MS/ME/RS，119个无候选自动OFF；
- 独立 overlap：固定64个state，按16个动作与三个split分层抽取；
- ON/OFF 只需两个门、subtype和证据码；仅 uncertain 强制写说明，避免再次把时间消耗在
  每条回复式长评语上；
- 审核结束后先算 raw agreement、Cohen's kappa、每组件一致率、ABSTAIN率、真实标签分布、
  template-only probe 与 constant baseline；全部通过才进入四个低容量 logistic heads。

## 产物

- 私有 blueprint：`data/pm_v1_5_final_candidate_first_v9_h1_v2/private/`
- 零 API raw states：`outputs/pm_v1_5_p2_h1_v2_zero_api_raw_v10/`
- exact Rank-1：`outputs/pm_v1_5_p2_h1_v2_exact_rank1_v10/`
- 静态门：`outputs/pm_v1_5_final_h1_v2_pre_human_static_gate_v1/static_gate_report.json`
- 主审与 overlap：`outputs/pm_v1_5_final_h1_v2_candidate/`

## 尚未证明

本报告没有证明四个 head 可学习、16动作可稳定联合运行、RAG/记忆一定改善回复，或外部
ESConv/EvoEmo 泛化成立。它只证明：这次交给 H1 的题目不再带已知的重复、恒定标签、候选
错位和 RS 预解开关问题，值得进行最后一次 Step 1 gold 审核。
