# PM V1.5 D3 RS / MP / ME 质量聚合与最后风险门（2026-08-01）

## 当前结论

原 D3 160 对质量人评完整有效，但原 MS 40 对已因 seeker/supporter query 污染整体撤销，并由独立的40对MS替换实验封账。因此本轮不按结果挑样本，而是机械保留原 D3 的全部 RS、MP、ME 120 对，机械排除原 MS 全40对。

解盲后的120对结果：

| 组件 | 开启胜 | 关闭胜 | tie | 32个state的稳定1.0正例 | 0.5重复分歧 |
|---|---:|---:|---:|---:|---:|
| RS | 13 | 20 | 7 | 9 | 2 |
| MP | 7 | 13 | 20 | 6 | 1 |
| ME | 15 | 19 | 6 | 9 | 5 |

40对/组件包含32个独立state和8个结果前冻结的重复生成；训练与资格判断只按32个state等权，不把40次点击当40个独立group。风险前的state target分别为：RS=`21×0, 2×0.5, 9×1`；MP=`25×0, 1×0.5, 6×1`；ME=`18×0, 5×0.5, 9×1`。

## 对四条 memory fidelity 决胜项的回查

原 D3 中 MP/ME 共有且仅有4条以 visible-context fidelity 决胜。逐条把两臂实际授权的同用户严格过去证据对回回复后，没有发现像MS补裁那样足以推翻原人评的缺失证据：

- MP breakup 开胜项中的“hurt can come back suddenly”已经由当前用户原话逐字支持，不依赖私人历史；
- MP sleep 关胜项的profile只支持轮班改变作息，不足以把“身体正在适应新节律”升级成已证实个人事实；
- ME move 关胜项中，新城市与独居有证据，但“尤其在不忙于跑腿或活动的日子”仍是额外推断；
- ME breakup 关胜项中，既往分手与提醒事件有证据，但“已经有一段时间没有想起”没有得到授权证据支持。

因此这4条保留原人评，不新增样本、不翻票。该回查只确认缺失证据是否会否定原判，不冒充第二位独立质量评审。

## 最后一次人工门

只对35个“组件开启且质量实质胜出”的具体回复执行一次 interaction-and-grounding material-risk 审核：RS 13、MP 7、ME 15。页面隐藏组件、arm、state/user和资源ID；memory回复显示同用户严格过去的授权证据与相对年龄，以便判断无依据、陈旧或冲突使用。

该审核的分母只是35个质量胜出的component-on pair realizations。结果不能写成组件整体风险率，也不能与未审核的control臂相减为因果风险差。

审核完成后，程序会先用风险门修正每个pair observation，再按contrast slot聚合成96个state targets；随后与已经封账、但不具判别训练资格的MS结果合并。RS/MP/ME只有在冻结的user-grouped OOF中胜过先验并达到最低双向决策与稳定性门，才能称为学会；否则同样如实报告负结果，不调seed、C、阈值或encoder挽救。

## 资产

- 质量聚合：`outputs/pm_v1_5_d3_rs_mp_me_quality_aggregation_v1/aggregation_report.json`
- 120个pair测量：`outputs/pm_v1_5_d3_rs_mp_me_quality_aggregation_v1/pair_quality_measurements_pre_risk.jsonl`
- 96个risk前state targets：`outputs/pm_v1_5_d3_rs_mp_me_quality_aggregation_v1/state_soft_targets_pre_risk.jsonl`
- 最终风险页面：`outputs/pm_v1_5_d3_rs_mp_me_on_win_risk_review_v1_candidate/human_risk_review.html`

