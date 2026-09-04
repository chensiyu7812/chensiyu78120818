# PM V1.5 RS 自然检索修复与留出资格

状态：`FINAL HOLDOUT READY / WAITING FOR 16 HUMAN DECISIONS`

## 已经确定的事实

- 首批自然 Top-1 开发审计为 13/32 安全合适、9/32 触发硬排除；
- 17/32 的更好候选已经位于 Rank-2/3，只有 2/32 在可见 Top-3 中无安全卡；
- 这支持“catalog 多数时候有附近候选，但原排序器不合格”，不支持“30/32 已经形成
  clean treatment”；
- lexical score 和 margin 不能作为适用性置信度；
- 首批 32 条只作 development，不是 PM component-effect label。

机器报告：

- `outputs/pm_v1_5_rs_natural_retrieval_review_v1_analysis/analysis_report.json`
- `outputs/pm_v1_5_rs_card_applicability_development_replay_v1/`

## 修复后的固定运行顺序

1. phatic、stop、active-high-stakes、非实质状态先 hard-off；
2. 从可见 cue 决定 execution profile 和 eligible family；
3. 按每张卡的显式前提与排除条件做 card-level applicability；
4. 没有适用卡则 abstain；
5. 只在适用卡内用 term-frequency cosine 排序；
6. Top-1 score 低于 0.05 则 abstain；
7. 最多注入一张卡；
8. PM_RS 随后判断该固定 state-card pair 是否有较大概率产生 risk-first material
   benefit。PM 不负责修复召回。

card-level 规则只使用当前和近期 seeker 文本中的显式 cue。它不使用 ESConv 下一条
supporter response、原生 strategy label、回复生成、judge preference、formal test
或 EvoEmo。

## 数据防作弊

- Development：首批 32 dialogues，可用于修规则，不得用于无偏准确率；
- Pre-freeze dry run：16 dialogues，在无人评、无回复 outcome 时发现技术遗漏后撤销；
- Final holdout：另 16 dialogues，与上述 48、V1–V4 direct-effect、formal test
  全部零重叠；
- 最终 holdout 人评后禁止继续调规则；
- ESConv formal test 与 EvoEmo 只在 PM、Bank、retriever、generator、metrics 全部
  冻结后运行。

## 最终留出与门槛

最终页面：

`outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_final_candidate/human_retrieval_review.html`

固定门槛：

- 安全合适 Top-1 至少 12/16；
- hard exclusion 必须为 0；
- 每个出现的 family 的安全合适率至少 50%；
- 只有 `top1_fit=yes`、`hard_exclusion_triggered=no`、
  `better_candidate=none` 才是 clean treatment。

通过后进入 train-only R0/RS matched generation。失败后不再对这16条调参，V1.5
回退六卡 runtime 或把 expanded natural retrieval 写成局限性。
