# PM V1.5 G0 完成：分歧裁定与 Atomic-Move Codebook

状态：`G0 COMPLETE / G1 IN PROGRESS / 2026-07-29`

规范产物：

- `data/pm_v1_5_contracts/strategy_g0_disagreement_adjudication_v1.json`
- `data/strategy/pm_v1_5_strategy_atomic_move_codebook_v1.json`

生成审计：

- `outputs/pm_v1_5_strategy_g0_freeze_v1/g0_freeze_report.json`
- `outputs/pm_v1_5_strategy_g0_freeze_v1/adjudicated_human_labels.jsonl`

## 1. 结论

另一份分析所说的“只完成 G0 一半”在执行前是准确的：两份 72 条人评存在 21 条至少一个
字段分歧，当时只有操作定义，没有逐条机器可读裁定，也没有冻结 codebook。

现在 G0 已完成：

- 21/21 条分歧按预先冻结的字段语义解决；
- 未解决分歧为 0；
- 72 条形成统一 source-screening 标签；
- 开放动作归并为 17 个可多标签使用的 atomic moves；
- 五个行为区域均有 move 覆盖。

这不是第三位独立盲法人评。人评页面曾展示模型提示，当前裁定又使用了两份标注及已冻结
规则，因此只能称为 `rule-based methodological synthesis`。它足以冻结 codebook 和
source-screening 语义，不能增加 IAA、不能成为 PM gold。

## 2. 裁定后的数据

| 字段 | true | false |
|---|---:|---:|
| meaningful support action | 58 | 14 |
| mainly externally verifiable information | 6 | 66 |
| substantive supporter self-disclosure | 19 | 53 |
| reusable execution | 52 | 20 |
| broad source risk/boundary problem | 14 | 58 |

机械要求 meaningful、reusable、非 information、非 self-disclosure、非 risk 后得到 29 条
候选种子。这个数字不是最终合格来源数。人工标注仍可能漏掉混合回复中的真实人生经历、
高风险事实内容或不安全上下文；G1 必须回看 literal source response。

## 3. 17 个 atomic moves

Codebook 不是 17 个互斥类别；一条回复可以分配多个 move：

1. invite open expression；
2. ask one focused clarification；
3. check feeling or coping；
4. tentative paraphrase and check；
5. grounded validation；
6. bounded normalization；
7. acknowledge demonstrated effort/strength/resource；
8. repair misunderstanding；
9. collaborative problem structuring；
10. one optional micro-step；
11. reinforce a user-generated plan；
12. explore trusted support；
13. bounded hope；
14. bounded conversational transition；
15. explore an interpersonal boundary；
16. gentle behavior-focused challenge；
17. brief non-domain rationale。

其中 challenge 和 rationale 当前只有 abstract/counterexample 证据，G1 找不到安全正向来源
就删除，不能为了完整性造卡。Repair 等来自混合高风险上下文的 move 也必须另找干净来源。

`permission`、`tentativeness`、`low-burden wording`、`warmth` 和 `gentle humor` 被定义为
执行约束或 style modifier，不被人为扩成新动作类别。

## 4. 21 条分歧怎样解决

主要规则是：

- 个人身份、账户、婚姻和宠物回答若没有推进支持，meaningful=false；
- 平台消息、quit、众包时长是 corpus artifact，不是 external information；
- self-disclosure 看支持机制是否依赖真实生活经历，不按是否出现 “I” 判断；
- reusable 只问当前执行能否安全直接转成 LLM technique，不问“修好后是否可能有用”；
- source risk fail-closed，所以永久可用承诺、宗教价值引入、公共卫生过度概括、诈骗线索
  正常化、高风险 DV 指导、确定性预测和虚假希望都排除直接来源。

风险回复仍可以贡献 counterexample 或 abstract move，例如把
“你很 manipulative”转化为“只挑战可见行为，不给人贴标签”。但原句不计来源支持。

## 5. 同时关闭一个 G4 小缺口

“OOF 同时产生 on/off”不能等看到结果后再解释。现在预先固定为：

- 只在该组件 opportunity 存在、in-support 的 OOF groups 上计算；
- `on` 和 `off` 各至少 4 个独立 groups；
- `on` 和 `off` 各至少占 10%；
- 两个条件同时满足才算没有塌缩；
- external 自然机会分布不强制满足该比例，只报告真实 coverage。

这防止 95% off / 5% on 被含糊写成“已经产生两种决策”，也不会要求外部域人为配平。

## 6. G1 已启动

G1 不直接生成 17 张卡。当前进度为：

1. 全部 9,148 条已由 lexical+BGE 做 candidate-recall scoring；
2. 每 move 的 120 个语义候选、40 个词法候选和 5 个固定低分 controls 合并后，得到
   1,273 条、600 个独立 dialogues 的候选池；
3. 122 条 Coder B pilot 已完成 16 个有效 batch calls；逐字证据、合法 ID、全部
   17 moves 至少一次激活均通过；
4. pilot 仍只产生弱候选。66 条无硬排除候选不是 gold；低 BGE 分也不是 gold
   negative，不能据此计算“准确率”；
5. controls 的 5 个 move 激活逐条存在真实可见动作，因此当前证据指向原
   BGE/TF-IDF shortlist 召回不足，而非 Coder B 系统性过贴；
6. 已加入只用于 recall 的 train-only native-label strata 和透明 pattern，且对 coder
   隐藏。原 Stage 1 323 条整包未运行，只接受 64 条/8 批；
7. 90 条/12 批 adaptive wave 与 63 条/8 批 coverage-completion wave 已完成；选择未
   查看 validation/test/EvoEmo/PM/回复质量 outcome，且合同规定停止后续弱来源扩张；
8. 四个互斥集合共 339 条、280 个 train dialogues。AM01/02/04/05/10/14 达到 20 个
   独立 clean 弱来源标准门；AM07=17、AM15=11 为显式窄动作例外候选；其余 9 moves
   不进入 V1.5 Bank 候选；
9. 两份 outcome-blind 人评均已完成。40 个 card-example judgments 实际对应 39 个
   unique items、38 个 unique dialogues；move-fit 一致 31/40，hard-exclusion 一致
   26/40；
10. weak-source qualification 因重复、定义漂移和遗漏 hard exclusions 撤销，来源只作
    formative provenance；不进入 generator、PM label 或 permission gold；
11. 两人共同拒绝 AM07/AM15 narrow exceptions，且不补位。其余 AM01/02/04/05/
    10/14 按具体 corrections 冻结为 6-card G2 development Bank；
12. G1 人评结束，不再重抽、重评或补卡。正式资格由 G2 同栈 eligibility/retrieval
    和 G3 matched-pair component effect 决定。

因此当前可以准确写为：研究方法已定，G0 已完成，G1 已启动但尚未完成，G2–G5 尚未
开始；还不能训练 PM，也还不能运行确认性外部实验。
