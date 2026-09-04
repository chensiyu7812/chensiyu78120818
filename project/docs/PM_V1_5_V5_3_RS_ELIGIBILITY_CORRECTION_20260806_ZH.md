# RS eligibility发现更正（2026-08-06）：区分两套真实并存的实现，修正抽样身份

**状态：更正脚本65/文档`PM_V1_5_V5_3_RS_ELIGIBILITY_DOMAIN_COMPARISON_20260806_ZH.md`和
账本`V15-DATA-81`。不删除原文档/原ledger行（项目惯例），本文件是权威更正记录。账本行本身
的编辑留给leader复核后处理，本文件先把正确结论摆出来。**

## 原结论错在哪

脚本65把"训练域160个RS state用真实生产eligibility runtime算出0%机会"这句话里的
"真实生产runtime"，实际调用的是`v1_5_strategy_rag_runtime.observable_flags()`+
`eligible_moves()`——这是一套**较新、独立的"6卡原子动作"系统**（`AM01-AM14`，卡片来源
`strategy_cards_v1_5_minimal.jsonl`，恰好6行）。但V3 blueprint构造RS候选时，实际调用的
是`v1_5_strategy_rag_repair.effect_study_observable_flags()`——一套建立在
`v1_5_strategy_rag_v4.py`之上、**用ESConv原生5个安全策略家族**（Question/Restatement/
Reflection/Affirmation/Providing Suggestions，"50核心卡→100执行卡"）的独立系统。两套
都是项目里真实存在、目前并存的实现，脚本65拿A系统去审计用B系统构造的数据，结论必然对不上，
不是"训练数据有问题"。

**代码证据**：`grep`确认`v1_5_strategy_rag_runtime`被`prompts.py`（真实回复生成的
system prompt构建）和`v1_5_v5_3_contribution_slot_features.py`引用；
`v1_5_strategy_rag_repair`被`v1_5_v3_candidate_materialization.py`（V3构造本身）引用；
`v1_5_strategy_rag_v4`还被`v1_5_final_raw_generation.py`直接引用（绕过repair层）。也就是
说**这不是"一个真一个假"，是三层实现（v4基础层→repair/effect_study覆盖层→runtime新6卡层）
在代码库里同时存在、被不同模块各自引用**，哪个该作为V5.3正式生产标准，不是这次脚本能替
leader决定的事。

## 同时修正抽样身份（脚本65的另一个真实问题）

脚本65的"真实ESConv 3.4%"来自`ESConv.json`前300段（脚本默认截断），不是完整或正式测试集；
EvoEmo只用了138个state（qualification+lockbox），漏了66个diagnostic state。这次改用：

- **真实ESConv**：`data/esconv_test_v1_5/pm_v2_states.jsonl`，169个已冻结、不重叠的正式
  test dialogue的全部2112个support-eligible turn；
- **真实EvoEmo**：`outputs/pm_v1_5b_final_external_panel_v2/evoemo_panel_private.jsonl`，
  204个正式state（60 qualification + 78 lockbox + 66 diagnostic）。

## 更正后的真实结果（脚本`66_rs_eligibility_domain_comparison_v2_v1_5.py`）

| 域 | n | `effect_study`（V3实际用的系统）not-hard-off率 | 6卡AM系统any-move-eligible率 |
|---|---:|---:|---:|
| 训练域（V3 RS构造） | 160 | **100.0%** | 0.0% |
| 真实ESConv（169正式test dialogue，2112 turn） | 2112 | 94.4% | 4.8% |
| 真实EvoEmo（204正式state） | 204 | 95.6% | 9.3% |

## 结论：原来的"严重问题"判断需要拆成两半

1. **用V3构造实际调用的系统（`effect_study_observable_flags`）审计，训练域和两个真实域
   完全可比**（100% vs 94.4%/95.6%），没有发现严重域间差距——**原来"RS训练时只见过OFF、
   问题比ME更严重"这个结论不成立，是审计工具选错了，不是数据真的有问题**。
2. **但6卡AM系统（`eligible_moves`）这个真实、独立存在的系统上，训练域0% vs真实域
   4.8%/9.3%这个差距，在修正抽样身份后依然存在**——这是一个更窄但依然真实的发现：**如果
   `v1_5_strategy_rag_runtime`这套6卡系统就是（或将成为）V5.3真正的生产判定标准，那么V3
   构造的RS训练数据从未在这套系统下验证过、也确实通不过**。这不该被完全撤回，应该降级重
   新表述为："两套并存实现之间的构造/运行时对齐问题"，而不是"训练数据只见过OFF"。

## 交给leader的具体问题

1. V5.3正式生产到底采用`v1_5_strategy_rag_runtime`（6卡AM）还是
   `v1_5_strategy_rag_v4`+`v1_5_strategy_rag_repair`（50核心/100执行卡，5家族）体系，
   还是需要重新统一？这不是本文能替leader做的决定；
2. 如果最终选定某一套，RS训练构造必须改成调用**同一个**判定接口，不能像V3这次一样用一套
   系统构造、被另一套系统审计（或未来被另一套系统拿去做真实检索）；
3. 账本`V15-DATA-81`需要leader复核后正式更正措辞（本文件先给出正确结论，未直接改账本行，
   遵照这一轮"不修改三份权威事实源"的要求）。
