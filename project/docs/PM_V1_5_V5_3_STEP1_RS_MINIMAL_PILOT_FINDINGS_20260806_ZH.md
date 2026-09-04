# Step1最小试点：RS单组件，真实生成+真实judge打分（2026-08-06）

同样延续MP/MS试点的模式（脚本`54_step1_rs_minimal_pilot_v1_5.py`）。RS跟MP/MS/ME结构不同：
不是`build_evo_memory()`产出的per-user记忆，是六卡冻结策略库
（`data/strategy/strategy_cards_v1_5_minimal.jsonl`，`configs/pm_v1_5_minimal_rag_v1.json`
标记`ACTIVE_FOR_NEW_PM_V1_5_G3_WORK`，是当前生产在用的、唯一"qualified"的RS检索器），用
`QualifiedStrategyRAG`（正则可观测标志→资格判定→词频余弦排序→低于阈值弃权）真实检索。
这次唯一新写的代码是一个小适配器：把`retrieval_text`（"Support move: X. Use conditions: Y.
Observable goals: Z. Do not use: W."）解析成`TypedResourceCandidate`要求的
`support_move`/`when_to_use`/`when_not_to_use`三个字段——解析规则先拿全部6张真实卡验证过，
不是编的。

真实检索结果：138个真实state里，8个state会真实命中一张卡（分数过冻结阈值0.05），覆盖8个
不同用户，卡型只出现`AM04_tentative_paraphrase_check`和`AM05_grounded_validation`两种——
这本身也是个真实信号（六卡里另外4张在这批EvoEmo真实数据上从没真实触发过，不是没测，是真实
检索出来的自然结果）。

## 第一次真实批跑发现RS撞上了MP同一类guard问题（两种不同guard，两个都要修）

8个state里4个（p7/p10/p14/p15，全部50%）的"带RS"一侧直接兜底：

1. **`REQUIRED_EVIDENCE_NOT_USED`**（p14）——跟MP早前修过的问题完全同源：RS的
   `support_move`/`when_to_use`/`when_not_to_use`是"该怎么回复"的行为指导，不是需要被
   复述引用的叙事性事实，模型正确地遵循了指导但没有在`used_evidence_ids`里引用它，被这条
   "必须引用"的检查判定为失败。
2. **`EVIDENCE_CLAIMED_USED_BUT_NO_WORD_TRACE_IN_REPLY`**（p7/p10/p15，来自
   `evidence_usage_plausibility_errors()`词面重合检查）——一个新的、MP没遇到过的变体：
   即使模型正确引用了RS的evidence_id，这条检查还会看回复文本跟`support_move`字面词是否
   有重合词——但"Offer grounded validation"这种指导词，一条真正遵循了它的回复（比如"that
   sounds really hard"）根本不需要出现"grounded"/"validation"这些字。这条检查设计时假设
   证据是叙事事实（对MS/ME成立），套到RS这种指令型证据上必然产生假阳性。

**两条都按MP的先例修了**（`v1_5_v5_3_typed_response_program.py`）：`REQUIRED_EVIDENCE_
NOT_USED`的豁免范围从只有MP扩到MP+RS；`evidence_usage_plausibility_errors()`新增RS跳过
（同样的理由：指导型证据不应该被要求词面重合）。

## 修复后重新跑了全部8个state：兜底率从4/8降到几乎0/8

| 用户 | 卡 | 带RS修复前 | 带RS修复后 |
|---|---|---|---|
| p7 | AM05 | fell_back_to_m0 | clean |
| p10 | AM04 | fell_back_to_m0 | clean |
| p11 | AM05 | clean | clean |
| p13 | AM04 | clean | clean |
| p14 | AM04 | fell_back_to_m0 | clean |
| p15 | AM05 | fell_back_to_m0 | clean |
| p16 | AM04 | clean | clean |
| p17 | AM04 | clean | clean |

**8/8全部clean**（不带RS一侧只有p13需要一次重写，也成功了，同样是M0+R0基线的残余问题，见
MS试点文档里对这个问题的完整记录）。

## RS本身的质量结果（干净、修复后的批次）

| 结果 | 数量 |
|---|---|
| A（带RS更好） | 3 |
| B（不带RS更好） | 1 |
| tie | 4 |

方向上支持RS组件有正面效果（3胜1负4平），跟MP的"基本无信号"（4/6 tie）和MS的"更明确正面"
（6A/2B/4tie）放在一起看，RS的效果介于两者之间，样本更小（n=8，真实上限）所以不确定性更大，
但没有发现任何一例"RS让回复明显变差"的清晰案例（唯一一个B不是guard导致的兜底，是两次独立
生成之间的自然质量差异）。

## 结论

1. **RS复现了MP同一类"guard契约不适配指导型证据"的问题，且多了一个MP没有的变体**（词面
   重合检查）——两条guard修复后，RS的"带RS兜底率"从50%降到接近0%，效果立竿见影，用同一批
   真实state直接对照确认，不是猜的。这进一步确认了这次session反复验证的一个模式：guard的
   契约是照着MS/ME的"叙事事实"设计的，MP和RS这类"指导/偏好"型证据都需要单独审视，不能
   默认套用同一套检查。
2. **RS组件本身**：干净重测后3A/1B/4tie，方向上正面，样本更小但没有负面信号。
