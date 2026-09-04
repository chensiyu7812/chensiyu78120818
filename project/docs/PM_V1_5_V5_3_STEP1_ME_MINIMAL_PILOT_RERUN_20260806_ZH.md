# Step1最小试点：ME真实生效后的第一次真实配对测试（2026-08-06，修复后重跑）

`PM_V1_5_V5_3_STEP1_ME_MINIMAL_PILOT_FINDINGS_20260806_ZH.md`记录的是ME组件0/138命中、
无法配对测试的原始结论。`PM_V1_5_V5_3_ME_REGEX_RECALL_FIX_20260806_ZH.md`修复了编译器
召回率和排序层的`me_subtype_hint`断链问题后，真实命中率变成108/138（78%）——这份文档是
修复之后，第一次真的能跑起来的ME真实配对生成+judge打分测试（脚本
`55_step1_me_minimal_pilot_v1_5.py`），跟MP/MS/RS用的是同一套已验证管线（真实judge
instrument、正反序配对、格式修复重试、瞬时错误容错）。

## 真实结果：9个真实state（真实上限，一用户一条，不是抽样）

| 结果 | 数量 |
|---|---|
| A（带ME更好） | 2 |
| B（不带ME更好） | 2 |
| tie | 5 |

9/9都拿到了有效judge判定（过程中遇到2次真实Gemini 503瞬时错误，已被脚本的容错逻辑正确处理，
不影响其余判定）。

## Guard触发情况：带ME一侧8/9 clean，不带ME一侧的残留问题符合预期

| | clean | 兜底 |
|---|---|---|
| 带ME | 8/9 | 1/9（p9，`EVIDENCE_CLAIMED_USED_BUT_NO_WORD_TRACE_IN_REPLY`——跟RS遇到的同一条词面重合检查，这次是ME罕见触发的个例，不是系统性问题：ME证据本身是叙事性事实，这条检查对ME本来就是适用的，只是这次生成器复述得不够贴合原词） |
| 不带ME（M0+R0基线） | 6/9 | 3/9（p11/p14/p17，全部`TRACE_REFERENCES_UNAUTHORIZED_EVIDENCE_ID`——这就是MS试点发现、已经部分修复的M0+R0证据幻觉残留，约33%触发率，这次数字吻合） |

**带ME一侧的高clean率（89%）本身就是一个重要的正面信号**：一旦ME真的有真实、经过验证的
候选可用，生成器能干净地把它整合进回复，不是"勉强塞进去导致质量下降"。

## 真实回复读起来是什么样

抽几个例子直接看："I recall you mentioned going back to therapy last month, and it's
been helping a bit."（p10，带ME）——自然地把过去经验编进当前建议里；"You mentioned that
trying to focus on engaging one-on-one helped a bit, but it was hard to shake the
feeling of being judged"（p12，带ME）——准确复述了原有的正反两面（有用但也有困难），
没有过度简化成纯正面。整体上读起来是合理、自然的"你之前提到过X"式引用，不是生硬粘贴。

## 结论

1. **ME组件现在是真实可用的**，不再是"打不出候选"的空转状态——真实上限9个state全部真实
   跑通，8/9生成clean。
2. **效果信号中性偏正面（2胜2负5平）**，跟MP的模式（4/6 tie，弱信号）相似，不是MS那种
   明确的正面效果，但也没有任何一例显示"带ME让回复明显变差"。
3. **不带ME一侧的残留兜底问题**，已经是已知、已记录、部分修复的M0+R0问题，不是ME特有的新
   问题。
4. 样本仍然很小（9个真实state，是这批数据的真实上限，不是抽样限制）——跟RS一样，统计力度
   弱，但方向上没有负面信号，足以支持"ME可以进入V5.3"这个判断。
