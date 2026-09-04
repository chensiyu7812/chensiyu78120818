# PM V1.5 V3 Observation 人评与语义修复审计

日期：2026-08-03  
范围：`Observation / Eligibility`，不含 Step1 `worth_opening`、Step2、generator 或回复质量。

## 结论

两份标注总体可靠，不能因独立评审提出两点担忧就把96项整包判废：

- 24项重叠的最终eligibility与subtype均为`24/24`一致；
- owner raw agreement=`22/24`、κ=`.871`；
- goal raw agreement=`23/24`、κ=`.909`；
- boundary与specific increment均为`24/24`、κ=`1.0`。

但V1不能直接训练。人评揭示两个经actual Rank-1与私有构造逐项核实的真实题面缺陷，使
FACTOR_FIT的人类标签不再满足预冻结8/8：MP owner=`12/4`，RS goal=`5/11`。

## 两个真实根因

1. `MP_PREFERENCE`的owner负例构造错误。用户在谈朋友时，回复格式偏好仍属于用户；诚实负例应是
   偏好已撤回、被更新或时间失效，而不是把当前故事主体换成朋友。
2. RS只保证粗策略族一致，没有保证actual Rank-1 card自己的`when_to_use`成立。一般性“欢迎一个建议”
   不能自动使“联系可信任的人”卡合格；“情绪难以命名”也不能使要求explicitly named emotion的卡合格。

## 哪个评审意见不构成重建理由

独立评审在24项overlap上观察到eligible与ineligible的重复filler均值不同。完整审计没有把这一描述直接
当成捷径，而是以真正的四个训练factor为标签，在64项FACTOR_FIT上做counterfactual-grouped五seed OOF。
filler-only BA均值依次为`.526/.527/.541/.534`，全部低于冻结`.65`阻断线。因此这不是训练阻断。
训练模型输入仍应删除已知padding，避免无意义词面进入模型。

## 显式slot语言的正确解释

这批数据用于资格化有限、显式可观察的候选关系：owner/time、goal/function、boundary/burden、specific
increment。它不是复杂用户需求诊断数据。训练未见的同义表述确认只能支持“受控表述迁移”，不能支持
“自然对话隐含语义已经解决”。ESConv/EvoEmo最终实验必须另报同一Observation合同的in-support与
abstain覆盖；范围外能力留给embedding/NLI/LLM challenger或未来研究。

## 已完成的修复与下一步（历史阶段，以下最终结果覆盖本段）

V2已完成零API、零response、零external lockbox的版本化修复和同栈重放：

- 96/96 exact Rank-1 candidate present；
- 96/96 subtype与constructed target匹配；
- RS family 24/24；
- 完整decision surface重复0；
- topic、length、frozen nuisance gate均通过。

V1到V2只有16项公开题面改变（MP 5、RS 11）；其余80项在所有reviewer-visible字段上完全相同并可
carry forward。完成唯一16项delta主审后，合并80+16、复查8/8标签形状，再运行预冻结的transparent、
lexical logistic、BGE-small factor bakeoff。此处仍不训练Step1，也不评价generator。

产物：

- `outputs/pm_v1_5_v3_observation_review_audit_v1/audit_report.json`
- `outputs/pm_v1_5_v3_observation_review_audit_v1/report.html`
- `outputs/pm_v1_5_v3_observation_semantic_repair_delta_v2_candidate/human_review.html`

## 最终执行结果：标签修复成功，Observation确认未通过

16项delta与80项未变化标签已精确合并为96项V2 reference：

- 64项FACTOR_FIT中，每组件每个适用因子严格8 yes/8 no；
- 32项ELIGIBILITY_CONFIRMATION中，每组件严格4 eligible/4 ineligible；
- uncertain=0，未生成Step1 `worth_opening`，未读取回复或外部lockbox。

这证明本次人评与题面修复可用，但不等于Observation已经资格化。

冻结资格赛的结果为：

| 候选 | FIT结果 | 是否读取确认集 |
|---|---|---|
| transparent V1 | 四因子均未全门通过 | 否 |
| lexical logistic | owner、goal通过；boundary、increment失败 | 否 |
| BGE-small hybrid | 四因子均失败 | 否 |
| fixed DeBERTa NLI | 四因子均失败 | 否 |
| transparent independent V2 | FIT四因子通过；一次性确认失败 | 是，仅一次 |

最后一项确认综合BA=`.71875`、recall=`.9375`、specificity=`.50`。虽然BA超过`.70`，但未过预冻结
specificity≥`.60`，因此H-Step2、Step1和generator实验均未授权。

## 真正根因与责任边界

代码审计发现透明V1把`goal`和`specific_increment`与owner/boundary等其他门做`min()`合取，违反四门
独立监督合同。解除错误耦合后，FIT从全候选失败变成四门全过，证明这是实质实现bug。

但未见表述确认显示：goal/function BA=`.96`，owner/time、boundary、increment约为机会水平或默认放行。
因此剩余根因不是Step1、retriever或generator，而是把不同性质的事实都强迫成自由文本语义分类：

- owner/user与候选时间、版本有效性本来应由memory backend元数据确定；
- boundary只在明确可解析时可判，否则应unknown并fail-closed；
- increment应结合候选内容与当前可见内容做受控差异判断，不能只靠topic相似度；
- goal/function是本轮唯一表现出受控未见措辞迁移的语义门。

## 冻结后的下一步

不再补人评、不再扩regex、不再换Qwen/BGE-M3或NLI prompt。下一方法修订必须版本化地把backend-known
owner/time/version字段移入结构化hard gate，并把无法解析的boundary/increment显式记为unknown→off，
同时在ESConv/EvoEmo报告in-support/abstain coverage。该修订不能复用已消费的32项确认集宣告成功；
若继续完整四资源主张，必须在后续全新sealed系统证据中验证，而不能把当前FIT成绩写成泛化成功。
