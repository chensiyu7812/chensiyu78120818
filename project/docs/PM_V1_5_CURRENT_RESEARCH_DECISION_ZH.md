# PM-v1.5 当前研究判断与最短发表路线

状态：`ACTIVE RESEARCH DECISION / 2026-07-28`

> 2026-07-29 起，活跃规范已收敛到
> `docs/PM_V1_5_FINAL_RESEARCH_PLAN_ZH.md`。本文保留为形成该方案前的研究判断，
> 若有冲突以最终方案及其机器可读合同为准。

## 1. 现在到底到了哪里

| 层级 | 当前证据 | 判断 |
|---|---|---|
| 可见输入与数据隔离 | 新 auxiliary/test 不再暴露 ESConv `situation`；RS pilot 进一步主动省略来源不够强的合成 session summary | 可用于 train-only 机制试验 |
| RS 资源本身 | Strategy Bank V2 五卡人评 5/5 批准、置信度均为 4；原始 supporter examples 不进 prompt | 可用于 train-only pilot，不等于正式证明有效 |
| R0/RS treatment | 49 个状态、24 个独立 user groups、98 个同栈调用已全部完成；两臂只差一张 technique card | 输入、配对、完成状态和 token usage 已复核，可进入正式测量 |
| Judge | 旧多任务/composite judge 资格赛失败；新的 pairwise quality + pointwise atomic risk instrument 已冻结，10-call clear-control dry-run 已完成 | 旧标签不可当 gold；新 instrument 尚待真实 control 资格检查 |
| PM 可学习性 | 尚未产生新的 clean pair effect label，也未拟合新 head | 完全未知，不能说已学会 |
| 最终 quality/risk/cost | 指标与 margin 已预先定义，但没有新 evaluation outcome | 完全未知，不能写结果性主张 |

因此，新的 V1.5 不是“已经从根本上成功”，而是第一次把“PM 可能及格”的必要条件
整理正确了。真正的生死点仍是接下来的 treatment uptake 与 grouped-OOF learnability。

## 2. 旧 V1.5 为什么失败

旧路线同时混入了五个不同问题：

1. 用 5 类 support mode、16 actions 和多个因子描述“用户需要什么”，目标远大于约
   40 个独立锚点能支持的容量；多个正类 recall 为 0。
2. 把通用 embedding 的相似度误当成需求/资源效用信号。BGE 可以表示或检索相似文本，
   不能直接回答“开这个资源是否让本回合回复更好”。
3. component outcome 不够干净：资源、prompt、judge、旧 action 体系一起变化，PM
   很难知道该学哪一个边际效应。
4. 旧 judge 同时评多个含混维度，并保留 weighted composite。已测候选对 AB/BA
   顺序敏感，人工方向一致率也有限；这些标签不能当无噪声 gold。
5. 部分 ESConv adapter 把 corpus-level `situation` 放进运行时 summary，形成部署时
   不可获得的捷径；重复 action/judge 行又容易虚增样本量。

“PM 学不出来”不是一个单一模型失败，而是 treatment、target、measurement 和数据
边界同时不清楚。

## 3. 新 V1.5 改了什么，又没有承诺什么

新路线只先问 RS 一个组件：

> 对同一个可见状态、同一个 Llama generator、同一个 system prompt、seed 和输出
> 上限，仅增加一条经人评批准的 technique guidance，回复是否发生可辨识且适切的变化？

若答案是否定的，立刻停止：这说明 RS treatment 没有 uptake，不能靠换 BAAI、扩卡、
调 judge 或训练 PM 挽救。

若 uptake 成立，才拟合一个低容量 RS component-effect head，预测“这个状态开 RS
是否比关 RS 更可能有益”。这才是 V1.5 唯一需要学习的核心参数。

V1.5 不主张：

- 诊断用户的心理状态；
- 学会完整五类 support mode；
- 四个 memory/strategy 组件全部最优；
- 临床安全或真实用户长期获益；
- 一个统一分数证明 quality、risk、cost 联合最优。

## 4. quality、risk、cost 到底是什么

### Quality

即时回复质量的主结果是匿名 R0/RS pairwise preference：

`NetWin = (wins - losses) / (wins + losses + ties)`

Judge 必须分别显示：

- 是否符合用户当前明确请求/边界；
- 情绪回应是否贴合；
- 对本回合是否有直接帮助；
- 是否清晰自然；
- 哪一项是最终偏好的决定理由，以及两边逐字 excerpt。

正序/反序结论指向不同 underlying response 时解析为 `tie`；任一顺序
`insufficient` 时解析为 `abstain`。它仍然是 LLM-defined proxy，不伪称 human
preference。

### Risk

论文统一称 `interaction-and-grounding risk proxy`，不是笼统 safety，也不是只指
memory misuse。四项分别报告：

1. 违反用户逐字可引用的互动边界；
2. 编造或无证据强化个人事实/原因/偏好/历史；
3. 在 memory-on 时误用陈旧或冲突证据；
4. 过早建议、多任务堆叠或过度结构化。

本轮 RS 不使用 memory，所以第 3 项是 N/A，不是“零风险”。第 1、2、4 项才是
RS pilot 的主要 risk。每个 violation 要求 response excerpt、evidence excerpt 和
0–3 severity；四项不求和、不平均。

### Cost

主成本是 provider 实际记录的 generator input tokens；retrieval calls、进入 prompt
的卡数和 output tokens 分开报告。不能把 input-token 节省写成总费用、延迟或能耗
必然下降。

所谓“平衡”不是三项加权：

1. quality 先满足非劣；
2. 每项 risk 分别不明显恶化；
3. 在满足前两项的 policy 中选 input tokens 更低者；
4. PM 本身还必须在 held-out user groups 上胜 constant prior，并同时做出 on/off。

## 5. 数据如何才算可用

数据角色必须分清：

- 历史 SupportNeed 人评：只用于解释旧失败和研究特征，不作为新 PM 的 gold target；
- 当前 49-state clean pairs：只用于 train-only treatment/learnability pilot；
- Judge verdict：带不确定性的 proxy label，不是事实标签；
- final evaluation：只能来自方法、prompt、threshold 固定后仍未参与选择的 user groups。

所有切分以 dialogue/user group 为单位。AB/BA、多个 judge、多个 actions 或同义改写
都不增加 ESS。若最后没有未参与方法选择的新 groups，只能写 exploratory，
不能伪称 confirmatory。

## 6. BAAI 和 RAG 的位置

BAAI/BGE 只可以做表示或 Strategy Bank 候选检索，不能直接决定“用户需要什么”或
“资源一定有用”。本轮最小 pilot 甚至不需要 BGE：

1. 逐字显式边界先确定 eligible family；
2. 在 eligible technique-only 子集内用同一个 lexical retriever 排序；
3. PM 决定开 RS 后才把同一个 retriever 的结果放进 prompt；
4. learned、rule、always-off、high-resource fixed 都必须复用完全相同的 Bank、
   query builder、eligible filter、retriever、prompt 和 generator。

若以后加 BGE，它只能替换所有 policy 共用的检索器，并先用任务内 retrieval
relevance 做校准；不能只给 learned PM 一套更好的 RAG。

## 7. 对另一份分析的取舍

应采纳：

- 失败主因不是 BAAI 型号；
- 需求语义、资源机会、component effect 必须分开；
- quality/risk/cost 不合成；
- judge 分岗位、AB/BA、分歧不硬造标签；
- 先测 treatment uptake，再训练 PM。

不作为 V1.5 前置：

- 五卡先扩成 10–20 卡：五卡现已 5/5 人评批准，一卡/家族足以测 technique
  treatment；只有 uptake 因覆盖不足失败时才扩；
- 先实现 SupportNeed zero-shot 两层模型：显式 boundary 已足够完成第一轮 RS
  clean-pair；更细诊断留给 V2；
- 先清空所有旧 composite 代码：活跃 MVP 路径已使用独立 metric registry，旧
  runner 保留为历史资产且不得进入新结果；
- 把 evidence/risk 全交给 deterministic code：prompt 内容、来源、token 和 excerpt
  可由代码验证，但“某个人格结论是否被文本支持”仍是语义判断，只能用带引用的
  LLM/human proxy；
- 因 ESConv 原生标签互信息低就停用 ESConv：这只证明
  `problem_type/emotion_type -> strategy` 不是好监督标签，不证明可见 dialogue
  不能作为状态或评测素材；
- 先深挖 EvoEmo：这是 memory 外部泛化问题，非 RS-first V1.5 的发布前置。

## 8. 接下来只走这五步

1. 为当前冻结的 judge instrument 运行 10 个真实 clear-control 调用；schema、
   literal excerpt、3 个 quality AB/BA 和 4 个 atomic-risk 控制必须全部通过；
2. control 通过后按可恢复方式运行 49 对的 98 个 quality AB/BA 调用和 98 个
   pointwise risk 调用；
3. 检查 uptake、AB/BA consistency/coverage 和每个 family/cue 的可辨识性；
4. 只有测量门通过且正/非正独立 group 足够，才生成 effect targets 并做
   user-group OOF head；
5. RS 管线成立后以同一方法加入最小的一个长期 memory source；只有至少一个
   memory component head 通过，才进入 EvoEmo 长期外部评测。

这条路线允许三种诚实结论：RS treatment 不成立；RS 有效但 PM 未学会；RS 有效且
PM 学到最低可泛化信号。只有第三种支持“可发表的及格 PM”主张。
