# PM V1.5：两份 Strategy 开放编码人评的最终使用决定

状态：`TWO SETS COMPLETE / VALID DISCOVERY EVIDENCE / NOT PM TRAINING GOLD`

机器审计：
`outputs/pm_v1_5_esconv_strategy_dual_human_analysis_v1/dual_annotation_analysis.json`

## 1. 结论

两份标注不需要二选一，也不能逐字段简单多数投票。正确用法是：

1. 共同结论作为高置信的动作发现证据；
2. 第一份较宽的“关系性/抽象动作”判断用于发现潜在 atomic move；
3. 第二份较保守的“当前实现能否直接成为通用技术”判断用于来源资格；
4. 任一标注报告风险、不确定、高风险领域、事实型内容或真实人生自我披露时，
   该原回复都不能直接成为 LLM Strategy Card 的来源；
5. 这两份数据只建立 codebook 和筛选规则，不产生 PM 的 RS on/off 标签。

经过这一决策，两份人评是有用的，而且足以结束“先验拍脑袋定五类”的阶段；它们还不能
证明最终 Bank 覆盖充分、检索正确或 RS 对回复质量有增益。

## 2. 数据质量

两份均为同一组 72 个唯一 Tier-1 条目，必填字段、动作与风险说明内部完整。逐字段结果：

| 字段 | 第一份正例 | 第二份正例 | 一致率 | Cohen κ |
|---|---:|---:|---:|---:|
| meaningful support action | 62 | 58 | 94.4% | 0.801 |
| mainly information | 6 | 11 | 93.1% | 0.670 |
| mainly self-disclosure | 19 | 19 | 94.4% | 0.857 |
| reusable general technique | 61 | 52 | 87.5% | 0.638 |
| clear risk/boundary problem | 11 true + 1 uncertain | 12 true | 93.1% | 0.785 |

共有 21 条在至少一个字段上分歧。分歧高度集中于：

- 互惠自我介绍、回答个人问题、共享宠物兴趣是否已构成“支持”；
- `reusable` 指“抽象动作经修理后合法”，还是“当前实际执行本身可直接复用”；
- 信息是否包含平台事务，以及混合回复中事实解释是否为主要动作；
- 宗教建议、绝对可用性承诺、疫情陈述、诈骗警示和高风险遗漏的 risk 边界。

这不是随机混乱，而是可通过操作定义解决的构念差异。

## 3. 模型 coder 的地位

Coder A 在当前 72 条中有 33 条 evidence excerpt 不属于目标回复；全部 160 条中有
81 条失配。Coder B 全部 160 条只有 1 条失配。

因此：

- Coder A 整个 pass 不参与多数票、IAA 主结论、全量弱回标或卡片资格；
- 原 A/B 一致性只保留为失败诊断；
- Coder B 可以在冻结 codebook 后做全量候选回标，但只能是 weak proposal；
- 人工页面展示过 A/B 输出，因此 human-vs-B 也不能写成完全独立的模型准确率。

## 4. 冻结后的字段定义

### 4.1 meaningful support action

目标回复必须对当前情绪支持、理解、探索、稳定或有限行动帮助作出可观察贡献。

以下记 false：

- 平台/众包任务操作；
- 纯寒暄和纯告别；
- 离题闲聊；
- 只回答 supporter 自身身份、账户、婚姻等个人问题，但没有推进支持；
- 仅靠“我也……”建立轻微关系、却没有回应当前支持需要的片段。

因此，对 Bank 构造采用第二份较保守的边界。第一份识别出的关系性动作仍可写入质性讨论，
但不据此创建卡。

### 4.2 mainly external information

建议将字段改名为 `mainly_externally_verifiable_information`：

> 回复的主要价值是否依赖需要在对话外核实的事实、资源、法律、医学或机制性陈述。

平台事务另记 `corpus_artifact`，不靠 information 字段承担。建议不是因为有指令性就变成
information；询问是否愿意接收资源也不是具体事实内容。

事实型原文不能进入 Strategy Bank。若保留“征得许可后提供已核验资源”这一 meta-technique，
资源核验与 technique card 必须分开。

### 4.3 mainly self-disclosure

只要支持作用实质依赖 supporter 的身份、地点、关系、疫苗、工作、婚姻、 coping history
或其他生活经历，即记 true；纯礼貌性的 “I am well” 不算实质披露。

这是 LLM Bank 的关键硬排除项。人类 supporter 的披露可能有意义，但 LLM 没有相应真实
经历。把“分享自己的类似经历”写成卡会诱发 `unsupported_personal_claim`。

### 4.4 reusable execution

`reusable` 冻结为：

> 当前可见动作按其实际执行方式，是否已经安全、topic-agnostic、足够自包含，并可直接
> 转写成 LLM technique card。

不再把“如果去掉羞辱、武断、虚假承诺或危险细节后其实是个合法技术”记为 reusable=true。
那个更宽的概念另记 `abstract_move_candidate`，只服务 codebook 发现。

因此，九个 reusability 分歧采用第二份较保守的来源资格。一个风险回复仍可帮助发现
“confront behavior, not person”等抽象动作，但风险原句不计 card source support。

### 4.5 source risk 与论文 risk 分开

`source_risk` 用于建库，范围可以较宽：

- 医疗、法律、财务、DV、诈骗等未经核实或需专业处理的指导；
- 羞辱和人格评判；
- 无依据的确定性预测；
- 虚假保密、永久可用等角色承诺；
- 强加价值观；
- 对明确高风险线索的明显遗漏。

任一标注为 true/uncertain 或明显属于专业高风险领域时，来源 fail-closed。

论文正式 `interaction-and-grounding risk proxy` 仍只有：

1. explicit boundary violation；
2. unsupported personal claim；
3. stale or conflicting evidence use；
4. excessive directiveness。

来源排除不自动成为 PM outcome risk。这样既能安全建库，又不把 V1.5 临时改成临床安全
研究。

## 5. 两份数据怎样合并

### 5.1 高置信来源种子

机械地要求两份同时满足：

- meaningful=true；
- reusable=true；
- risk=false；
- information=false；
- self-disclosure=false；

可得到 28 条高置信 LLM-compatible source seeds。

这 28 条仍只是 codebook/source seeds，不是 28 张卡，也不是最终全部合格来源。诸如家庭
决策、住房或其他高风险上下文仍需经过专业领域/边界排除。

### 5.2 分歧的处理

不再重做整批 72 条。只按上面的冻结规则处理 21 条分歧：

- 支持贡献和当前执行资格采用保守定义；
- substantive self-disclosure 采用机制判断，而非字面是否出现第一人称；
- platform artifact 单列；
- source risk 取 fail-closed；
- 正式 PM risk 只映射到四个冻结维度。

若仍有无法按规则解决的条目，只需对该条做一次看不到 A/B 的独立复核；未解决则从来源
证据排除，不为了凑数量强行裁定。

## 6. 允许和禁止的用途

允许：

- 归纳 bottom-up atomic-move codebook；
- 定义 information/self-disclosure/high-stakes/source-risk 排除规则；
- 为 9,148 条 ESConv train supporter turns 的弱回标提供种子；
- 选择需要人工复核的边界样例；
- 检查现有 top-down 50-card reference 中哪些定义确有数据支持。

禁止：

- 直接训练 PM 的 on/off head；
- 把 58/62 meaningful 或 52/61 reusable 当 ESConv 总体比例；
- 两人同意就自动生成卡；
- 用这 72 条证明 Bank 覆盖 ESConv/EvoEmo；
- 用 source risk 代替最终回复的四项 PM risk；
- 用 BGE cosine 自动裁定两个人的动作短语谁正确。

Tier-1 本来就是分歧/风险富集样本，不能承担总体分布估计。最终 Bank 资格仍由全量 train
回标、独立 dialogue 来源数、卡片审核、检索审核和 R0/RS treatment outcome 决定。

## 7. 对现有五族和 50 张卡的影响

两份开放动作中稳定出现了复述/理解确认、情绪反映、validation、聚焦探索、合作梳理、
轻量步骤、已有资源肯定、建议转换和边界管理。因此原五个行为区域可以保留为覆盖检查
清单。

但数据没有证明：

- 动作天然恰好聚成五个互斥类别；
- 每族应当恰好十张；
- 现有 50 张全部获得至少 20 个独立 dialogue 支持；
- self-disclosure 或 factual information 应进入 LLM Technique Bank。

下一步应从开放短语归并出自然数量的 atomic moves，再与 50-card reference 对齐；不是
从五族出发把每族硬扩成 8–10 个子策略。

