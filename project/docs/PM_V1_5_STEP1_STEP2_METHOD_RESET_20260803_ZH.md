# PM V1.5 Step1/Step2 能力边界与方法重置

状态：`V4 DESIGN FROZEN / IMPLEMENTATION NOT STARTED / ZERO API`

## 技术结论

当前不能说“PM 训练失败”，也不能说“Step1 已经做到极限”。当前 V3 正式的四个
`worth_opening` head 尚未训练；已经完成的是候选构造、检索、Eligibility、部分 Observation、
16动作投影与多轮 Step2 开发。最新32项结果把主要阻断定位在 Step2 和固定8B generator：机器
exact-span仅12/32调用通过；人工功能上MP/MS/ME/RS单项为5/6、3/6、5/6、6/6，multi为4/8，
material misuse为10/32，无资源回退只有14/20可采用。

因此下一步不是增加PM标签，也不是换embedding，而是进行一次方法级分责：Step1只输出四个资源
bit及候选ID；Step2由后端确定性编译资源的时态、owner、用途和禁区；generator只写回复，不再
选择资源、不再复制审计span、不再写自由reason。只有这个固定Step2通过一批未见确认后，才允许
生成完整效应数据并正式训练Step1。

## 当前每一层到底做到哪里

| 层 | 当前证据 | 能主张什么 | 不能主张什么 | 状态 |
|---|---|---|---|---|
| 候选构造/检索 | 同一compiler、exact Rank-1、冻结eligible reference | 能稳定提供可审计候选 | 已找到最优item | 可复用 |
| Eligibility | 128项每组件16/16；最终decision overlap 24/24 | 能以明确四门筛除显然错误候选 | 值得为它付成本开启 | 可复用的前置证据 |
| Observation | 受控FIT可高分，但一次未见表面确认只有goal保持BA .96；owner/boundary/increment各约.5 | 对显式、结构化线索有有限能力 | 复杂隐含语义理解 | 边界已知，后端事实改硬门 |
| 旧opportunity proxy | MP/MS/ME BA约.703/.895/.625；formal pass为0 | 特征方向与简单applicability可表达 | QRC净效用head已学会 | 历史诊断 |
| 旧效应head | grouped OOF BA：RS .622、MP .503、MS .468、ME .445 | 首轮效应学习方案未过门 | 当前V3 Step1能力上限 | 失败baseline |
| RS same-bank机会head | H2 BA约.85、recall约.94、specificity约.76 | 冻结80-card Bank下能学基本RS机会 | RS item ranking、真实净收益或外部泛化已解决 | 有效开发基线 |
| 当前V3正式Step1 | 尚未获得合格Step2下的完整FIT labels | 无 | 任何正式四head准确率 | 未训练 |
| 当前Step2 | 单项MP/MS/ME/RS 5/6、3/6、5/6、6/6；multi 4/8；misuse 10/32 | RS原子动作和部分单记忆可执行 | 完整四资源安全可执行 | 未通过 |
| 最终QRC系统 | 尚未运行 | 无 | PM已找到质量-risk-cost平衡 | 未开始 |

## 为什么记忆看起来比RS更难

检索一条正确记忆本身并不难。困难来自旧Step2把四件事合并给同一个8B generator：理解记忆的
时间与owner、决定怎样使用、写自然回复、再自报使用证据。RS卡直接给出一个动作，因而单项10/10
做功；记忆是证据而非动作，必须再进行安全迁移：

- MP偏好通常应隐式改变格式，不能为了“看见做功”而复述偏好；
- MP资料只能约束现实可行性，不能泄露内部profile标签或说错owner；
- MS是过去会话证据，必须区分“直接回答历史事实”与“核对现在是否仍成立”；
- ME必须包含过去action+outcome，并作为当前可拒绝选项，而不是保证过去方法仍有效；
- multi要求这些贡献在同一条回复中不冲突、不过载、不重复。

因此最新失败不是“memory retriever不会调用”，而是“资源证据到回复的执行接口不稳定”。同时6/20
无资源fallback也不可采用，说明固定8B generator本身会虚构历史或无依据状态；PM无法靠开关修复
一个即使资源关闭仍会幻觉的generator。

## 历次方案为什么没有累积成正式训练

### 旧V1.0：动作、资源和域一起变化

直接或近似16分类、高维HGB、小量独立group以及训练/外测使用不同RAG或memory内容，使模型可以
学习动作prior、数据域或模板，而不是组件边际价值。旧数字仅作历史baseline，不能迁移到当前栈。

### Proxy阶段：把“可用”误当“有净收益”

构造式applicability标签产生过漂亮分数，但没有clean component treatment。它只证明简单条件可被
回放，不能证明资源在固定generator下改善回复。

### 单次效应阶段：把generator噪声写入PM标签

旧四head直接使用单次回复胜负或不稳定Step2结果；相同资源可能因随机措辞反转，Step2空转又被当成
OFF。结果是MP/MS/ME OOF接近随机，RS也未过门。

### 自报trace阶段：把审计写作当资源能力

generator被要求输出use/ignore、reason、resource span和response span。它会真实使用资源却抄错span，
也会复制历史后给无关建议却自称applied。该接口既低召回又不能证明功能。

### V3 typed+exact span：修了串线，再次引入旧负担

V3修正了eligible候选、typed plan、component-specific provenance与joint projection，但为了阻止跨组件
借证据，又要求8B复制两段精确span。43个组件中machine/human交叉为16双通过、1机器误放、17机器
误杀、9双失败。机器门确实过严，但人工仍发现MS、multi、misuse和fallback真实失败，不能把全部
失败归为审计器。

## Step1 的正确任务与有限能力边界

### Step1 应做什么

Step1只回答四个问题：在当前状态、当前exact candidate和固定Step2下，分别开启MP、MS、ME、RS
是否有正的预期净价值。四个概率经过联合可执行与预算投影形成16个合法动作之一；`M0+R0`始终合法。

它不挑item、不解释记忆、不写回复、不证明一次随机生成一定更好。

### V1.5合理期待

V1.5应能学习：明确建议/倾听/停止线索、后端owner/time、粗粒度目标匹配、是否重复当前内容、检索
强度、候选年龄、token成本以及简单反事实变化。它只需在独立确认上达到BA≥.65、recall≥.60、
specificity≥.60并胜过prevalence与透明rule的Brier；不是追求90%或复杂语义理解。

### V1.5明确做不到

不主张隐含心理需求诊断、讽刺/间接社会意义、自由文本中的复杂多实体时序关系、任意新memory schema
零样本迁移或每次回复必赢。超出训练支持范围时应abstain/off，并报告coverage。

### 何时才叫“已经最大化”

只有在固定V4 Step2通过后，完整64 FIT/head数据生成完毕，透明L2 primary与一个冻结BGE challenger
各训练一次，并一次性消费32 confirmation/head，才能判断Step1是否达到本方法上限。确认失败后不得换
特征、阈值或追加样本；那时才能诚实写为V1.5有限语义上限。当前尚未到这一步。

## V4 Step2 根治架构

### 后端负责资源事实，不让generator自证

每个资源在进入prompt前由运行时绑定`resource_id/owner/time/source_type/allowed_claim/forbidden_claim`。
generator看不到内部标签，也不输出use/ignore、ID、span或reason。

### 四个确定性adapter

1. MP_PREFERENCE：条件触发时变成格式与负担约束；未触发不测。
2. MP_PROFILE：非冗余事实只作静默可行性约束，禁止逐字暴露`Stable ... Constraint`。
3. MS_SESSION：事实回忆题直接答；其他状态固定为“此前记录X，这次仍是这样吗？”式暂定连续性。
4. ME_REUSABLE_OUTCOME：固定为“过去做X产生Y；如果合适，可把X作为现在一个可拒绝选项”。
5. RS_ATOMIC_MOVE：严格只执行卡片允许的一个动作。

### Generator与fallback

Generator只把已经编译的约束和证据写成自然回复。后处理只检查可机器证明的内部标签、无授权过去
归因、owner冲突、RS动作数、显式边界和高风险禁区。失败时使用确定性的context-only安全短句，不再
进行第二次自由LLM回退；因此fallback不可能虚构“上次”或私人记忆。

## 数据怎样支持“平时学习—外部考卷”

V4不重建RAG、不更换memory compiler、不读取EvoEmo lockbox。复用同一资源ontology和运行机制：

- 内部受控域训练四组件、hard negatives、MP_PREFERENCE、multi与16动作；
- ESConv只检验其自然覆盖的RS和memory-unavailable/off；
- EvoEmo只检验MP_PROFILE、MS_SESSION、ME_REUSABLE_OUTCOME及in-support coverage；
- 外部没有覆盖的能力只在内部受控实验中主张，并明确synthetic限制；
- 外部内容从不进入内部训练memory，训练与外测只共享schema/compiler/retriever/adapter，不共享用户内容。

正式效应数据仍为每head 64 FIT、32 confirmation、32 sealed，状态以独立用户为group，每个contrast
两个冻结seed。无效Step2记unknown而非OFF。质量用“是否足以改变采用决定”的pairwise偏好，risk用
interaction-and-grounding material类别，cost用真实token/call；训练标签由预注册净价值规则派生。

## 唯一后续路线

1. 零API实现V4 typed adapters、response-only generator接口和deterministic fallback。
2. 用历史失败例做单元/回归测试，冻结代码身份。
3. 消费一次全新32项Step2 confirmation；其中8项做固定overlap。不得按结果修复。
4. 通过后一次性生成256个FIT states并正式训练四head；失败则停止。
5. 一次训练透明L2 primary与一个冻结BGE challenger。
6. 一次性消费128 confirmation；通过后才冻结并运行sealed、ESConv、EvoEmo与最终QRC baselines。

该路线不保证一定得到优秀PM；它保证下一次失败可以明确归因，并且只有一个终点，不再出现相同失败
方案换名字重跑。
