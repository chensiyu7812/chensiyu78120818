# PM V1.5 有限终局执行路线

状态：`FROZEN ONE-WAY PIPELINE / 2026-08-03`

## 1. 为什么不再循环

从现在起，回复输出只承担两种角色：

1. 在最终训练标签尚未产生前，做一次预注册的 Step2 功能资格门；
2. 在资格门通过后，作为同状态 component-on/off 效应观测，产生 PM 的 `worth_opening` 标签。

不得再按“生成十几条—人评发现问题—改 prompt—再生成十几条”的方式推进。任何已经消费的人工
结果都不会用于继续修改同一阶段的 prompt、阈值、特征或样本选择。

## 2. 单向阶段

### T0：代码与静态合同修复（已于 2026-08-03 完成并冻结，零 API、零新增人评）

必须一次性完成：

- 每组件独立的 response evidence span 与 resource support span；
- 单组件 claim 只允许由自己的 exact resource 授权，资源 union 只做全局安全扫描；
- MP 条件偏好触发、MS 具体正文、ME action+outcome、RS atomic move；
- 复用现有 `project_joint_feasibility`，形成 requested→feasible→realized 链；
- replacement builder 只从冻结 eligible exact surfaces 选单项，并在 multi 中断言非回声和联合可执行；
- 单元测试覆盖最新 28 条暴露的全部反例。

T0 可以反复运行本地确定性测试，因为它不读取新生成回复或人工 outcome；这属于实现修 bug，不是
看考卷调答案。T0 完成后冻结 prompt/schema/validator/builder identity。

T0 冻结结果：32 条调用计划（MP/MS/ME/RS 单项各 6 条、8 条不同 multi）全部通过 exact eligible
lineage、typed surface、非回声、联合可执行与来源不复用静态门；50 项相关回归测试通过。调用计划
SHA256 为 `354a0eef8e9f56f0d7e2fa97b6762b9d9bd5e534d871a6e947a117066ae64964`，冻结清单位于
`outputs/pm_v1_5_v3_replacement_h_step2_plan_v1_candidate/t0_freeze_manifest.json`。截至冻结时 API=0、
新回复=0、外部 lockbox 未读。下一步只能消费该唯一 T1，不得重建或重选。

### T1：唯一一次 replacement H-Step2（32条）

- 24条单组件：MP/MS/ME/RS各6条；
- 8条多组件；
- 单项每组件≥5/6；multi≥6/8；material misuse≤1/32；fabricated recall=0；明确边界违反=0。

这32条只判断 Step2 是否能执行已确认合格的资源，不训练 PM，也不评价 Step1 开关。

若在页面生成前发现 candidate ID、current echo、schema、资源缺失等机器合同错误，builder直接失败，
不消费T1。若T1人工结果已经产生后仍未过门，V1.5停止修改Step2；不追加第二个小包。论文报告完整
四资源Step2未达资格，并保留已通过组件的分层证据。

T1 实际机器门结果（2026-08-03）：32/32调用完成、JSON结构错误0，但只有12/32通过逐组件exact
evidence binding，20条按冻结安全规则回退到M0+R0。单项机器通过为MP 0/6、MS 2/6、ME 1/6、
RS 6/6；multi为3/8。因此预注册的exact binding=100%门已经失败，完整四资源Step2不能晋级，T2
不得启动。现有32条人工页面只可用于描述“语言层面是否做功”和界定RS等局部证据，不能推翻机器门
或触发prompt/schema/validator重修重跑。

T1 人工功能门随后完整消费：单项MP/MS/ME/RS为5/6、3/6、5/6、6/6，multi all-functional为
4/8，material misuse为10/32，20条回退中仅14条可采用。机器与人工43个组件判断的交叉表为
`machine yes/human yes=16`、`machine yes/human no=1`、`machine no/human yes=17`、
`machine no/human no=9`。这说明exact-span机器门确实严重漏报功能使用，但人工证据仍独立确认MS、
multi协调、误用和无资源fallback存在真实问题；不能把T1失败全部归为审计字段。

### T2：EFFECT_FIT（256个状态；本段执行口径已由 V5 ITT 修订）

V5 不再要求新的 T1 人工语义晋级门；只需 typed adapter、确定性 fallback 与 assignment/binding/
identity 机器不变量通过，并在新 outcome 前冻结完整计划。四组件各64个独立用户，每个状态只改变
一个 requested bit，两个冻结generation seed。
客观hard gate失败的状态不生成treatment；语义goal/increment正负面作为Step1输入而不是提前人工
筛掉。Step2未实际执行或发生阻断误用的pair记`unknown`，不能记OFF。

256个状态形成一个盲评包；每项同时呈现两个seed的匿名on/off对，按冻结quality、interaction-and-
grounding risk和cost规则机械派生 ITT requested-value 标签。generator未利用资源、合法fallback、
quality tie/loss或material misuse均保留为请求动作的真实outcome；仅机械实验失效删行。它是训练数据，
不再用来修generator。

若每head无法得到至少8个ON和8个nonpositive独立组，或重复seed方向极不稳定，则停止：结论是固定
资源/生成栈没有提供最低可学习信号，不补造正例。

### T3：训练一次

- 四个独立head；
- 每head 5–7个source-specific透明因子；
- standardized L2 logistic primary；BAAI/BGE-small仅作一次冻结challenger；
- grouped OOF BA≥.65，recall≥.60，specificity≥.60；Brier胜prevalence和transparent rule；
- nuisance-only BA<.65。

若FIT失败，不换模型、不追加数据、不改阈值；直接形成`NOT_LEARNED_ON_FROZEN_FIT`结论。

### T4：FRESH_CONFIRMATION（128个状态，只消费一次）

每head 32个新用户，至少8 ON/8 OFF。模型、特征、阈值在打开标签前已冻结。确认集只决定是否晋级，
不能反向修改模型。任一完整主张所需head失败，则完整四资源主张不晋级；不建立V2确认集。

### T5：SEALED_INTERNAL_TEST、ESConv、EvoEmo和最终系统比较

仅在T4通过后冻结完整16动作runtime。随后一次性运行：

- 128个sealed internal状态，补全MP_PREFERENCE、hard negatives和多组件冲突；
- ESConv：RS与memory-unavailable/off；
- EvoEmo：MP_PROFILE、MS_SESSION、ME_REUSABLE_OUTCOME及in-support coverage；
- always-off、fixed-high-eligible、transparent-rule、cost-matched-fixed、learned-PM；
- 可精确复现时加入legacy V1和raw-session baselines。

所有外部、sealed和最终盲评结果只写论文，不再回流修改PM。

**进入状态（2026-08-04）：** V5.1 FRESH_CONFIRMATION 已按结果前冻结的完整系统门通过。learned PM
相对 fixed-high 的质量 group-bootstrap 95% 下界为 `-0.0430`（门槛 `-0.05`），material-risk 点差
`-0.0078`、95% 上界 `+0.0117`（门槛 `+0.05`），prompt token 减少 `11.07%`，ON 比例
`51.56%`，关键 grounding 事件 `1` 对 `6`。T4 现永久关闭，不允许用该 confirmation 修模型；下一
合法动作仅为冻结当前16动作runtime并执行T5。

**T5冻结状态（2026-08-04）：** 零response静态物化已覆盖sealed internal `128`、ESConv `122`、
EvoEmo p7–p12 `60`与p13–p18 `78`，共`388`个状态。EvoEmo qualification中MP/MS/ME均达到
至少`20`个candidate-present且typed-executable状态，故lockbox仅按静态transport门获准开启；不读取
回复、质量、风险或judge。五策略×双seed按相同action去重为`2,920`次Llama调用，预算代理上界
`$0.8551`。cost-matched动作冻结为sealed=`ME+RS`、ESConv=`M0+R0`、EvoEmo=`MPMSME+R0`，
lockbox直接继承qualification选择。最终quality人评`691`对、GPT-4o独立judge全比较与20%位置反转
计划均已在生成前哈希冻结；任何T5结果不得回流修改上述内容。

## 3. 固定人工工作上限

后续不再创建任意数量的小包。完整主张最多只消费：

- T1：32条Step2功能审核；
- T2：256个FIT状态；
- T4：128个confirmation状态；
- T5：128个sealed状态；
- 预先哈希选择的固定重叠复核，不因结果好坏增减。

后三阶段可以在同一个预生成盲评工具中分区，但confirmation和sealed在前一阶段冻结前不得打开。
最终系统外测审核另按实验协议一次性冻结样本数；不以“再看十几条”方式追补。

## 4. 哪些旧工作保留

不重做Strategy Bank、H-Eligibility reference、V3 640状态蓝图、compiler、query builder、retriever、
外部分区和失败账本。由于Step2 schema/prompt会修复，旧回复不能充当新treatment，但状态、候选和
资源lineage可复用；只重新生成必须受新Step2合同约束的response arms。

## 5. 终点

该路线只有三种终局，不存在无限循环：

1. `FULL_V1_5_PASS`：四head通过确认并完成完整QRC比较；
2. `BOUNDED_COMPONENT_PASS`：部分head或Step2 surface成立，完整主张不成立，按组件报告；
3. `NOT_LEARNED_ON_FROZEN_PROTOCOL`：可靠数据上未超过基线，作为可复现负结果和V2动机。

“修复机器可证的实现bug”允许发生在T0；“看到人工或外部结果后继续调到通过”从T1开始永久禁止。

## 6. 2026-08-04 V5.1 系统级决策修订

T3原先把四个head全部通过统一BA/Brier/leave-family门设为进入T4的唯一条件。V5完整FIT后发现，该门
适合作为逐head机制诊断，却与论文的非对称quality-risk-cost系统estimand不完全一致：冻结的
grouped-OOF learned policy虽然没有任何head通过全部旧门，但相对component-fixed-high取得质量
84胜/81负/347平、material risk 50对103、prompt token均值224.4对250.1，并保持52.3%的非退化
ON比例。

因此旧T3结论`V5_FIT_COMPLETE_NO_HEAD_PROMOTED`永久保留，不改阈值、不补特征、不宣称单head
晋级；同时允许当前冻结完整policy进入一次V5.1系统级FRESH_CONFIRMATION。确认合同位于
`data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json`，判断质量非劣、risk、cost、
transparent-rule比较、ON比例与关键grounding错误。它不再要求四个head逐一成为优秀语义分类器。

该修订只改变研究决策层，不改变V5任何response、label、feature、model、threshold、resource或
executor。V5.1 confirmation通过才进入T5；失败即按本路线的bounded/negative终点报告，不建立第二
份confirmation，也不回到小包盲评或结果后修Llama。

V5.1调用API前另有一次零outcome数据完整性门：actual Rank-1必须由同一冻结retriever物化；FIT与
confirmation不得存在规范化`current state + candidate`重复；反事实group不得跨split或跨fold；
construction enrichment、split身份和topic-prefix长度不得进入模型输入；可见cue不得完美预测标签或
subtype。该门只允许在任何回复/标签产生前修机械构造bug，不授权结果后改题。

## 7. 2026-08-04 V5.2 终局执行修订

V5.1 T5揭示历史出处仍由自由generator决定，V5.2因此按`V15-ARCH-26`更换为backend-locked executor，
并在新的FIT上重新生成、评审和训练。V5.2 FIT没有让四个单head全部通过旧机制门，但冻结的系统OOF
相对fixed-high呈现质量近似保持、risk下降和cost下降的Pareto信号，因此只允许一次新的内容独立系统确认。

旧EFFECT_FIT、FRESH_CONFIRMATION与SEALED_INTERNAL均已暴露，禁止复用。新的确认身份固定为128个
state-candidate单位/64个反事实组，四组件各32；与旧内部split的state、group、可见表面和完整Rank-1
决策面全部零重叠。四策略、四head、`.5`阈值、16动作组合、backend-locked executor、两个hash seed、
512个ON/OFF调用和实现哈希均已在结果前封印。learned-PM请求ON为`65/128`，满足非退化要求。

唯一下一动作是运行已封印的`27j_run_v5_2_confirmation_generation_v1_5.py --run`。执行后只能按冻结合同
制作一次确认评测并给出PASS、BOUNDED或NOT_LEARNED结论；无论结果如何都不得再修模型、executor、
prompt、阈值或样本。ESConv/EvoEmo只作为修复后外部复现实验，不再称未触碰lockbox。
