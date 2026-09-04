# PM V1.5 V3 Observation 资格结果与一次性修复路线

日期：2026-08-03  
阶段：`V3-P2 / Observation`  
状态：`V2 FIT PASS / ONE-SHOT CONFIRMATION FAIL / STEP1 NOT TRAINED / STEP2 NOT AUTHORIZED`

## 1. 结论

冻结的128项Eligibility reference已经足以检验当前透明Observation，但当前实现没有及格：

| 审核因子 | BA | Recall | Specificity | 全局门 |
|---|---:|---:|---:|---|
| owner/time/entity | .848 | .982 | .714 | PASS |
| goal/function | .750 | 1.000 | .500 | FAIL |
| boundary/burden | .582 | .925 | .238 | FAIL |
| specific nonredundant increment | .715 | .705 | .725 | PASS |

四因子合取后的Eligibility诊断为BA=`.6484`、recall=`.7188`、specificity=`.5781`。更重要的是，
四个因子的最小反事实方向率依次为`.7143/.5000/.1905/.4750`，全部低于冻结门`.75`。

因此不能开始Step1训练，也不能把问题归给generator。当前责任层是：

`visible state + actual Rank-1 candidate → Observation factors`。

## 2. 根因不是BAAI，也不是样本数量

当前V1是人工透明正则表达式与词面重叠规则，失败集中在四类可解释缺口：

1. `goal_function_fit`把“同主题”过度当作“同功能”，产生24个false positive；尤其无法稳定区分
   MP实际约束与无关profile、MS具体区分与topic-only记录、ME可复用结果与当前非行动目标、RS卡族与
   用户本轮功能。
2. `boundary_burden_compatible`主要识别stop/listen/brief的少量词形，对memory action禁令、未来可用但
   本轮禁用、RS profile档位与已执行边界覆盖不足，21个负例仅识别5个。
3. `specific_increment`混合了候选是否具体、是否新、是否服务当前目标和是否已在当前请求中完整指定；
   当前规则有26个false negative与11个false positive。
4. 规则在同一topic的最小对照上常给相同分数，说明不是阈值问题。继续调`.5`或增加同模板数量不会修复
   表达能力。

BAAI尚未进入这一正式资格结果。本结果只说明透明V1规则不足；不能据此说embedding无用或PM学不会。

## 3. 一次性修复边界

不在当前128项上反复改regex直到通过，也不把修后的同集分数冒充确认结果。当前reference从此承担两种
角色：Observation V1正式失败证据，以及后续候选的训练/开发数据；不再是修复后独立确认集。

下一候选固定为四个低容量factor heads，而不是16分类器：

- 输入仅含可见current/dialogue、component、actual Rank-1 candidate及透明retrieval descriptor；
- 透明V1 factor scores作为可审计基线特征；
- owner/time与boundary增加codebook级结构特征，不允许按state ID、topic ID、logic family或构造intent
  建特征；
- goal/function与specific increment资格赛加入现有BAAI embedding challenger；BAAI只表达state-candidate
  关系，不决定`worth_opening`；
- 每个factor使用强正则logistic，counterfactual group整组进入同一fold；预处理只在train fold拟合；
- 同时报五折grouped OOF、leave-semantic-family-out和最小反事实方向；任一门不过即不冻结；
- 透明V1、lexical-only、BAAI hybrid三者同一split比较，选择一标准误差内最简单候选；禁止看外部
  ESConv/EvoEmo或回复outcome选模型。

这一步是有限的“候选资格语义”，不是复杂用户需求诊断，也不是Step1净效用学习。

## 4. 如何避免继续做人评小包

128项现有reference可用于factor heads的grouped OOF开发与模型选择。修复候选若在严格family-holdout仍能
过门，才用全部128项拟合最终Observation；其独立性限制在论文中如实写为“受控内部reviewer-proxy
qualification”。外部ESConv/EvoEmo只在完整系统冻结后做支持范围验证，不回流选择Observation。

若family-holdout不过，不再追加零散10/16/24项标注或换模型凑通过；V1.5把相应能力限制写为局限，并由
保守Eligibility mask处理。这样既不伪造泛化，也不会再次陷入无限人评。

## 5. 后续顺序

1. 实现并预冻结Observation factor-head训练与group split；
2. 在128项上一次性运行transparent、lexical、BAAI-hybrid资格赛；
3. 过门才拟合最终Observation，并进入全新H-Step2；
4. H-Step2通过后生成effect clean pairs；
5. H-Effect才派生Step1 `worth_opening`，再训练四个component heads与完整16动作。

正式结果：`outputs/pm_v1_5_v3_observation_qualification_v1/report.json`。

## 6. 一次性资格赛结果：三种候选均未通过

预冻结协议后，已在相同128项reference上运行transparent、lexical logistic和BAAI/bge-small hybrid。
没有下载模型；正式Python的CUDA 13.0 build与当前CUDA 12.8 driver不兼容，按预定兼容性规则改为CPU
Transformers，编码结果不变，只影响速度。

BAAI有真实局部增益，但未达到冻结门：

| Candidate / factor | BA | Counterfactual direction | Leave-family-out BA | 结果 |
|---|---:|---:|---:|---|
| lexical / owner | .987 | 1.000 | .826 | factor PASS |
| lexical / goal | .750 | 1.000 | .713 | specificity=.500，FAIL |
| lexical / boundary | .713 | .952 | .523 | family FAIL |
| lexical / increment | .688 | .900 | .536 | BA/family FAIL |
| BGE / owner | .956 | .929 | .813 | factor PASS |
| BGE / goal | .800 | .708 | .756 | direction FAIL |
| BGE / boundary | .774 | .762 | .479 | family FAIL |
| BGE / increment | .725 | .775 | .490 | family FAIL |

更上游的数据门也失败：只使用topic、history scale、component和文本长度等nuisance，四因子的grouped
OOF BA仍为`.875/.771/.897/.714`，全部超过冻结上限`.65`。进一步单变量检查表明topic可预测owner
（约`.84`），current长度可预测boundary（约`.83`）。因此当前reference适合证明旧Observation失败，
不适合宣称任一监督模型学到了语义而不是构造捷径。

正式资格赛：
`outputs/pm_v1_5_v3_observation_factor_bakeoff_v1/qualification_report.json`。

## 7. 下一步改为一次完整的因子正交数据修复

不再运行pairwise补丁、阈值校准或更大embedding。下一批必须在生成前满足：

- 先冻结factor×component applicability matrix；例如RS的owner/time是结构性恒真，不人为生成假负例，
  由透明规则处理并从学习与BA分母排除；
- 每个真正可变化的factor×component格子至少8正/8负，四因子在共同适用范围内两两相关接近0；
  最终eligibility平衡不能代替适用格子的单因子平衡；
- topic、history scale、component、公开前缀及文本长度对每个factor的nuisance-probe BA均<`.65`；
- 每个可学习语义条件同时存在跨topic、跨scale和反长度方向的最小对照；
- actual Rank-1重新物化后才审核；construction intent仍不作gold；
- 只形成一个完整版本化H-Observation/Eligibility包及固定overlap，不再拆成10/16/24条小包。

在该静态门通过前，禁止再次运行embedding bakeoff。这样修的是训练数据可识别性，而不是把失败模型调到
当前题面上及格。

## 8. 因子学习与最终Eligibility确认必须拆成两条数据轨

四个独立factor若各自50/50且彼此正交，四门全过的自然比例只有`1/16`。因此同一小数据集不可能同时
满足“四因子独立平衡”和“最终Eligibility 50/50”，过去强行同时追求两者正是标签相关与捷径的来源。

新合同将其拆开：

- `FACTOR_FIT`：64 states。MP/MS/ME各用一次完整`2^4`设计，RS的owner结构恒真，其他三门用两次
  `2^3`；每个真正学习格子恰好8正/8负。最终Eligibility比例不作门。
- `ELIGIBILITY_CONFIRMATION`：另32 states，每组件4 eligible/4 ineligible，只在factor model完全冻结后
  确认四门合取；禁止参与模型或阈值选择。
- 两轨合并为一个96项主审核包与24项固定overlap，不再拆零散小包。

合同：`data/pm_v1_5_contracts/v3_observation_factor_orthogonal_dataset_v1.json`。

## 9. 正交私有蓝图已完成，但尚未开放人评

新协议不复用旧蓝图的 `reason -> requirements` 间接映射。代码审计发现旧实现只有
`REDUNDANT` 分支真正执行，owner、stale、goal、boundary、increment 的其余修改位于提前
`return` 之后，属于不可达代码。旧128项仍保留为真实失败诊断，不用修过的源码反写旧哈希或旧标签。

新蓝图直接为每一行写入四个私有factor bit及相应构造指令，静态结果为：

- 96个独立用户与96个unique decision surfaces；
- FACTOR_FIT 64项，ELIGIBILITY_CONFIRMATION 32项；
- MP/MS/ME四门、RS三个可学习门均严格8正/8负；RS owner/time恒真且不作为学习目标；
- 每组件可学习factor两两最大`|phi|=0`；
- topic、history scale、prefix与预定长度方向在私有矩阵层面对每个可学习factor均条件平衡；
- API=0、人评=0、外部lockbox读取=0、response generation=0。

这只通过了“题目矩阵”门，不能证明自然语言实现仍保持平衡。下一步必须先实现96个可见状态、用同一正式
compiler/retriever绑定actual Rank-1，再运行公开文本的长度与nuisance probe。全部通过后才允许渲染一次
96+24审核包。

机器证据：
`outputs/pm_v1_5_v3_observation_orthogonal_blueprint_v1/audit_report.json`。

## 10. 可见实现、同栈Rank-1与最终pre-human gate已通过

为防止模型只背正负句式，FACTOR_FIT中每个factor×label使用4个表面族、每族2例；独立
ELIGIBILITY_CONFIRMATION只使用训练轨未见的另外4个表面族。随后重新运行正式memory compiler、同一
80-card Strategy Bank和同一pre-PM Rank-1检索器，得到：

- 96/96 candidate present，96/96 subtype match，96/96 constructed target成为actual Rank-1；
- RS 24/24实际strategy family与目标family一致；
- 96个完整`visible state + component + actual candidate`决策面零重复；
- topic-only与length-only对所有factor的BA均`.500`；
- 冻结综合nuisance probe BA为owner`.500`、goal`.4969`、boundary`.4875`、increment`.5031`；
- retrieval descriptor单独最高BA约`.546`，未形成近似标签代理；
- 每个可学习factor都有8条跨topic、跨scale边，并同时覆盖positive-longer和negative-longer。

因此单一审核包现已开放：96项primary（每组件24；64 factor fit + 32 frozen confirmation）和24项固定独立
overlap（每组件6）。页面不显示track、private bit、topic/scale/prefix/length元数据、rank score、回复或
outcome；RS owner/time按共享Bank结构真值固定，不让评审伪造不存在的owner负例。该包仍不产生Step1
`worth_opening`标签。

静态门：`outputs/pm_v1_5_v3_observation_orthogonal_pre_human_gate_v1/static_gate_report.json`。  
主评页面：`outputs/pm_v1_5_v3_observation_review_candidate_v1/human_review.html`。  
独立重叠页面：`outputs/pm_v1_5_v3_observation_review_candidate_v1/independent_overlap_review.html`。

## 11. V2 delta合并与冻结reference

16项语义修复主评完成后，与80项公开decision surface完全未变化的V1标签精确合并。冻结结果：96项完整，
FACTOR_FIT每适用格8/8，ELIGIBILITY_CONFIRMATION每组件4/4，uncertain=0。reference SHA256为
`e1d368c4a8f2576515d8e685531050822416d384b5379c7edb0a18400757dc6a`。

产物：`outputs/pm_v1_5_v3_observation_human_reference_v2/reference_report.json`。

## 12. BGE与NLI为什么没有解决

64项FIT上的冻结结果：lexical通过owner与goal，但boundary的leave-family-out BA=`.50`、increment全局
BA=`.625`；BGE-small四项均失败。它能表达话题相似，不稳定表达否定、时效、边界与“已说过/未说过”
关系。随后按早期合同预授权的固定DeBERTa NLI也未过：四因子BA约`.604/.672/.609/.703`，确认集仍未读取。

这不是继续换模型的理由，而是证实BAAI/NLI都不能替项目定义语义门。两者均冻结为负结果。

## 13. 因子独立实现bug与有效修复

透明V1在代码中把goal、specific increment与owner/boundary合取，和正交数据的独立factor定义冲突。
V2按门独立计算。首次实现又把`leaves room for`误当成`leave ... out`；该无效运行保留，修正词边界与
显式许可优先级后，同一冻结FIT结果为：

| Factor | BA | Recall | Specificity | Direction | Worst family BA |
|---|---:|---:|---:|---:|---:|
| owner/time/entity | .9375 | 1.000 | .875 | .875 | .750 |
| goal/function | .9375 | .9375 | .9375 | .875 | .9375 |
| boundary/burden | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| specific increment | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 14. 一次性确认失败与方法范围修订

FIT全门通过后才读取32项确认一次。结果：

| Factor | Confirmation BA | Recall | Specificity |
|---|---:|---:|---:|
| owner/time/entity | .500 | 1.000 | .000 |
| goal/function | .960 | .920 | 1.000 |
| boundary/burden | .481 | .962 | .000 |
| specific increment | .500 | 1.000 | .000 |
| 四门合取Eligibility | .71875 | .9375 | .500 |

综合specificity未过冻结`.60`，所以不能进入H-Step2。确认揭示的是规则只背会FIT措辞，而不是PM或
generator失败。后续不能再按确认错误扩写regex；owner/user、时间与版本有效性应改由backend结构化元数据
硬判，无法确认的自然语言boundary/increment应`unknown→off`并报告coverage。goal/function保留为有限
显式语义输入。确认集已消费，禁止再作为修订后的独立成功证据。

最终产物：

- `outputs/pm_v1_5_v3_observation_orthogonal_bakeoff_v2/qualification_report.json`
- `outputs/pm_v1_5_v3_observation_nli_recovery_v1/qualification_report.json`
- `outputs/pm_v1_5_v3_observation_independence_repair_v2/qualification_report.json`

## 15. 结构化责任边界已经落地，但尚未宣告通过

`v3_observation_structured_hard_gate_amendment_v1.json`已经冻结，并在
`v1_5_observation_contract.py`实现V3入口：候选presence、owner一致性、active/superseded状态由后端
元数据确定；goal/function保留为有限语义判断；boundary和specific increment输出
`allow/deny/unknown`，unknown运行时OFF但不会被写成监督negative。该接口明确保持MP/MS/ME/RS和16个
合法动作，也明确Eligibility不是Step1 gold。

这只是修复了任务责任边界和实现接口，不等于已经通过新资格赛。已消费的32条确认不会再用于选择或
宣告V3通过；`observation_structured_system_qualification_passed`当前仍为false。下一步只能使用带真实
owner/version元数据的新版本化系统证据验证coverage与错误率，通过后才能进入H-Step2。
