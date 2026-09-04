# PM V1.5 V3 H-Eligibility 双评审计与局部修复

日期：2026-08-03  
阶段：`V3-P2`  
结论：`ELIGIBILITY REFERENCE V3 FROZEN / OBSERVATION QUALIFICATION NEXT / NOT STEP1 GOLD`

## 1. 两份审核能不能用

能用来证明四门资格口径可稳定执行，但不能把初版128项直接冻结为训练gold。

- 主审128项、独立重叠32项均可解析，ID与冻结binding完整对齐；
- 最终eligibility为31/32一致，raw agreement=`.96875`，Cohen κ=`.9375`；
- 分组件raw agreement：MP=`1.0`、MS=`.875`、ME=`1.0`、RS=`1.0`；
- subtype为32/32一致；具体增量门最难，raw=`.84375`、κ=`.6040`，仍刚过预冻结下限；
- 当前文件来自两个评审代理，不能冒充“已核验身份的人类gold”；论文中的annotator身份需据实说明。

## 2. 为什么高一致仍不能立刻训练

一致性回答“评审是否按同一口径判断”，不回答“出题器是否真的造出了它声称的正负构念”。在人工
标签完成后，construction intent只被用于出题QA，而未被用来选择标签。该关联检查发现14/128错位：

| 逻辑族 | 数量 | 实际问题 |
|---|---:|---|
| MS_ELIGIBLE_MS_DISTINCTION | 4 | 目标candidate只是“先理解压力再规划”，没有提供两种压力的具体区分 |
| MS_WRONG_OWNER_OR_GOAL | 4 | current也被改成friend，导致friend-owned candidate反而owner正确 |
| RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE | 4 | current只指定“一个问题”，card仍额外指定“澄清已出现的感受”，因此并不冗余 |
| RS_ELIGIBLE_RS_SUGGESTION | 2 | shift/sleep状态Top-1取回communication-opening card，功能不符 |

最危险的是MS：一个应全ON的四条族变成全OFF，一个应全OFF的四条族变成全ON，总计仍为16 ON / 16
OFF。若只看类别平衡，会错误宣布数据完美。这次数据质量审计因此要求按semantic family而不只按总数验收。

## 3. 修了什么、没有修什么

修复只改零outcome的表面实现，不读回复、质量胜负、外部lockbox或sealed结果：

- MS distinction生成具体的“情绪不确定性 vs 实务负荷”区分；
- MS wrong-owner让current明确属于用户本人，而candidate明确属于friend；
- MS factual recall显式生成可被回忆的具体时间观察；
- RS suggestion的current function与实际environment/routine卡对齐；
- RS current-request负例完整指定card将提供的同一功能，使“无增量”可观察。

正式环境为
`/home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python`，Python 3.13.2，
并设置`PYTHONNOUSERSITE=1`。旧`distress_build`重建哈希全部作废。

V2机械结果：640/640 exact Rank-1存在；四组件subtype、constructed target、catalog size均160/160；
RS family 160/160；完整Step1 decision surface重复0；API=0，responses=0，external lockbox未读。

## 4. 为什么只复核24项

完整eligibility decision surface定义为：公开对话、当前消息、组件、exact Rank-1候选文本和age。V1→V2
只有24/128项变化（MS 12、RS 12）；其余104项surface hash完全不变，可在修复双评通过后机械继承原标签。
改变项绝不自动继承旧答案，两个评审都复核全部24项。这不是新的零散实验，而是同一H-Eligibility数据集
的版本化bug-fix delta。

## 5. V2 delta 复核后的最终裁决

两份24项复核对最终eligibility为24/24一致，MS和RS各为8 eligible / 4 ineligible；但逐门审计
不能只停在最终标签：

- owner、goal、boundary和subtype均24/24一致；
- `specific_nonredundant_increment`只有20/24一致，raw agreement=`.8333`、κ=`0`；
- 四条wrong-owner MS中，主审把“具体增量”也判为no，副审判为yes。最终采用yes：它确实具体且新增，
  只是属于朋友并且不服务当前目标。四门必须独立，最终仍为ineligible；
- 两位评审都把四条“用户逐字要求一个聚焦问题”的RS判为eligible，但这是共享系统误差：一方面旧
  ranker错误选择了与one-point边界冲突的非低负担profile；另一方面，用户已经完整指定了相同原子动作，
  因而card相对R0没有新增指令。不能用“动作尚未执行”替代“资源是否有增量”。

透明low-burden解析器现已识别`one focused question`，四条exact Rank-1只发生profile级单调修复：
support move不变，改为低负担兼容variant。最终四门为owner=yes、goal=yes、boundary=yes、increment=no，
故仍为ineligible，但理由从“候选违反边界”变为科学上正确的“资源冗余”。无需再请求第三轮人工来投票
覆盖codebook；裁决和具体decision surface均留下机器可复算血缘。

最终唯一reference为128项，MP/MS/ME/RS各16 eligible / 16 ineligible；104项由V1 exact-surface
继承，16项为V2双评完全一致，4项为V2字段独立性裁决，4项为V3 profile修复+冗余裁决。出题意图只作
事后QA（128/128匹配），未作为gold来源。该平衡是受控资格集设计，不可解释为自然开启率。

正式文件：

- `outputs/pm_v1_5_v3_h_eligibility_reference_v3/eligibility_reference.jsonl`
- `outputs/pm_v1_5_v3_h_eligibility_reference_v3/freeze_report.json`

两个附件记录的是两个独立review-agent身份，不能在论文中冒充已核验身份的人类gold。

## 6. 接下来的唯一顺序

1. 在冻结的128项reference上资格化Observation各因子；
2. 再做全新H-Step2资格门；
3. 两层均通过后才生成正式effect clean pairs；
4. 只有H-Effect的质量、风险、成本结果才形成`worth_opening`并训练四个heads。

因此当前仍没有训练PM，也没有评价generator；H-Eligibility已关闭，下一责任层是Observation，随后才是
Step2。修复不改变quality-risk-cost主张，也不缩减MP/MS/ME/RS或16动作。
