# W9：P2R learning blueprint独立只读复核（2026-08-07）

复核对象：Leader commit `bb1fead`（Build V5.3 P2R learning blueprint）。**只读复核，未修改
任何Leader文件**（`v1_5_v5_3_p2r_learning_blueprint.py`、`78k`/`78l`/`79l`脚本及其output）。
不训练、不调用API、不生成回复、不冻结split。

脚本：`scripts/v1_5/80w_audit_p2r_learning_blueprint_v1_5.py`
output：`outputs/pm_v1_5_v5_3_p2r_learning_blueprint_worker_review_v1/report.json`

## 总体结论

**结构性声明基本属实，可独立复现**；发现一个真实、范围明确的问题（4/12 interaction的
候选话题不一致），以及Leader自己的`candidate_family_alignment`诊断指标本身有两处会
制造假阳性的逻辑问题（跟数据构造本身无关）。**没有发现会推翻整体结论的问题**，但建议
在split冻结前处理下面的Category A项，或至少把受影响的4个interaction明确排除在
"四组件联合效应"主张之外。

## 独立复现：结构性数字完全一致

重新执行`audit_blueprint_rows()`（不是重跑`79l`脚本本身，是独立import同一个Leader
函数库对同一份output数据重新计算）：`row_count=868`、`unique_counterfactual_
groups=381`、`unique_user_clusters=44`、`critical_failures`全部为0、
`full_four_candidate_interactions=12`——与Leader `79l`报告的数字**逐项一致**。

## 逐项复核结果（12问，标注 A=阻塞正式effect / B=需披露不阻塞 / C=审计器误报）

**1. 严格过去/owner/version/time**——**通过**。`is_active_strict_past()`同时检查
`created_session<current_session_index`、`valid_from_session`、`valid_until_session`；
`future_or_current_candidates=0`（独立复算一致）。额外抽查：5条发生过版本更新的MP字段，
**0条**MP行选中了已被superseded的旧版本作为Rank-1。

**2. MP结构化选择，不依赖用户复述隐藏field_value**——**通过**。`_structured_mp_rank()`
用`field_type`+`topic_key`（应用范围描述）匹配当前话语，不用`field_value`。抽查真实样本：
current_user_text说"...realistic for my name and the way I am addressed"（泛化范围描述），
真实检索到的候选是"Name: goes by a nickname with close friends"——具体field_value从未
出现在当前消息里，确认不是W7R那种"answer已经在题干里"的循环。

**3. MS真实用BGE-M3对完整严格过去目录排序**——**通过**。`report.json`的
`formal_ms_retriever`字段显示`BGE_M3_COSINE_FULL_CAUSAL_POOL`；全部MS行的
`retrieval_method`均为该值（无lexical debug fallback）；抽查真实semantic
score（0.699/margin 0.206，真实浮点值非占位符）；候选池规模15–31条（非单候选）。

**4. ME严格exact Rank-1 compile-or-off，不升级Rank-2**——**通过**。243条ME行中
`rank1_compiler_valid=False`有32条，**0条**同时满足`candidate_present=True`；这些行
的`execution_candidate_text`确认为`None`，`topk_candidate_ids`仍保留完整排序列表
（用于descriptor审计）但不进入执行——机制正确。

**5. RS使用冻结6-card Bank，候选层只做机械安全判断**——**通过**。`materialize_rs_
state()`直接复用W7/W7R已验证、未改动的`rs_mechanical_candidate_pool()`/
`rs_shared_candidate_top1()`。

**6. Step1特征全部运行时可观察，无gold/condition/outcome/答案ID捷径**——**通过**。
`P2RModelInput.no_gold_or_identity_fields`是Pydantic强制校验器（不是约定，是无法构造
违规行的硬约束）；`model_input_gold_leaks=0`（两次审计一致）。额外核查`candidate_role_*`
这类字段：来自候选自身的构念类型（是否已解决/是否compiler-valid的REUSABLE_OUTCOME等），
跟`me_subtype_hint`是同一类"候选内在属性"特征，不是"这个state该不该开"的构造意图，判定
合法。

**7. counterfactual_group_id与split_group_key分责**——**通过**。`split_group_key`
是`catalog_user_id`的纯函数（跟component/family/condition完全无关），并有校验器强制
`split_group_key == make_split_group_key(user_id)`，结构上不可能违反"同用户同split"；
`split_leak_groups=0`。

**8. 跨用户重复/模板标签捷径/family-condition锁定/外部同源**——**通过**。独立复算：
`accidental_cross_user_exact_surface_duplicate_groups=0`，`families_without_
condition_crossing=0`，外部exact/8-gram overlap均为0（868条内部surface vs
48,912条外部）。12组"重复"全部是同一interaction的4个组件视图（intentional，非跨用户）。

**9. 12个interaction是否真实四组件、无为凑数重新检索**——**结构上通过，但发现真实
话题不一致问题（见下）**。代码层面`assemble_interaction_row()`明确禁止重排（复制已
独立物化的候选，不重新检索），核对无误。**但新增检查发现：4/12 interaction中，ME或
MS的候选虽然真实存在、compiler-valid、结构正确，却是**另一个话题**的内容**——
"relocation abroad"这个interaction的ME候选是关于"financial stress"；"starting
therapy"的ME候选是"grief and loss"；两个"workplace conflict"分别是MS候选变成
"exam anxiety"、ME候选变成"a sibling estrangement"。这不是"重新检索凑四组件"（确认
没有），是检索排序本身在该用户的多话题竞争下选错了话题——跟本session W6已经系统性
测过、三种排序器（production/BGE/hybrid）都无法解决的"同用户多候选话题竞争"问题
是同一根因，不是新发现的失效模式。**判定：这4条interaction row本身是Category A**
（不能直接拿去做"四组件联合效应"主张，因为候选组合本身话题不自洽）；**对整个蓝图是
Category B**（其余8个interaction和全部381个非interaction组不受影响，且这是已知、
已充分调查过的局限，不是本轮新增的构造错误）。

**10. 语义扩展探针保留，未被误改成负例**——**通过**。`semantic_extension_or_effect_
only_rows`跟`candidate_present`分开统计，不relabel。核对真实分布：MP `positive_
opportunity`条件下166/167（99.4%）观测到显著特征，`goal_mismatch`条件下0/124观测到
（这是设计预期——`goal_mismatch`状态本来就要求"不要基于画像定制"，特征观测不到是正确
行为，不是漏检）；ME正向邀请28/89、明确拒绝52/77，MS正向连续性26/89——数字与Leader
报告完全一致。

**11. 48个`groups_without_condition_crossing`**——**已定位，Category B，不阻塞**。
独立分解：全部48个精确对应"12个interaction × 4个组件视图"，全部单一条件
（`positive_opportunity`），没有掺杂任何非interaction的组。这不是"数据构造漏了负例"，
是interaction目前的设计范围本来就只测"联合候选是否存在"，还没有配对ON/OFF反事实条件——
**如实的结论是：这48个组现在还不能支持"interaction层面的paired effect"估计，但不影响
其余333个（381-48）单组件组，那些组已经真实交叉了多个condition**。这是一个需要向leader
明确的能力边界，不是需要现在修的bug。

**12. 是否用历史quality/risk/V5.2/外部结果反推**——**通过**。三个脚本和库模块通篇没有
读取任何quality/risk/judge/human outcome文件的代码路径；`generated_response_or_
outcome_read`/`quality_risk_judge_or_human_outcome_read`全程为`False`，人工审查代码
逻辑（`materialize_memory_state`/`materialize_rs_state`/`assemble_interaction_row`）
未发现任何隐藏的outcome依赖。

## 额外发现：Leader自己的`candidate_family_alignment`诊断指标有两处会制造假阳性

`79l`报告MP有77/322"mismatch"，乍看是个不小的比例（24%），但逐条分解后：

| 类别 | 数量 | 判定 |
|---|---:|---|
| MP_PREFERENCE被正确检索，但审计脚本的字段回退链（`topic_thread or field_type or field_value`）先读到恒非空的`field_type='response_format'`，从未走到`field_value`比较 | 49 | **Category C，审计器自身逻辑问题，不是数据问题** |
| interaction的`family`字符串（如"interaction::relocation abroad"）跟MP `field_type`字符串（如"location"）逐字比较，两者只通过78k手工维护的`field_for_topic`映射间接相关，不是应该逐字相等的东西 | 12 | **Category C，命名比较方式问题，不是检索问题** |
| 真实的跨字段串扰（比如education语境查询检索到job字段候选；`reflection_question`偏好家族检索到`direct_response`候选） | 16 | **Category B，真实、小比例（≈5%）问题，同ME/MS话题竞争同一根因** |

如果直接用"77"这个数字判断MP有严重问题，会把61个（49+12）审计器自身的噪声当成数据
缺陷——这正是复核任务要求的"不要仅凭字段名判失败"。真正需要披露的MP问题规模是16/322
（约5%），量级上跟ME/MS话题竞争问题一致，不是一个独立的新问题。

## 给Leader的建议（不代Leader决定，只是复核意见）

1. 4个话题不一致的interaction row（`relocation_abroad`×1、`starting_therapy`×1、
   `workplace_conflict`×2）建议要么修复该4个state的检索排序，要么在split冻结前明确
   标注排除出"四组件联合效应"主张，只保留其余8个；
2. `79l`的`candidate_family_alignment`诊断建议后续版本修复field_value回退优先级和
   interaction命名比较逻辑，避免未来又被这61个噪声干扰判断；
3. 48个"缺条件交叉"的interaction-only组，建议在正式文档里明确写清楚"当前interaction
   只测联合候选可用性，不测联合效应"，避免后续误用为effect训练组；
4. MP/ME/MS各自约1.5%–6%的真实跨字段/跨话题混淆，属于本session已经反复验证、三种
   排序器都未能解决的已知局限，建议如实披露，不建议现在投入新的排序算法改造。

## 提交、命令

```bash
cd /home/tokkio/snap/metacom_v33_pm_v1_5_repair/project
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python scripts/v1_5/80w_audit_p2r_learning_blueprint_v1_5.py
python -m pytest tests/test_v1_5_v5_3_p2r_learning_blueprint.py -v   # 8 passed，独立复核未改动此文件
```
