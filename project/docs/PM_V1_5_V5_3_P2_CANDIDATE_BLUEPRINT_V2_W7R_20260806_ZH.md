# W7R：P2候选蓝图修复（2026-08-06/07）

响应leader审计`docs/PM_V1_5_V5_3_P2_CANDIDATE_BLUEPRINT_LEADER_AUDIT_20260806_ZH.md`
（状态`REPAIR_REQUIRED_BEFORE_P2_READY`）。**不原地改写W7（脚本71w/v1 output）**，新脚本
`72w_materialize_p2_candidate_blueprint_v2_v1_5.py`，新output
`outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2/`。零API，不生成回复，不训练head，
不读取quality/risk/outcome，未修改三份权威事实源、leader审计脚本/文档、static release
bindings、MS/ME冻结方法、Strategy Bank、typed Step2/baselines/accountability/formal
runner。

## 自我复核结果（复用审计报告已公开的字段名/口径自查，未执行或修改leader的75l脚本）

| 审计发现 | 检查项 | W7（v1） | W7R（v2） |
|---|---|---:|---:|
| AUDIT-01 MP隐藏增量 | positive状态被判`current_redundant` | 14/16（原始）→28/32（W7重新统计） | **0/32** |
| AUDIT-01 | profile positive的`profile_goal_needs_advice_or_arrangement=False` | 8/8 | **0/32** |
| AUDIT-02 MS单候选 | `n_candidates_in_catalog==1` | 16/16 | **0/32**（全部3条候选） |
| AUDIT-02 | 缺失`score_top1_semantic_relevance` | 16/16 | **0/32** |
| AUDIT-03 ME答案泄漏 | `step1_features`含`rank1_matches_intended_target` | 32/32 | **0/40** |
| AUDIT-06 RS | `exact_rank1_subtype`误存selection_mode | 6条 | **0/13** |
| AUDIT-04 | `current_goal`是构造标签 | 82/82 | **0/133+5** |
| AUDIT-06 | 缺`candidate_version`/`current_surface_sha256`/`candidate_surface_sha256` | 79/79 | **0/133** |
| AUDIT-06 | interaction缺`model_visible_surfaces`/`step1_features`/`candidate_lineage` | 3/3 | **0/5** |

## 每组件具体修法

**MP**：preference候选文本改成`"prefers {word_a} and {word_b}"`这种最小受控短语，只包含
两个来自检索器`_PREFERENCE_SCOPE_WORDS`词表的词；positive current turn用完全独立措辞的
句子+这两个词，不复述候选整句。**第一版仍然踩雷**：查询沿用了候选句里的连接词（比如
"instead of"），导致意外共享3个词触发`current_redundant`；第二版把候选文本精简到只剩
两个目标词，查询也改成完全独立的句式，问题清零。profile positive用"What should I
do about the {某个具体词}situation..."这类真实匹配`explicit_advice_welcome`的句式，
只引用field value里的一个具体词而非整个value。redundant对照允许复述，且改成4套轮换
措辞降低跨族相似度。每行新增`field_type`/`field_value`/`candidate_version`。

**MS**：每个state目录从1条扩到3条（intended target + 同族不同session的干扰项 + 跨族
filler），全部严格过去、同用户，不复制EvoEmo/ES-MemEval文本。真正调用BGE-M3对全部候选
排序，保存`score_top1_semantic_relevance`、`score_rank1_rank2_semantic_margin`和完整
`topk_lineage`（每条候选的rank/semantic_score/created_session）。

**ME**：`rank1_matches_intended_target`及`negative_kind`/target ID全部移进`audit_only`
字典，`step1_features`对ME清空（当前没有需要暴露给模型的运行时特征，跟production方法
本身不变一致）。继续用冻结production lexical+typed-tier，Rank-1编译失败记
`me_available=False`、不晋升Rank-2，跟W6/W7完全一致，未重新搜索方法。新增8个ME族
（social_anxiety/grief_loss/financial_stress/relocation_abroad/health_diagnosis/
caregiving_burnout/workplace_conflict/identity_question），复用脚本68同一套真实
compiler+真实Rank-1双重验证方法。**同时重写了复用自脚本68的explicit_decline/
goal_mismatch两个变体的措辞**——原文本只换主题词、句式相似度约0.90，被审计明确点名；
现在按8个族分别用不同的自然拒绝/情绪优先表达。

**RS**：`exact_rank1_subtype`只存卡片子类型（`RS_ATOMIC_MOVE`），新增独立
`selection_mode`（`transparent_priority`/`lexical_fallback`）和`transparent_rule_on`
字段。继续用冻结6-card Bank+`rs_mechanical_candidate_pool`+`rs_shared_candidate_top1`，
未改检索方法。每个AM家族增加一条自然改写版本（比如open_expression从1条自然发言扩到2条
措辞不同的），加已执行排重变体，共13个state。

**全组件**：`current_goal`现在直接等于可见的`current_user_text`本身（自然、运行时可得），
不再是`positive`/`goal_mismatch`等构造标签；构造标签移入`audit_only.construction_
condition`，绝不进入`model_visible_surface`/`step1_features`/`current_goal`任何一个
会喂给模型的字段。每个单组件行新增`candidate_version`（本轮全部为1）、
`current_surface_sha256`、`candidate_surface_sha256`。

**interaction**：5个state（3个MS+ME+RS三组件、2个MS+RS真实pairwise——后者是因为这两个
新ME族的候选不在脚本68复用的原state文件里，interaction构造时如实没有找到ME候选，不是
刻意设计的pairwise，但确实产出了真实的两组件覆盖，如实报告）。每个state补齐
`model_visible_surfaces`（每组件一条或None）、`candidate_lineage`（含candidate_id/
surface hash/created_session/version）、`step1_features`。**MP在全部5个interaction
state里仍然缺失**——interaction查询措辞是围绕ME/MS话题构造的，没有专门对齐MP的偏好/
画像词表，如实报告，没有为了凑MP硬调查询。

## 真实规模（不冻结最终N，供leader审计后决定）

| 组件 | W7（v1） | W7R（v2） |
|---|---:|---:|
| MP state / group | 24 / 8 | 48 / 48 |
| MS state / group | 16 / 8 | 32 / 32 |
| ME state / group | 32 / 8 | 40 / 40（8旧族×4变体+8新族×1变体） |
| RS state / group | 7 / 7 | 13 / 13 |
| interaction | 3 | 5（3个三组件+2个真实pairwise） |
| 唯一用户 | 58 | 90 |
| 唯一family | 15 | 23 |

候选池规模分布：MP恒为2（preference+profile两条，设计如此）；MS恒为3（新的多候选目录）；
ME恒为7（延续脚本68的真实密度设计）；RS为0（explicit_stop硬关闭）/5（已执行排重后）/6
（正常）。

候选可用率：MP 100%、MS 100%、ME `me_available` 90%（36/40，新8族有小样本压缩、旧32州
延续W6/W7已知的96.9%附近水平，整体略降是因为新族更小样本更容易出现单点失败，未做进一步
调参掩盖）、RS 92.3%（1个explicit_stop硬关闭，符合设计）。

## 句式相似度（SequenceMatcher≥0.72，按component×negative_kind分桶）

修复前（W7分析）：ME explicit-decline/goal-mismatch跨族约0.90（已点名），MS
continuity-positive约0.87。**修复后**：ME两个问题变体的高相似对**清零**；MS
continuity-positive从58对降到11对、current_redundant从16对降到2对；MP
profile_positive 18对、current_redundant 24对、preference_positive 3对——**不是零**，
如实报告，未继续无限打磨到全部消失，因为审计本身不要求"全绿"，只要求真实改善。

## 未解决限制

1. MP在全部5个interaction state里都不可用（同W7），未设计专门对齐MP词表的interaction
   查询；
2. MP profile_positive/current_redundant仍有18/24对句式相似度≥0.72，主要是
   "What should I do about the X situation..."和"Like I said, I prefer..."这两类
   模板轮换数（5/4套）小于家族数（16），必然出现重复轮换，需要更多样板才能进一步降低；
3. ME新8族只做了"positive"一个变体（未复刻旧8族的4变体网格），组间比较不完全对称；
4. `proposed_split`全部占位`TBD_by_leader`，未做shortcut/duplicate/同源overlap复核
   （按分工这是leader下一步）；
5. RS仍是6-card系统专属修复，50核心/100执行卡系统未动（跟W7一致，等leader定基调）。

## 提交、命令、测试

```bash
cd /home/tokkio/snap/metacom_v33_pm_v1_5_repair/project
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python scripts/v1_5/72w_materialize_p2_candidate_blueprint_v2_v1_5.py
python -m pytest tests/ -q -k "v5_3 or candidate_discovery or v1_5_v5_2_atomic_memory or strategy_rag"  # 全部通过，无回归
```

脚本：`scripts/v1_5/72w_materialize_p2_candidate_blueprint_v2_v1_5.py`
output：`outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2/{summary,mp_states,ms_states,
me_states,rs_states,interaction_states}.json`
