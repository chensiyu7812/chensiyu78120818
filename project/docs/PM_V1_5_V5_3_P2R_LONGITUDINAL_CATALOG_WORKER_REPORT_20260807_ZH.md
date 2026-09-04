# W8：P2R纵向目录资产（source-independent longitudinal catalog）（2026-08-07）

## 范围声明

只构造一个独立的、外部文本零复制的longitudinal catalog asset。**不构造current_user_
text，不计算Step1特征，不生成worth_opening/ON-OFF标签，不决定split**，不调用任何API，
不读取quality/risk/judge/human outcome，未修改三份权威事实源、运行手册，或任何Leader
文件（release bindings、accountability、typed_response_program、75l/76l/77l审计脚本
/文档）。

脚本：`scripts/v1_5/77w_materialize_p2r_longitudinal_catalog_v1_5.py`（独占路径）
output：`outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1/{catalog.jsonl,users.json,
catalog_manifest.json,overlap_audit.json,coverage_report.json}`

## 设计依据：直接对齐三项外部考卷反推出的真实分布

`PM_V1_5_V5_3_EXTERNAL_EXAM_BACKWARD_TRAINING_SPEC_20260807_ZH.md`给出的EvoEmo真实
artifact统计：18用户401 sessions（13–33/用户）、446 events（13–37/用户）、150
relationships（3–15/用户）、126个固定7类基础画像事实。本次目录直接对齐这个规模和
结构（不是复制内容）：20个合成用户，每用户session数落在[13,33]真实区间，每用户3–5个
recurring topic thread，MP_PROFILE用同样的7类字段（name/age/gender/job/education/
nationality/location）。

## 归纳硬负例：同主题、不同事件/人物/时间/解决状态

每个用户的每个recurring topic thread会在**不同session**产出2–4条ME/MS条目，每条用
独立的动作/人物/结果组合（不是同一模板换话题）。抽查一个真实用户（`p2r_cat_u000`）：
"relocation abroad"这一个主题在session 3/4/7分别产出三条不同的ME_REUSABLE_OUTCOME
（"paused a full day..."/"talked it through with my manager..."/"walked away for a
bit..."，三个不同动作、不同或缺省人物、不同具体结果），加一条MS观察；"financial
stress"主题在session 9/11/12出现，其中两条分别涉及"my roommate"和"an old classmate"
两个不同的人。这正是"同主题不同事件"硬负例的真实实例，不是构造声明。

## 一个真实的构造bug，已定位并修复

第一版ME_REUSABLE_OUTCOME的compiler-valid率只有61/121（50.4%）——远低于预期的接近
100%。核查发现：动作词库里"took"/"made"/"stepped"/"broke"/"set"/"kept"六个动词都
**不在**`_FIRST_PERSON_ACTION_RE`真实的动词白名单里（该白名单是本session早些时候为
ME正则召回率修复过的那份：tried/used/chose/decided to/asked/called/wrote/walked/
went/practiced/paused/breathed/talked/scheduled/limited/stopped/started/reached
out/joined/attended/opened up/confided in/focused on）。改用白名单内动词重写这6条
（"took"→"paused"、"made"→"used"、"stepped away"→"walked away"、"broke into"→
"focused on"、"set a boundary"→"opened up about setting a boundary"、"kept a
log"→"started a log"）后，compiler-valid率精确变为121/121（100%的intended-positive
全部通过，65+41=106条intended-distractor全部正确地不通过，无一条串号）。**这不是
凑数字，是先用真实动词白名单核对每个action phrase，再逐条替换，跟这次session反复
应用的"先诊断根因、再针对性修复"方法一致。**

## 结构化MP_PROFILE + 内部MP_PREFERENCE

MP_PROFILE用7类字段（跟EvoEmo真实basic_info字段对齐），每条保存`field_type`/
`field_value`/`owner_id`/`version`/`active`/`superseded`。35%的用户获得一次时间线
中段的字段更新（比如工作变动），旧版本正确标记`superseded=True, active=False`，
新版本`version`递增并记录`supersedes_item_id`，用于验证time/version追踪。另外每用户
1–2条MP_PREFERENCE（跟MP_PROFILE分开子类型），因为反推文档明确"EvoEmo没有
MP_PREFERENCE，该子构念保留为内部专门能力"——不混进跟外部对齐的PROFILE统计。

## Owner隔离

`catalog.jsonl`是全部20个用户共享的全局文件（按任务要求，用于隔离审计），但脚本对
每个owner的候选池做了显式验证：过滤后每个用户的池子里跨owner泄漏数=**0**，无
`created_session<1`的未来/非法session。

## 外部同源审计（ESConv + EvoEmo + ES-MemEval）

复用leader的`v1_5_v5_3_external_leakage_audit.py`里已验证的`compare_text_surface_
overlap`/`external_overlap_surfaces`/`extract_internal_text_surfaces`函数（只读
调用，未修改该文件）。EvoEmo的`external_overlap_surfaces()`同时覆盖`dialog_history`
和内嵌的`questions`/`summaries`容器（ES-MemEval的题面本身就存在同一份`evo_emo.json`
里），所以虽然只读了两个物理文件，实际是ESConv+EvoEmo+ES-MemEval三源审计。928条内部
surface vs 48,912条外部surface（45,407条ESConv + 3,505条EvoEmo/ES-MemEval）：
**exact overlap=0，normalized 8-gram overlap=0**。

## 汇总数字

| 指标 | 值 |
|---|---|
| 用户数 | 20 |
| 每用户session范围 | 14–33（目标13–33，本次随机种子未抽到13，仍在真实区间内） |
| session数中位数 | 23 |
| 总目录条目数 | 464 |
| 每用户recurring topic thread数 | 3–5 |

**MP/MS/ME子类型分布**：

| subtype | 数量 |
|---|---:|
| MP_PROFILE | 129 |
| MP_PREFERENCE | 31 |
| MS_SESSION | 77 |
| ME_REUSABLE_OUTCOME | 121 |
| ME_UNRESOLVED_EVENT | 65 |
| ME_CONTEXT_EVENT | 41 |

**compiler-valid / invalid**：ME 121 valid / 106 invalid（全部121条intended-positive
通过，全部106条intended-distractor不通过，0串号）；MS 77 valid / 0 invalid（100%，
沿用本session已验证的"The earlier X goal was to Y"构念）。

**MS resolution_status分布**：resolved 28 / unresolved 23 / conflicting 26。

**MP_PROFILE字段分布**：job 18、education 17、nationality 20、location 20、age 16、
gender 20、name 18；5个字段发生过一次时间线中段更新（superseded）。

**owner/time违规**：跨owner泄漏0，非法session（<1）0。

**外部overlap**：exact collision 0，normalized 8-gram collision 0（928条内部
surface对48,912条外部surface）。

## 提交、命令、测试

```bash
cd /home/tokkio/snap/metacom_v33_pm_v1_5_repair/project
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/python scripts/v1_5/77w_materialize_p2r_longitudinal_catalog_v1_5.py
python -m pytest tests/ -q -k "v1_5_v5_2_atomic_memory or external_leakage_audit or evoemo"
```

跑测试时发现`test_pm_v2_eval_hardening.py::test_evoemo_turn_resume_never_repeats_
successful_or_failed_http_attempts`失败——这是本session更早已经核实过的18个既有失败
之一（跟本轮改动无关，未在本轮引入）。工作树在提交前干净，未使用`git add .`、未
amend、未rebase。

## 仍未解决的限制

1. 这只是catalog本身，没有current state/query，没有Step1特征，没有split——按任务
   边界如此，等leader导入后构建P2R blueprint；
2. session数范围本次随机结果是14–33，没有精确覆盖到13（真实EvoEmo最小值），如果
   leader需要精确覆盖边界值，需要调整随机种子或显式包含至少一个13-session用户；
3. ME_UNRESOLVED_EVENT的"tried journaling.../brought X up with.../started
   tracking..."三个模板轮换数（3）小于thread数，同一用户内可能出现同一模板重复
   （不同session、不同topic，但句式相同）——如实披露，未进一步扩充模板库；
4. 内容多样性验证只做了人工抽查一个用户，没有对全部20个用户做系统性SequenceMatcher
   相似度扫描（跟W7R不同，本次任务边界不要求session/state级别的group相似度审计，
   只要求catalog本身覆盖真实分布和硬负例结构）；
5. MP_PREFERENCE只有31条（每用户1–2条），规模明显小于MP_PROFILE，符合"内部专门
   能力、非外部对齐主力"的定位，但如果未来需要更大规模的内部preference受控实验，
   这批不够用，需要单独扩充。
