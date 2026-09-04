# Worker本轮完工报告：P2候选与数据基础收口（2026-08-06，零API）

写在最前面：本文是这一轮零API工作的收口报告，交给leader复核。工作过程中发现leader会话
已经把`PM_V1_5_V5_3_EXTERNAL_EXPERIMENT_AND_PARALLEL_RUNBOOK_20260806_ZH.md`正式建成
"双Codex协议"（第7节），把我的角色定为Worker、任务拆成W1-W5。本文按该文件第8节"给Worker
的启动指令"要求的六项回答，同时满足用户直接要求的五项交付物。**这份协议是leader会话在我
已经开始/部分完成本轮工作期间建立的，本文发现后没有中途改变已经做的方法，只在这里如实
对照、回答。**

## 一个需要如实说明的情况：本轮期间检测到并发提交

任务开始时用户已明确说"暂不进入双Codex并行阶段"。但工作过程中，`git log`发现同一仓库在
本轮期间被另一会话（同一git author）交替提交了8次（`e2691ab`/`79cd17d`/`0cc1544`/
`0f7a685`/`a3db5f2`/`400f204`/`8b8877f`/`d146119`），包括建立上述运行手册、
`v1_5_v5_3_response_baselines.py`、`v1_5_v5_3_accountability_runner.py`、
`v1_5_v5_3_external_leakage_audit.py`等新文件，以及扩写`v1_5_v5_3_typed_response_
program.py`和`v1_5_v5_3_accountability.py`。核对后没有文件级冲突（`git status`全程
clean），也没有与我的文件重名覆盖，运行手册明确把`scripts/v1_5/64_mp_ms_contribution_
slot_domain_comparison_v1_5.py`标记为"Worker所有权，leader不得修改"。**但脚本编号确实
撞了两次**：`68_build_me_superdomain_v1_5.py`（我）vs
`68_freeze_v5_3_response_baseline_scaffold_v1_5.py`（leader）；
`69_build_rs_superdomain_v1_5.py`（我）vs
`69_audit_v5_3_external_source_leakage_v1_5.py`（leader）——两个文件都完整保留、
互不覆盖，只是不再严格顺序编号，如实报告，未擅自重命名任何一方的文件。

## 对Worker启动指令六项的回答（事后补齐，因为协议是工作过程中才建立的）

1. **任务ID**：对应W1（三态资格补全）、W2（MP/MS域审计，用户已知悉是我做的）、W3（RS域
   审计）、W4（全新ME superdomain）；W5（shortcut/leakage/duplicate审计）**本轮未做**，
   是明确的遗留项，见下文"仍未解决的问题"。
2. **创建/修改的精确文件路径**：见下方"新增文件清单"，全部为新文件，未修改任何已有冻结
   模块的实现（新增`v1_5_v5_3_candidate_layer_responsibility.py`是加法式新模块，不是
   修改）。
3. **明确不会修改的共享文件**：三份权威事实源（计划文档、机器合同JSON、失败账本）、
   运行手册。本轮全程未touch。
4. **是否读取任何已有quality/risk/outcome**：否，全部脚本零API、不读取任何quality/risk
   字段。
5. **是否调用API及预算identity**：否，全程零API，本轮开始前用户已明确暂停付费paired
   generation。
6. **commit、命令、测试、output、仍未解决限制**：见下文。

## 独立commit列表（本轮8次，全部我做的，与leader会话的8次交替提交区分开）

```text
6781aac research(Step1): real content-independent qualification for ME action-readiness observer
46aecf5 research(Step1): MP/MS contribution_slot domain comparison (real EvoEmo data)
4257102 research(Step1): RS eligibility domain comparison -- training construction never fires the real gate
249bd9b correct(Step1): RS eligibility finding was comparing the wrong implementation
5c22130 feat(Step1): unified candidate-layer responsibility split (RS)
a4a07f4 research(Step1): balanced content-independent qualification, ME action readiness
2dd8ccd research(Step1): new ME superdomain -- real density, double-verified, honest 44% pass rate
9d0859e research(Step1): new RS natural-language superdomain, 4/5 families verified
```

## 新脚本、数据、output、报告路径

| # | 脚本 | 主要output | 对应文档 |
|---|---|---|---|
| 63 | `scripts/v1_5/63_me_action_readiness_content_independent_qualification_v1_5.py` | `outputs/pm_v1_5_v5_3_me_action_readiness_content_independent_qualification_v1/report.json` | `docs/PM_V1_5_V5_3_ME_ACTION_READINESS_CONTENT_INDEPENDENT_QUALIFICATION_20260806_ZH.md` |
| 64 | `scripts/v1_5/64_mp_ms_contribution_slot_domain_comparison_v1_5.py` | `outputs/pm_v1_5_v5_3_mp_ms_contribution_slot_domain_comparison_v1/report.json` | `docs/PM_V1_5_V5_3_MP_MS_CONTRIBUTION_SLOT_DOMAIN_COMPARISON_20260806_ZH.md` |
| 65 | `scripts/v1_5/65_rs_eligibility_domain_comparison_v1_5.py`（已被66更正，保留不删） | `outputs/pm_v1_5_v5_3_rs_eligibility_domain_comparison_v1/report.json` | `docs/PM_V1_5_V5_3_RS_ELIGIBILITY_DOMAIN_COMPARISON_20260806_ZH.md` |
| 66 | `scripts/v1_5/66_rs_eligibility_domain_comparison_v2_v1_5.py` | `outputs/pm_v1_5_v5_3_rs_eligibility_domain_comparison_v2/report.json` | `docs/PM_V1_5_V5_3_RS_ELIGIBILITY_CORRECTION_20260806_ZH.md` |
| — | `src/metacom_pm/v1_5_v5_3_candidate_layer_responsibility.py`（新模块）+ `tests/test_v1_5_v5_3_candidate_layer_responsibility.py`（8测试） | — | `docs/PM_V1_5_V5_3_CANDIDATE_LAYER_RESPONSIBILITY_SPLIT_20260806_ZH.md` |
| 67 | `scripts/v1_5/67_me_action_readiness_balanced_qualification_v1_5.py` | `outputs/pm_v1_5_v5_3_me_action_readiness_balanced_qualification_v1/report.json` | `docs/PM_V1_5_V5_3_ME_ACTION_READINESS_BALANCED_QUALIFICATION_20260806_ZH.md` |
| 68 | `scripts/v1_5/68_build_me_superdomain_v1_5.py` | `outputs/pm_v1_5_v5_3_me_superdomain_v1/{report,states_all,states_verified}.json` | `docs/PM_V1_5_V5_3_ME_SUPERDOMAIN_CONSTRUCTION_20260806_ZH.md` |
| 69 | `scripts/v1_5/69_build_rs_superdomain_v1_5.py` | `outputs/pm_v1_5_v5_3_rs_superdomain_v1/report.json` | `docs/PM_V1_5_V5_3_RS_SUPERDOMAIN_CONSTRUCTION_20260806_ZH.md` |

## 运行命令与测试

```bash
cd /home/tokkio/snap/metacom_v33_pm_v1_5_repair/project
source /home/tokkio/snap/metacom_v33_pm_v1_5_repair/.venv-pm-v1-5/bin/activate
python scripts/v1_5/63_me_action_readiness_content_independent_qualification_v1_5.py
python scripts/v1_5/64_mp_ms_contribution_slot_domain_comparison_v1_5.py
python scripts/v1_5/66_rs_eligibility_domain_comparison_v2_v1_5.py
python scripts/v1_5/67_me_action_readiness_balanced_qualification_v1_5.py
python scripts/v1_5/68_build_me_superdomain_v1_5.py
python scripts/v1_5/69_build_rs_superdomain_v1_5.py
python -m pytest tests/test_v1_5_v5_3_candidate_layer_responsibility.py -q   # 8 passed
python -m pytest tests/ -q -k "v5_3 or strategy_rag or candidate_discovery"   # 全部通过
```

**全量`pytest tests/ -q`（1700+条）复核**：18条失败，逐条用`git worktree`核对到本轮
开始前的`2f6de12`checkpoint，**全部18条在本轮任何一方提交之前就已经失败**（含账本
`V15-PM-41` ID重复、V3候选物料化MP缺失、`test_posthoc_v1_bank_probe.py`等），本轮
新增/修改的代码**零新增失败**。18条清单：

```text
test_mock_pipeline.py::test_mock_sweep_judge_gate
test_pm_v2_eval_hardening.py::test_evoemo_turn_resume_never_repeats_successful_or_failed_http_attempts
test_posthoc_v1_bank_probe.py::test_dry_run_is_deterministic_balanced_and_changes_only_strategy_section
test_posthoc_v1_bank_probe.py::test_truncated_call_is_raw_logged_and_fails_closed
test_posthoc_v1_bank_probe.py::test_reported_prompt_token_overrun_is_raw_logged_and_fails_closed
test_posthoc_v1_bank_probe.py::test_run_rejects_unaccepted_cost_hash_before_client
test_posthoc_v1_bank_probe.py::test_client_initialization_failure_does_not_reserve_physical_attempt
test_posthoc_v1_bank_probe.py::test_complete_run_builds_reverse_order_blind_review_package
test_posthoc_v1_bank_probe.py::test_final_observed_cost_must_fit_accepted_and_cli_budgets
test_scientific_safeguards.py::test_release_preflight_includes_model_family_independence
test_v1_5_esconv_auxiliary_generation_runner.py::test_dry_run_is_action_first_not_condition_first
test_v1_5_esconv_auxiliary_generation_runner.py::test_dry_run_shape_is_stable_across_invocations
test_v1_5_esconv_generation_runner.py::test_esconv_generation_dry_run_is_action_first_not_condition_first
test_v1_5_final_construction_realization.py::test_construction_realization_checks_only_mechanical_promises
test_v1_5_latest_protocol_repairs.py::test_paid_entrypoints_use_legacy_release_or_active_resumable_mvp_guard
test_v1_5_learnable_transfer_readiness.py::test_failure_ledger_has_unique_ids_and_v2_root_causes
test_v1_5_v3_candidate_materialization.py::test_v3_formal_exact_rank1_materialization_passes_without_labels_or_outcomes
test_v1_5_v3_candidate_materialization.py::test_every_target_component_surface_is_present_and_exact_rank1
```

这份既有失败清单本身值得leader关注（尤其账本ID重复这条直接影响机器可读的问题清单完整性），
但不是本轮引入，不在本轮范围内修。

## 四组件candidate-preflight汇总表

| 组件 | 候选存在/覆盖 | 关键构念合格率 | 已知domain gap | wrong-owner/future/label-shortcut |
|---|---|---|---|---|
| **MP** | EvoEmo 138 state中27个Rank-1命中MP（其余111个无MP候选，未测owner/time原因） | `MP_PROFILE`两域皆有支持；`MP_PREFERENCE`训练域80/160，**EvoEmo域真实0个**（直接读取全部126条真实item确认，非检索遗漏） | `profile_goal_needs_advice_or_arrangement`两域均0%（跟ME`current_action_invitation`同一族信号过窄，非MP独有） | 未测（W5未做，见下） |
| **MS** | EvoEmo 138/138全部有候选 | `has_specific_prior_observation`（编译资格）训练域/EvoEmo域**均100%**，没有ME那种域间差距 | `continuity_request`两域均0%（信号本身未生效，非域间差距）；production是否默认BGE-M3仍未核实（opt-in，见开放问题） | 未测 |
| **ME** | 新superdomain 32个state、每state 7条候选（真实密度，非旧语料2条） | intended-positive经真实compiler+真实Rank-1双重验证：**14/32（43.75%）通过**，其余18个是同主题干扰项赢Rank-1（真实排序鲁棒性问题，非构造bug） | 三态观察器：70条真实样本假阳性0%但真正例仅2个；180条平衡集recall INVITES 50%、DECLINES 33%、UNKNOWN precision 100% | 新superdomain全新语义族，未做未来/他人历史专项测试 |
| **RS** | 6卡系统：`rs_mechanical_candidate_pool`修复后候选池只在`active_high_stakes`/`explicit_stop`时清空，其余场景恒定6个（或已执行后5个）；`effect_study`系统：训练100%/ESConv 94.4%/EvoEmo 95.6% not-hard-off | natural-language superdomain 5家族4/5命中transparent-rule预期（AM05因"feel so overwhelmed"副词间隔漏判，真实小缺口非bug） | 6卡系统 any-move-eligible：训练0%（构造语气非自然，已定位根因）vs真实ESConv 4.8%/EvoEmo 9.3%（169/204正式抽样）——**是否仍是问题取决于leader选定哪套Bank为正式系统**，未擅自决定 | 未测；6卡vs80-card（50核心/100执行）两套Bank并存，尚未统一 |

## 仍未解决的问题（不隐瞒、不代leader决定）

1. **W5未做**：shortcut/leakage/duplicate/同源审计——topic/长度/前缀/候选数/subtype与
   标签解耦、user/family/group split零交叉、P2与外部文本零复制的**系统性**核查，本轮
   只在各自脚本docstring里声明"全新撰写、未复制"，没有做统一的程序化审计。leader会话的
   `69_audit_v5_3_external_source_leakage_v1_5.py`可能已经覆盖部分范围，需要核对是否
   与W5要求重叠或互补。
2. **RS该用哪套Bank未决**：6卡`v1_5_strategy_rag_runtime`还是50核心/100执行卡的
   `v1_5_strategy_rag_v4`+`v1_5_strategy_rag_repair`，两套依然并存，`候选层责任分离`
   目前只覆盖了6卡系统。
3. **ME排序鲁棒性问题依然真实存在**：新superdomain即使query已刻意对齐target措辞，
   43.75%的通过率说明光靠构造技巧治标不了，需要leader决定是否要动排序算法本身。
4. **MS的BGE-M3是否已是生产默认**仍未核实（开放问题清单第4项遗留，本轮未重新核对）。
5. **三态观察器的已知缺口未修**："where can I go for X"、多种委婉拒绝表达——本轮如实
   记录、没有为了刷分改写句子或扩正则。
6. **8个新语义族×32个state远低于运行手册128组下限**，是零API单轮时间内的真实产出，
   不是formal scale FIT数据。
7. **账本三条修正（RS的V15-DATA-81、本轮8个commit的新发现）均未回写三份权威事实源**，
   按本轮约定留给leader统一同步。
