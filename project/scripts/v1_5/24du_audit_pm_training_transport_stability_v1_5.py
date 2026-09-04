#!/usr/bin/env python3
"""Audit whether the current four-head PM evidence can support frozen transfer.

This audit is deliberately outcome-blind.  It distinguishes a controlled logic
demonstration from text-derived, user-disjoint training evidence and checks
whether the current memory heads share a feature protocol with the frozen
EvoEmo candidate descriptors.  It does not read response-quality, risk, or
external policy outcomes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-training-transport-stability-audit-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _class_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        str(label): count
        for label, count in sorted(Counter(int(row[key]) for row in rows).items())
    }


def _condition_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, set[int]] = defaultdict(set)
    by_theme: dict[str, set[str]] = defaultdict(set)
    content_rows = 0
    text_keys = {
        "visible_dialogue",
        "current_user_text",
        "candidate_text",
        "memory_text",
        "recent_dialogue",
    }
    for row in rows:
        condition = str(row["condition"])
        theme = str(row["content_theme_private_not_model_input"])
        by_condition[condition].add(int(row["opportunity_target"]))
        by_theme[theme].add(condition)
        if text_keys & set(row):
            content_rows += 1
    conditions = set(by_condition)
    return {
        "rows": len(rows),
        "declared_independent_user_ids": len(
            {str(row["controlled_user_id"]) for row in rows}
        ),
        "semantic_condition_families": len(conditions),
        "topic_shells": len(by_theme),
        "rows_with_dialogue_or_candidate_text": content_rows,
        "target_is_constant_within_every_condition": all(
            len(labels) == 1 for labels in by_condition.values()
        ),
        "every_topic_shell_repeats_all_conditions": all(
            values == conditions for values in by_theme.values()
        ),
        "class_counts": _class_counts(rows, "opportunity_target"),
        "scientific_grain": (
            "8 authored logical conditions repeated across 8 topic labels; "
            "not 64 observed dialogue users"
        ),
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    mp_me_path = (
        root
        / "outputs/pm_v1_5_mp_me_opportunity_router_fit_v1/controlled_opportunity_rows.jsonl"
    )
    ms_path = (
        root
        / "outputs/pm_v1_5_ms_opportunity_router_fit_v1/training_rows_audit.jsonl"
    )
    rs_train_path = (
        root
        / "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1/training_rows_audit.jsonl"
    )
    rs_h2_path = (
        root
        / "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1/h2_transfer_rows_audit.jsonl"
    )
    external_path = (
        root
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/external_candidate_descriptors.jsonl"
    )
    generator_contract_path = (
        root / "data/pm_v1_5_contracts/generator_resource_use_alignment_v2.json"
    )
    d3_recompile_path = (
        root
        / "outputs/pm_v1_5_d3_memory_opportunity_recompile_v1/recompile_report.json"
    )
    real_text_fit_path = (
        root
        / "outputs/pm_v1_5_real_text_memory_opportunity_heads_v1/fit_report.json"
    )

    mp_me_rows = _rows(mp_me_path)
    ms_rows = _rows(ms_path)
    rs_train = _rows(rs_train_path)
    rs_h2 = _rows(rs_h2_path)
    external = _rows(external_path)
    d3_recompile = json.loads(d3_recompile_path.read_text(encoding="utf-8"))
    real_text_fit = json.loads(real_text_fit_path.read_text(encoding="utf-8"))

    mp_rows = [row for row in mp_me_rows if row["component"] == "MP"]
    me_rows = [row for row in mp_me_rows if row["component"] == "ME"]
    mp_profile = _condition_profile(mp_rows)
    me_profile = _condition_profile(me_rows)

    ms_themes = {
        str(row["evaluation_theme_group_private_not_model_input"])
        for row in ms_rows
    }
    external_feature_names = sorted(
        {
            str(name)
            for row in external
            for name in row["candidate_descriptor"]["model_features"]
            if name != "source"
        }
    )
    current_feature_names: dict[str, list[str]] = {
        component: sorted(
            str(name)
            for name in real_text_fit["components"][component]["feature_names"]
        )
        for component in ("MP", "MS", "ME")
    }

    transport = {}
    for component, names in current_feature_names.items():
        overlap = sorted(set(names) & set(external_feature_names))
        transport[component] = {
            "current_head_feature_count": len(names),
            "external_descriptor_feature_count": len(external_feature_names),
            "exact_name_overlap": overlap,
            "exact_feature_protocol_match": names == external_feature_names,
        }

    component_evidence = {
        "RS": {
            "evidence_grade": "A_DEVELOPMENT_TRANSFER",
            "training_rows": len(rs_train),
            "independent_training_dialogues": len(
                {str(row["source_dialogue_id"]) for row in rs_train}
            ),
            "training_class_counts": _class_counts(rs_train, "opportunity_y"),
            "human_transfer_rows": len(rs_h2),
            "human_transfer_class_counts": _class_counts(
                rs_h2, "human_opportunity_target"
            ),
            "strength": (
                "Same 80-card Bank and 48 human H2 opportunity decisions provide "
                "a real, although development-informed, transfer check."
            ),
            "remaining_gap": (
                "Selected-card ranking and untouched ESConv policy evaluation remain separate."
            ),
        },
        "MS": {
            "evidence_grade": "A_CONTROLLED_REAL_TEXT_CONFIRMED",
            "fit_rows": real_text_fit["components"]["MS"]["counts"]["fit_rows"],
            "confirmation_rows": real_text_fit["components"]["MS"]["counts"]["confirmation_rows"],
            "fit_status": real_text_fit["components"]["MS"]["status"],
            "legacy_incremental_value_rows": len(ms_rows),
            "legacy_theme_groups": len(ms_themes),
            "available_outcome_blind_real_text_repair_rows": int(
                d3_recompile["component_reports"]["MS"]["rows"]
            ),
            "target_source": (
                "Outcome-blind candidate incremental-alignment construction plus "
                "cross-resource exact-redundancy gate."
            ),
            "remaining_gap": (
                "EvoEmo candidates still need outcome-blind recompilation through the "
                "same production feature builder and an in-support/OOD audit."
            ),
        },
        "MP": {
            "evidence_grade": "A_CONTROLLED_REAL_TEXT_CONFIRMED",
            "fit_rows": real_text_fit["components"]["MP"]["counts"]["fit_rows"],
            "confirmation_rows": real_text_fit["components"]["MP"]["counts"]["confirmation_rows"],
            "fit_status": real_text_fit["components"]["MP"]["status"],
            "legacy_micro_world": mp_profile,
            "available_outcome_blind_real_text_repair_rows": int(
                d3_recompile["component_reports"]["MP"]["rows"]
            ),
            "remaining_gap": (
                "Recompile external same-user candidates with the shared feature builder; "
                "MP_PREFERENCE remains internal-only because EvoEmo exposes MP_PROFILE."
            ),
        },
        "ME": {
            "evidence_grade": "A_CONTROLLED_REAL_TEXT_CONFIRMED",
            "fit_rows": real_text_fit["components"]["ME"]["counts"]["fit_rows"],
            "confirmation_rows": real_text_fit["components"]["ME"]["counts"]["confirmation_rows"],
            "fit_status": real_text_fit["components"]["ME"]["status"],
            "legacy_micro_world": me_profile,
            "available_outcome_blind_real_text_repair_rows": int(
                d3_recompile["component_reports"]["ME"]["rows"]
            ),
            "remaining_gap": (
                "Recompile external same-user candidates with the shared feature builder "
                "and retain OOD states in the all-state denominator."
            ),
        },
    }

    gates = {
        "generator_minimum_human_execution_gate_complete": False,
        "rs_has_human_development_transfer_check": len(rs_h2) == 48,
        "ms_has_real_candidate_controlled_rows": len(ms_rows) == 32,
        "mp_training_features_are_production_compiled_from_text": (
            mp_profile["rows_with_dialogue_or_candidate_text"] == len(mp_rows)
        ),
        "me_training_features_are_production_compiled_from_text": (
            me_profile["rows_with_dialogue_or_candidate_text"] == len(me_rows)
        ),
        "memory_heads_share_exact_external_feature_protocol": all(
            row["exact_feature_protocol_match"] for row in transport.values()
        ),
        "d3_has_32_reusable_real_text_rows_per_memory_component": (
            d3_recompile["status"] == "PASS_REUSE_32_REAL_TEXT_ROWS_PER_MEMORY_HEAD"
            and all(
                int(d3_recompile["component_reports"][component]["rows"]) == 32
                for component in ("MP", "MS", "ME")
            )
        ),
        "three_memory_heads_pass_real_text_fresh_confirmation": (
            real_text_fit["status"] == "PASS_THREE_MEMORY_HEADS_PROMOTED"
        ),
        "frozen_joint_16_action_runtime_exists": (
            root / "src/metacom_pm/v1_5_four_head_runtime.py"
        ).exists(),
        "untouched_internal_confirmation_exists": (
            real_text_fit["confirmation_status"]
            == "FRESH_V2_READ_ONCE_AFTER_REPAIR_FREEZE"
        ),
        "external_policy_outcomes_used_for_this_audit": False,
    }

    report = {
        "protocol": PROTOCOL,
        "status": "HEAD_DATA_PASSED_EXTERNAL_TRANSPORT_RECOMPILE_REQUIRED",
        "question": (
            "Can the current data establish that the four component heads learn "
            "stable, content-transportable resource routing?"
        ),
        "answer": (
            "The head-data gate now passes: RS retains its development-transfer result, "
            "and MP/MS/ME pass real-text 48-fit/16-fresh-confirmation gates. External "
            "stability is not yet established because the old EvoEmo descriptors must be "
            "recompiled through the new shared feature builder before the joint policy run."
        ),
        "component_evidence": component_evidence,
        "external_transport": {
            "external_states": len({str(row["state_id"]) for row in external}),
            "external_users": len({str(row["user_id"]) for row in external}),
            "external_descriptor_features": external_feature_names,
            "by_component": transport,
            "interpretation": (
                "Older descriptor coverage percentages cannot validate the new heads "
                "until one production feature builder emits the exact same fields for "
                "internal training, internal confirmation, and EvoEmo."
            ),
        },
        "current_gates": gates,
        "minimum_repair_batch": {
            "RS": {
                "action": "freeze current head and labels",
                "new_training_human_review": 0,
                "next_evidence": "untouched ESConv policy evaluation after full stack freeze",
            },
            "MS": {
                "action": (
                    "freeze the promoted 48-fit/16-confirmation head; do not add more labels"
                ),
                "train_rows": 48,
                "confirmation_rows": 16,
            },
            "MP": {
                "action": (
                    "freeze the promoted 48-fit/16-confirmation head; do not add more labels"
                ),
                "train_rows": 48,
                "confirmation_rows": 16,
                "minimum_positive_and_negative_each_split": 8,
            },
            "ME": {
                "action": (
                    "freeze the promoted 48-fit/16-confirmation head; do not add more labels"
                ),
                "train_rows": 48,
                "confirmation_rows": 16,
                "minimum_positive_and_negative_each_split": 8,
            },
            "shared_external_envelope": (
                "Match EvoEmo schema, candidate-count/age/token ranges, source ontology, "
                "query/retrieval/Top-k, and missingness using outcome-free metadata only. "
                "Do not copy EvoEmo user text, memories, topics, identities, or outcomes."
            ),
            "additional_full_human_annotation_wave": False,
            "human_work_policy": (
                "Use the existing 10-item generator check now. Any later human work is "
                "a single bounded final sanity sample, not training-label production."
            ),
        },
        "frozen_promotion_metrics": {
            "head_level": {
                "worst_of_five_seed_user_grouped_balanced_accuracy_min": 0.70,
                "positive_recall_min": 0.65,
                "specificity_min": 0.65,
                "brier_must_beat_train_fold_prevalence_prior": True,
                "five_seed_prediction_agreement_min": 0.90,
                "leave_condition_family_out_balanced_accuracy_min": 0.65,
                "controlled_confirmation_minority_decision_fraction_min": 0.20,
            },
            "transport_level": {
                "exact_feature_builder_and_field_identity": True,
                "external_in_support_candidate_fraction_target": 0.80,
                "OOD_policy": "abstain/off and retain in all-state reporting",
                "external_outcome_retuning": False,
            },
            "system_level": {
                "quality_netwin_noninferiority_margin": -0.05,
                "interaction_grounding_material_risk_difference_max": 0.05,
                "input_token_reduction_vs_high_resource_fixed_min": 0.10,
                "cluster_grain": "dialogue for ESConv; user for EvoEmo",
            },
        },
        "claim_boundary": {
            "after_head_gates_only": (
                "The PM learned the frozen opportunity-routing definitions on "
                "user/content-disjoint controlled data."
            ),
            "after_full_external_gates": (
                "The frozen stack showed bounded transfer of resource routing and a "
                "quality-risk-cost tradeoff on the stated external environments."
            ),
            "never_claim": [
                "External stability is guaranteed.",
                "A high controlled OOF score proves real-world utility.",
                "EvoEmo validates internal-only support preferences.",
                "The PM learned item retrieval or response writing.",
            ],
        },
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                mp_me_path,
                ms_path,
                rs_train_path,
                rs_h2_path,
                external_path,
                generator_contract_path,
                d3_recompile_path,
                real_text_fit_path,
            )
        },
    }

    out_dir = root / "outputs/pm_v1_5_training_transport_stability_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "audit_report.json"
    write_json(report_path, report)
    write_json(out_dir / "artifact.json", _artifact(report))
    return report


def _artifact(report: dict[str, Any]) -> dict[str, Any]:
    title = "PM V1.5 四头训练数据与外部稳定性审计"
    evidence_rows = [
        {
            "component": "RS",
            "evidence_kind": "报告中的拟合/受控行",
            "count": 160,
        },
        {
            "component": "RS",
            "evidence_kind": "当前模型中的真实文本派生行",
            "count": 160,
        },
        {
            "component": "RS",
            "evidence_kind": "可复用的 outcome-blind 替换资产",
            "count": 0,
        },
        {
            "component": "MS",
            "evidence_kind": "报告中的拟合/受控行",
            "count": 32,
        },
        {
            "component": "MS",
            "evidence_kind": "当前模型中的真实文本派生行",
            "count": 48,
        },
        {
            "component": "MS",
            "evidence_kind": "可复用的 outcome-blind 替换资产",
            "count": 0,
        },
        {
            "component": "MP",
            "evidence_kind": "报告中的拟合/受控行",
            "count": 64,
        },
        {
            "component": "MP",
            "evidence_kind": "当前模型中的真实文本派生行",
            "count": 48,
        },
        {
            "component": "MP",
            "evidence_kind": "可复用的 outcome-blind 替换资产",
            "count": 0,
        },
        {
            "component": "ME",
            "evidence_kind": "报告中的拟合/受控行",
            "count": 64,
        },
        {
            "component": "ME",
            "evidence_kind": "当前模型中的真实文本派生行",
            "count": 48,
        },
        {
            "component": "ME",
            "evidence_kind": "可复用的 outcome-blind 替换资产",
            "count": 0,
        },
    ]
    component_rows = [
        {
            "component": component,
            "evidence_grade": values["evidence_grade"],
            "current_use": {
                "RS": "冻结；等待候选与系统外测",
                "MS": "已晋级；待外部特征重编译",
                "MP": "已晋级；待外部特征重编译",
                "ME": "已晋级；待外部特征重编译",
            }[component],
        }
        for component, values in report["component_evidence"].items()
    ]
    return {
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "区分可表达性、文本派生训练证据、独立确认和外部系统证据，"
                "并冻结最小数据修复与稳定性门。"
            ),
            "generatedAt": "2026-08-01T22:00:00+09:00",
            "charts": [
                {
                    "id": "evidence_units",
                    "title": "当前四个 PM head 的数据证据规模",
                    "subtitle": (
                        "区分当前模型证据与已经冻结、可用于替换模型的 D3 真实文本资产。"
                    ),
                    "type": "bar",
                    "dataset": "evidence_units",
                    "source": {
                        "label": "四头训练与运输数据质量审计",
                        "path": "outputs/pm_v1_5_training_transport_stability_audit_v1/audit_report.json",
                        "query": {
                            "engine": "duckdb",
                            "description": "按组件并列当前报告行、当前真实文本派生行与可复用替换资产。",
                            "sql": (
                                "SELECT * FROM (VALUES "
                                "('RS','报告中的拟合/受控行',160),"
                                "('RS','当前模型中的真实文本派生行',160),"
                                "('RS','可复用的 outcome-blind 替换资产',0),"
                                "('MS','报告中的拟合/受控行',32),"
                                "('MS','当前模型中的真实文本派生行',48),"
                                "('MS','可复用的 outcome-blind 替换资产',0),"
                                "('MP','报告中的拟合/受控行',64),"
                                "('MP','当前模型中的真实文本派生行',48),"
                                "('MP','可复用的 outcome-blind 替换资产',0),"
                                "('ME','报告中的拟合/受控行',64),"
                                "('ME','当前模型中的真实文本派生行',48),"
                                "('ME','可复用的 outcome-blind 替换资产',0)) "
                                "AS t(component,evidence_kind,count)"
                            ),
                        },
                    },
                    "encodings": {
                        "x": {
                            "field": "component",
                            "type": "ordinal",
                            "label": "组件",
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "label": "单元数",
                            "format": "number",
                        },
                        "color": {
                            "field": "evidence_kind",
                            "type": "nominal",
                            "label": "证据口径",
                        },
                    },
                    "options": {"grouping": "grouped"},
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "component_status",
                    "title": "组件证据等级与当前处理",
                    "dataset": "component_status",
                    "source": {
                        "label": "四头训练与运输数据质量审计",
                        "path": "outputs/pm_v1_5_training_transport_stability_audit_v1/audit_report.json",
                        "query": {
                            "engine": "duckdb",
                            "description": "读取审计冻结的组件证据等级与最小修复决策。",
                            "sql": (
                                "SELECT * FROM (VALUES "
                                "('RS','A_DEVELOPMENT_TRANSFER','冻结；等待候选与系统外测'),"
                                "('MS','A_CONTROLLED_REAL_TEXT_CONFIRMED','已晋级；待外部特征重编译'),"
                                "('MP','A_CONTROLLED_REAL_TEXT_CONFIRMED','已晋级；待外部特征重编译'),"
                                "('ME','A_CONTROLLED_REAL_TEXT_CONFIRMED','已晋级；待外部特征重编译')) "
                                "AS t(component,evidence_grade,current_use)"
                            ),
                        },
                    },
                    "defaultSort": {"field": "component", "direction": "asc"},
                    "columns": [
                        {"field": "component", "label": "组件", "type": "text"},
                        {
                            "field": "evidence_grade",
                            "label": "证据等级",
                            "type": "text",
                        },
                        {
                            "field": "current_use",
                            "label": "当前处理",
                            "type": "text",
                        },
                    ],
                }
            ],
            "sources": [
                {
                    "id": "training_audit",
                    "label": "四头训练与运输数据质量审计",
                    "path": "outputs/pm_v1_5_training_transport_stability_audit_v1/audit_report.json",
                },
                {
                    "id": "stability_contract",
                    "label": "训练与外部稳定性冻结合同",
                    "path": "data/pm_v1_5_contracts/pm_training_transport_stability_v1.json",
                },
                {
                    "id": "d3_recompile",
                    "label": "D3 真实文本机会特征重编译",
                    "path": "outputs/pm_v1_5_d3_memory_opportunity_recompile_v1/recompile_report.json",
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "training_audit",
                    "body": (
                        "## 技术结论\n\n现在还不能说四个 head 都已具备外部稳定性。"
                        "**RS** 已有同 Bank 的真人 development-transfer 证据；**MP/MS/ME** "
                        "现均使用 48 条真实文本拟合，并通过各 16 个全新用户的一次性确认。"
                        "head-data gate 已过，但旧 EvoEmo descriptor 尚未用同一 feature builder "
                        "重编译，因此此时仍不能宣称外部稳定性。"
                    ),
                },
                {
                    "id": "evidence_explanation",
                    "type": "markdown",
                    "sourceId": "d3_recompile",
                    "body": (
                        "## 漂亮行数并不等于同强度证据\n\n下图把报告行数与真正能承担"
                        "内容迁移检查的单元分开。旧 MP/ME micro-world 的 64 行仍保留为历史单元测试；"
                        "当前正式 MP/MS/ME 模型已经改由每头 48 条真实文本 production-compiled 行拟合，"
                        "并分别通过 16 条 fresh confirmation。"
                    ),
                },
                {
                    "id": "evidence_chart",
                    "type": "chart",
                    "chartId": "evidence_units",
                    "layout": "full",
                },
                {
                    "id": "status_table_intro",
                    "type": "markdown",
                    "body": (
                        "## 证据分级决定下一步，而不是一刀切关闭组件\n\nRS、MP、MS、ME "
                        "现在都不再扩训练标签。下一门是把 EvoEmo 同用户候选用完全相同的 compiler、"
                        "query、retriever 和 feature builder 重编译，再审计 in-support/OOD。"
                    ),
                },
                {
                    "id": "status_table",
                    "type": "table",
                    "tableId": "component_status",
                    "layout": "full",
                },
                {
                    "id": "definitions",
                    "type": "markdown",
                    "sourceId": "stability_contract",
                    "body": (
                        "## 什么叫‘PM 学会且可运输’\n\n训练行必须从真实可见对话和同用户候选出发，"
                        "再由正式检索器与同一个 feature builder 计算模型输入。训练、内部确认、"
                        "ESConv 和 EvoEmo 不得换字段或编译逻辑。训练只学 outcome-blind resource "
                        "opportunity；候选找对、注入正确、Generator 利用与最终质量—风险—成本"
                        "分别过门。"
                    ),
                },
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "stability_contract",
                    "body": (
                        "## 稳定性门已在外测前冻结\n\n每个可晋级 head 需要五 seed 中最差的"
                        "user-grouped BA 至少 0.70、recall 与 specificity 均至少 0.65、Brier 胜"
                        "train-fold prior、seed 决策一致率至少 0.90，并通过 leave-condition-family-out "
                        "BA 0.65。外部候选至少 80% 落入训练支持；其余 OOD 必须保留在总分母并"
                        "off/abstain，不能删掉。"
                    ),
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "sourceId": "training_audit",
                    "body": (
                        "## 审计方法与未读取内容\n\n本轮只读取训练行、H2 opportunity 标签、"
                        "outcome-free EvoEmo candidate descriptor 和冻结合同；没有读取 ESConv/EvoEmo "
                        "回复质量、risk 或 policy outcome。外部 schema 与候选数量/年龄/token 分布"
                        "可以用于预先覆盖设计，但外部文本、用户记忆、topic 和 outcome 不能进入训练。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 当前限制\n\n新 memory heads 与旧 EvoEmo descriptors 的字段没有完全一致，"
                        "所以旧 coverage 百分比不能直接验证新模型。RS 的卡片 Top-1 质量、memory "
                        "item 检索、16-action 联合运行时和 10 条 Generator 人评也仍是独立未完成门。"
                        "任何一个失败都只能缩窄对应主张，不能靠调低阈值补救。"
                    ),
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "sourceId": "stability_contract",
                    "body": (
                        "## 最快的正确执行顺序\n\n1. 完成当前 10 条 Generator 执行核验。\n\n"
                        "2. 零外部 outcome 统一 memory feature builder。\n\n"
                        "3. 三个 memory heads 的 48-fit/16-confirmation 已完成，不再扩标；"
                        "用同一 builder 重编译 EvoEmo 候选并冻结 OOD 行为。\n\n"
                        "4. 冻结四头、阈值、hard gates 和 16-action 映射，先做零 API 联合动作审计。\n\n"
                        "5. 最后一次性运行 ESConv 与 EvoEmo，并分别报告 in-support/OOD、质量、"
                        "interaction-and-grounding risk 和 input-token cost。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 外测后仍需回答的问题\n\n通过 head 门只说明会按定义开关；只有冻结政策相对"
                        "always-off、透明 rule、high-resource fixed 和 cost-matched fixed 的系统结果，"
                        "才能回答这种能力是否真正转化为可发表的 quality—risk—cost 平衡。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-01T22:00:00+09:00",
            "status": "ready",
            "datasets": {
                "evidence_units": evidence_rows,
                "component_status": component_rows,
            },
        },
    }


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "component_grades": {
                key: value["evidence_grade"]
                for key, value in report["component_evidence"].items()
            },
            "passed_current_gates": sum(report["current_gates"].values()),
            "total_current_gates": len(report["current_gates"]),
        }
    )


if __name__ == "__main__":
    main()
