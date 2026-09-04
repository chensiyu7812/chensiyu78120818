#!/usr/bin/env python3
"""Build the portable technical report for the finalized V5.2 FIT analysis."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3


ROOT = Path(__file__).resolve().parents[2]
FIT_DIR = ROOT / "outputs/pm_v1_5_v5_2_fit_final_training_v1"
BGE_DIR = ROOT / "outputs/pm_v1_5_v5_2_bge_challenger_v1"
PARETO_DIR = ROOT / "outputs/pm_v1_5_v5_2_oof_pareto_diagnostic_v1"
OUT = ROOT / "outputs/pm_v1_5_v5_2_fit_final_report_v1"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    fit = json.loads((FIT_DIR / "final_fit_report.json").read_text(encoding="utf-8"))
    bge = json.loads((BGE_DIR / "challenger_report.json").read_text(encoding="utf-8"))
    pareto = json.loads((PARETO_DIR / "diagnostic_report.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIT_DIR / "final_fit_report.json", OUT / "final_fit_report.json")
    shutil.copy2(BGE_DIR / "challenger_report.json", OUT / "bge_challenger_report.json")
    shutil.copy2(PARETO_DIR / "diagnostic_report.json", OUT / "oof_pareto_diagnostic.json")

    head_rows = []
    head_long = []
    for component in ("MP", "MS", "ME", "RS"):
        primary = fit["heads"][component]
        semantic = bge["heads"][component]
        transparent = primary["transparent_rule_metrics"]["balanced_accuracy"]
        row = {
            "component": component,
            "positive_states": primary["positive_units"],
            "nonpositive_states": primary["nonpositive_units"],
            "primary_balanced_accuracy": primary["five_seed_ba_mean"],
            "bge_balanced_accuracy": semantic["five_seed_ba_mean"],
            "transparent_balanced_accuracy": transparent,
            "primary_leave_family_out_ba": primary["leave_semantic_family_out"]["balanced_accuracy"],
            "primary_status": primary["status"],
        }
        head_rows.append(row)
        for model, value in (
            ("Transparent logistic", row["primary_balanced_accuracy"]),
            ("BGE challenger", row["bge_balanced_accuracy"]),
            ("Frozen rule", row["transparent_balanced_accuracy"]),
        ):
            head_long.append({
                "component": component,
                "model": model,
                "balanced_accuracy": value,
                "positive_states": row["positive_states"],
                "nonpositive_states": row["nonpositive_states"],
            })

    outcome_rows = []
    for component in ("MP", "MS", "ME", "RS"):
        value = fit["full_outcome_summary"][component]
        q = value["quality_winner_arm"]
        risk = value["on_material_risk"]
        function = value["on_resource_functionally_contributed"]
        outcome_rows.append({
            "component": component,
            "quality_on_wins": q.get("ON", 0),
            "quality_off_wins": q.get("OFF", 0),
            "quality_ties": q.get("tie", 0),
            "on_material_risk_rate": risk.get("yes", 0) / value["seed_pairs"],
            "function_rate": function.get("yes", 0) / value["seed_pairs"],
            "mean_incremental_tokens": value["mean_incremental_total_tokens"],
            "positive_states": value["positive_states"],
            "nonpositive_states": value["nonpositive_states"],
        })

    policy_rows = []
    policy_names = {
        "learned_grouped_oof": "Learned PM (OOF)",
        "transparent_rule": "Transparent rule",
        "component_fixed_high": "Fixed-high",
        "always_off": "Always-off",
    }
    for key in ("learned_grouped_oof", "transparent_rule", "component_fixed_high", "always_off"):
        row = pareto["policies"][key]
        policy_rows.append({
            "policy": policy_names[key],
            "on_fraction": row["requested_on_fraction"],
            "risk_rate": row["material_risk_rate"],
            "mean_prompt_tokens": row["mean_prompt_tokens"],
            "mean_total_tokens": row["mean_total_tokens"],
            "quality_better_vs_fixed_high": row["quality_versus_component_fixed_high"].get("better", 0),
            "quality_worse_vs_fixed_high": row["quality_versus_component_fixed_high"].get("worse", 0),
            "quality_tie_vs_fixed_high": row["quality_versus_component_fixed_high"].get("tie", 0),
        })

    learned = pareto["policies"]["learned_grouped_oof"]
    headline = pareto["headline"]
    database_path = OUT / "report_source.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        for table_name, table_rows in (
            ("head_metrics", head_rows),
            ("head_ba_long", head_long),
            ("component_outcomes", outcome_rows),
            ("policy_summary", policy_rows),
        ):
            columns = list(table_rows[0])
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            definitions = []
            for column in columns:
                value = table_rows[0][column]
                sql_type = "REAL" if isinstance(value, (int, float)) else "TEXT"
                definitions.append(f'"{column}" {sql_type}')
            connection.execute(f"CREATE TABLE {table_name} ({', '.join(definitions)})")
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                [[row[column] for column in columns] for row in table_rows],
            )
        connection.commit()
    finally:
        connection.close()

    def sql_source(source_id: str, label: str, table: str, sql: str) -> dict[str, object]:
        return {
            "id": source_id,
            "label": label,
            "path": "report_source.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql,
                "description": f"Reads the reviewed V5.2 {table} report snapshot.",
                "tables_used": [table],
            },
        }

    sources = [
        {"id": "fit", "label": "V5.2 final FIT aggregation and frozen primary training", "path": "final_fit_report.json"},
        {"id": "bge", "label": "V5.2 one-shot BGE-small challenger", "path": "bge_challenger_report.json"},
        {"id": "pareto", "label": "V5.2 grouped-OOF system Pareto diagnostic", "path": "oof_pareto_diagnostic.json"},
        sql_source("head_ba", "V5.2 head balanced-accuracy comparison", "head_ba_long", "SELECT component, model, balanced_accuracy, positive_states, nonpositive_states FROM head_ba_long;"),
        sql_source("head_metrics_table", "V5.2 primary head qualification metrics", "head_metrics", "SELECT * FROM head_metrics;"),
        sql_source("policy_summary", "V5.2 grouped-OOF policy summary", "policy_summary", "SELECT * FROM policy_summary;"),
        sql_source("component_outcomes", "V5.2 component outcome summary", "component_outcomes", "SELECT * FROM component_outcomes;"),
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# PM V1.5 V5.2 FIT：系统已有Pareto信号，但尚未完成独立确认"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "fit",
            "body": (
                "## 技术结论：不能再概括成‘四个组件都没学会’\n\n"
                "78条最终分歧裁决已回写到完整实验粒度：**512对质量、1024条risk、512条资源做功、256个Step1状态**；"
                "所有ITT对有效、零fallback、零uncertain。四个head均未通过全部严格晋级门，但失败原因不同："
                f"MP的grouped-OOF BA为**{fit['heads']['MP']['five_seed_ba_mean']:.3f}**，已形成实质分类信号，只败在预注册Brier校准门；"
                f"MS有**54/64**个正例，关闭组不足；ME与RS的冻结特征过于粗，无法表达大量同特征异标签状态。"
            ),
        },
        {
            "id": "pareto_result",
            "type": "markdown",
            "sourceId": "pareto",
            "body": (
                "## 组合PM在FIT开发面实现了目标QRC方向，明显好于固定规则\n\n"
                f"相对fixed-high，冻结的cross-fitted learned PM质量为**{headline['learned_quality_better_vs_fixed_high']}胜、"
                f"{headline['learned_quality_worse_vs_fixed_high']}负、{headline['learned_quality_tie_vs_fixed_high']}平/同臂**，"
                f"material risk减少**{pct(headline['learned_risk_reduction_vs_fixed_high'])}**，prompt tokens减少**{pct(headline['learned_prompt_token_reduction_vs_fixed_high'])}**。"
                f"相对always-off质量为**{headline['learned_quality_better_vs_always_off']}胜、{headline['learned_quality_worse_vs_always_off']}负**。"
                "透明规则相对fixed-high只有**11胜、82负**。这说明低容量学习策略已有意义，但仍只是FIT内grouped-OOF开发证据。"
            ),
        },
        {"id": "policy_table_block", "type": "table", "tableId": "policy_summary", "layout": "full"},
        {
            "id": "head_result",
            "type": "markdown",
            "sourceId": "fit",
            "body": (
                "## 单头门揭示了数据与表示边界，而不是一个统一失败\n\n"
                "MP已超过BA、recall、specificity、跨族和正负组门，只因class-balanced logistic的概率未胜prevalence Brier而不晋级。"
                "MS的资源版在128对中质量125胜、3负且做功128/128，导致只有5个独立负例组；它更像‘资格候选通常应开’，而不是有充足正负支持的价值分类任务。"
                "ME有39/64正例但相同粗特征格内正负混杂；RS的原子动作子型没有进入模型特征，导致不同动作的收益被压成少数相同向量。"
            ),
        },
        {"id": "head_chart_block", "type": "chart", "chartId": "head_balanced_accuracy", "layout": "full"},
        {"id": "head_table_block", "type": "table", "tableId": "head_metrics", "layout": "full"},
        {
            "id": "bge_result",
            "type": "markdown",
            "sourceId": "bge",
            "body": (
                "## BGE-small没有补上当前缺口，不应继续换encoder搜索\n\n"
                f"一次性冻结challenger的BA为MP **{bge['heads']['MP']['five_seed_ba_mean']:.3f}**、"
                f"MS **{bge['heads']['MS']['five_seed_ba_mean']:.3f}**、ME **{bge['heads']['ME']['five_seed_ba_mean']:.3f}**、"
                f"RS **{bge['heads']['RS']['five_seed_ba_mean']:.3f}**；无一通过全部门。"
                "通用current/candidate cosine没有编码‘偏好怎样改变回复’、‘哪种原子策略有用’或‘过去结果是否真正迁移’。"
                "因此BAAI可保留为负向challenger证据，不能把下一步变成Qwen/NLI/embedding轮换。"
            ),
        },
        {
            "id": "step2_result",
            "type": "markdown",
            "sourceId": "fit",
            "body": (
                "## Step2已经局部稳定，剩余失败集中在两类资源合同\n\n"
                "MS做功128/128、ME做功117/128，说明backend-locked历史出处修复有效；它们不再是过去那种大面积来源丢失。"
                "MP仅5/128做功，主因practical constraint只是向用户声明约束、没有改变后续回复；RS为75/128做功，事实凝练和对话开场子型仍频繁未执行。"
                "这些是冻结executor的机制边界，应在报告中分开披露，不能反写成Step1语义理解失败。"
            ),
        },
        {"id": "outcome_table_block", "type": "table", "tableId": "component_outcomes", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 范围、数据与指标定义\n\n"
                "本报告是内部EFFECT_FIT开发分析，不是独立确认。每组件64个state-candidate unit，每状态两个冻结生成seed；"
                "质量是盲态A/B/tie代理，material risk仅覆盖interaction-and-grounding构念，做功只回答资源是否可区分地改变回复，cost来自实际token用量。"
                "Step1 hard gold严格沿用结果前合同：两seed至少一次ON实质胜、无OFF实质胜、无uncertain、且ON臂无material risk；做功只作Step2机制指标。"
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 方法与稳健性\n\n"
                "主模型为5–7维透明特征、StandardScaler与L2 logistic；五个固定fold seed按V5.2 canonical visible-surface group分组。"
                "第三评审只覆盖两位评审的35个质量、30个risk和13个做功分歧，再传播回exact-duplicate全粒度；未把78条当独立训练样本。"
                "BGE challenger只执行一次、使用结果前冻结的两个cosine视图。OOF Pareto比较不读取confirmation或external，但仍与FIT标签同源，因此只能支持下一步确认，不能替代确认。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限与可主张边界\n\n"
                "人评是blind reviewer proxy而非客观gold；risk主评与独立复评阈值差异较大，第三裁决仅解决预先抽取的overlap分歧。"
                "MP的高BA尚未得到正确概率校准；MS负例支持不足；ME/RS特征碰撞表明当前低容量表示无法做细粒度收益预测。"
                "系统Pareto是cross-fitted开发结果，不证明ESConv/EvoEmo泛化，也不证明任意新Bank或记忆库无需重训。"
            ),
        },
        {
            "id": "next",
            "type": "markdown",
            "body": (
                "## 下一步：冻结当前PM，做一次新的内容独立系统确认\n\n"
                "1. 不再修改V5.2 executor、标签、特征、阈值或BAAI表示。\n"
                "2. 冻结当前四个primary head和完整16动作组合规则。\n"
                "3. 旧V5.1 FRESH/SEALED与EvoEmo结果已被读取，不能冒充V5.2 untouched confirmation；必须建立新的内容独立确认身份。\n"
                "4. 确认主比较仍为learned PM、fixed-high、transparent rule和always-off；主门是系统级quality非劣、risk不增加、cost降低和非退化开关。\n"
                "5. 使用完整自动cost/routing，固定LLM judge作敏感性，并只对预注册分层样本做人评；结果无论好坏都停止内部循环。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 独立确认仍需回答的问题\n\n"
                "- MP的分类信号能否在新语义表面保持，而不依赖当前偏好/资料比例？\n"
                "- MS在真实存在关闭机会时是否仍能区分，而不是退化为资格后全开？\n"
                "- ME/RS的系统组合收益能否在单头BA有限时仍稳定超过透明规则？\n"
                "- 修复后的MS/ME来源保持能否在EvoEmo新用户历史上复现？"
            ),
        },
    ]

    charts = [
        {
            "id": "head_balanced_accuracy",
            "title": "四组件Step1的grouped-OOF balanced accuracy",
            "subtitle": "64个state-candidate unit/组件；透明primary、冻结BGE challenger与透明规则。",
            "type": "bar",
            "dataset": "head_ba_long",
            "sourceId": "head_ba",
            "encodings": {
                "x": {"field": "component", "type": "nominal", "label": "组件"},
                "y": {"field": "balanced_accuracy", "type": "quantitative", "label": "Balanced accuracy", "format": "number"},
                "color": {"field": "model", "type": "nominal", "label": "方法"},
            },
            "yAxisTitle": "Balanced accuracy",
            "valueFormat": "number",
            "layout": "full",
        }
    ]
    tables = [
        {
            "id": "policy_summary",
            "title": "FIT开发面四套策略的质量、risk与成本",
            "subtitle": "每套策略512个seed-level选择；learned为五seed grouped-OOF决策。",
            "dataset": "policy_summary",
            "sourceId": "policy_summary",
            "defaultSort": {"field": "policy", "direction": "asc"},
            "columns": [
                {"field": "policy", "label": "策略", "type": "text"},
                {"field": "on_fraction", "label": "ON率", "format": "percent"},
                {"field": "quality_better_vs_fixed_high", "label": "质量胜fixed", "format": "number"},
                {"field": "quality_worse_vs_fixed_high", "label": "质量负fixed", "format": "number"},
                {"field": "risk_rate", "label": "Material risk率", "format": "percent"},
                {"field": "mean_prompt_tokens", "label": "平均prompt tokens", "format": "number"},
                {"field": "mean_total_tokens", "label": "平均总tokens", "format": "number"},
            ],
        },
        {
            "id": "head_metrics",
            "title": "四组件单头资格诊断",
            "subtitle": "BA为五个固定fold seed均值；LOFO按冻结语义族留一族。",
            "dataset": "head_metrics",
            "sourceId": "head_metrics_table",
            "defaultSort": {"field": "component", "direction": "asc"},
            "columns": [
                {"field": "component", "label": "组件", "type": "text"},
                {"field": "positive_states", "label": "正例", "format": "number"},
                {"field": "nonpositive_states", "label": "非正例", "format": "number"},
                {"field": "primary_balanced_accuracy", "label": "Primary BA", "format": "number"},
                {"field": "bge_balanced_accuracy", "label": "BGE BA", "format": "number"},
                {"field": "transparent_balanced_accuracy", "label": "规则 BA", "format": "number"},
                {"field": "primary_leave_family_out_ba", "label": "Primary LOFO BA", "format": "number"},
                {"field": "primary_status", "label": "冻结状态", "type": "text"},
            ],
        },
        {
            "id": "component_outcomes",
            "title": "V5.2各组件的直接效应、risk、做功与成本",
            "subtitle": "每组件128个seed-level ON/OFF对；做功只审核ON臂。",
            "dataset": "component_outcomes",
            "sourceId": "component_outcomes",
            "defaultSort": {"field": "component", "direction": "asc"},
            "columns": [
                {"field": "component", "label": "组件", "type": "text"},
                {"field": "quality_on_wins", "label": "ON质量胜", "format": "number"},
                {"field": "quality_off_wins", "label": "OFF质量胜", "format": "number"},
                {"field": "quality_ties", "label": "Tie", "format": "number"},
                {"field": "on_material_risk_rate", "label": "ON risk率", "format": "percent"},
                {"field": "function_rate", "label": "做功率", "format": "percent"},
                {"field": "mean_incremental_tokens", "label": "增量总tokens", "format": "number"},
            ],
        },
    ]
    datasets = {
        "head_metrics": head_rows,
        "head_ba_long": head_long,
        "component_outcomes": outcome_rows,
        "policy_summary": policy_rows,
    }
    (OUT / "report_data_snapshot.json").write_text(
        json.dumps(datasets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": "PM V1.5 V5.2 FIT：系统已有Pareto信号，但尚未完成独立确认",
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "package_info": {"generated_by": "scripts/v1_5/27g_build_v5_2_fit_report_v1_5.py"},
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ARTIFACT_READY", "artifact": str(OUT / "artifact.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
