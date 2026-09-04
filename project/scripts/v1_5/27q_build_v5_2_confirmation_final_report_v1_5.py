#!/usr/bin/env python3
"""Build the portable technical report for the frozen V5.2 confirmation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sqlite3

from metacom_pm.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "outputs/pm_v1_5_v5_2_confirmation_analysis_v1/confirmation_analysis.json"
SENSITIVITY = (
    ROOT
    / "outputs/pm_v1_5_v5_2_confirmation_analysis_primary_sensitivity_v1/confirmation_analysis.json"
)
LABELS = (
    ROOT
    / "outputs/pm_v1_5_v5_2_confirmation_final_labels_v1/label_resolution_manifest.json"
)
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_final_report_v1",
    )
    args = parser.parse_args()
    analysis = read_json(ANALYSIS)
    sensitivity = read_json(SENSITIVITY)
    labels = read_json(LABELS)
    contract = read_json(CONTRACT)
    if analysis["status"] != "FAIL_SINGLE_USE_V5_2_SYSTEM_GATE":
        raise RuntimeError("unexpected frozen confirmation status")
    if sensitivity["status"] != "FAIL_SINGLE_USE_V5_2_SYSTEM_GATE":
        raise RuntimeError("primary-only sensitivity did not reproduce the gate result")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy_labels = {
        "always_off": "Always-off",
        "component_fixed_high": "Fixed-high",
        "transparent_rule": "Transparent rule",
        "learned_pm": "Learned PM",
    }
    policy_rows = []
    for key, label in policy_labels.items():
        row = analysis["policy_summary"][key]
        policy_rows.append(
            {
                "policy_key": key,
                "policy": label,
                "selected_on_fraction": row["selected_on_fraction"],
                "quality_mean_vs_off": row["quality_mean_vs_always_off"],
                "material_risk_rate": row["material_risk_rate"],
                "mean_prompt_tokens": row["mean_prompt_tokens"],
                "mean_total_tokens": row["mean_total_tokens"],
                "critical_grounding_events": row["critical_grounding_event_count"],
            }
        )

    comparison_rows = []
    for key in ("component_fixed_high", "transparent_rule", "always_off"):
        row = analysis["learned_comparisons"][key]
        comparison_rows.append(
            {
                "baseline": policy_labels[key],
                "quality_difference": row["quality_mean_difference"],
                "quality_ci_low": row["quality_cluster_bootstrap_95_ci"][0],
                "quality_ci_high": row["quality_cluster_bootstrap_95_ci"][1],
                "risk_rate_difference": row["material_risk_rate_difference"],
                "risk_ci_low": row["risk_cluster_bootstrap_95_ci"][0],
                "risk_ci_high": row["risk_cluster_bootstrap_95_ci"][1],
                "prompt_token_difference": row["mean_prompt_token_difference"],
                "quality_better": row["quality_better"],
                "quality_worse": row["quality_worse"],
                "quality_tie_or_same": row["quality_tie_or_same_arm"],
            }
        )
    comparison_chart_rows = [
        {
            "baseline": row["baseline"],
            "metric": metric,
            "difference": row[field],
        }
        for row in comparison_rows
        for metric, field in (
            ("Quality difference", "quality_difference"),
            ("Risk-rate difference", "risk_rate_difference"),
        )
    ]

    gate_rows = []
    for gate, row in analysis["gate_results"].items():
        gate_rows.append(
            {
                "gate": gate,
                "value": json.dumps(row["value"], ensure_ascii=False),
                "threshold": json.dumps(row["threshold"], ensure_ascii=False),
                "result": "PASS" if row["pass"] else "FAIL",
            }
        )

    component_notes = {
        "MP": "开启率适中，但质量净增益接近零，且风险未低于全开。",
        "MS": "ON回复质量常有收益；learned PM漏开造成相对fixed-high的主要质量损失。",
        "ME": "当前最稳定的记忆收益组件；风险低，且多数状态不因关闭受损。",
        "RS": "路由显著降低全开风险，但同一‘可信联系人’卡产生4次虚构回忆。",
    }
    component_rows = []
    for component, row in analysis["component_diagnostic_not_a_gate"].items():
        component_rows.append(
            {
                "component": component,
                "seed_rows": row["n_seed_rows"],
                "learned_on_fraction": row["learned_on_fraction"],
                "quality_mean_vs_off": row["learned_quality_mean_vs_always_off"],
                "learned_risk_rate": row["learned_material_risk_rate"],
                "fixed_high_risk_rate": row["fixed_high_material_risk_rate"],
                "interpretation": component_notes[component],
            }
        )

    critical_rows = [
        {
            "component": "RS",
            "states": 2,
            "seed_level_events": 4,
            "card": "Contact one trusted person",
            "observed_failure": "回复虚构用户以前提到过名为 Sarah 的可信联系人或小组负责人。",
            "responsibility": "Step2/card-executor interaction；learned、fixed-high和透明规则均开启，PM没有过滤该组合。",
        }
    ]
    sensitivity_rows = [
        {
            "label_version": "Construct-valid final labels",
            "quality_vs_fixed_high": analysis["learned_comparisons"]["component_fixed_high"]["quality_mean_difference"],
            "quality_ci_low": analysis["learned_comparisons"]["component_fixed_high"]["quality_cluster_bootstrap_95_ci"][0],
            "critical_events": analysis["policy_summary"]["learned_pm"]["critical_grounding_event_count"],
            "result": analysis["status"],
        },
        {
            "label_version": "Primary-only sensitivity",
            "quality_vs_fixed_high": sensitivity["learned_comparisons"]["component_fixed_high"]["quality_mean_difference"],
            "quality_ci_low": sensitivity["learned_comparisons"]["component_fixed_high"]["quality_cluster_bootstrap_95_ci"][0],
            "critical_events": sensitivity["policy_summary"]["learned_pm"]["critical_grounding_event_count"],
            "result": sensitivity["status"],
        },
    ]

    sqlite_path = args.out_dir / "report_source.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE policy_outcomes (policy_key TEXT, policy TEXT, selected_on_fraction REAL, "
            "quality_mean_vs_off REAL, material_risk_rate REAL, mean_prompt_tokens REAL, "
            "mean_total_tokens REAL, critical_grounding_events INTEGER)"
        )
        connection.executemany(
            "INSERT INTO policy_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row.values()) for row in policy_rows],
        )
        connection.execute(
            "CREATE TABLE learned_comparisons (baseline TEXT, quality_difference REAL, "
            "quality_ci_low REAL, quality_ci_high REAL, risk_rate_difference REAL, "
            "risk_ci_low REAL, risk_ci_high REAL, prompt_token_difference REAL, "
            "quality_better INTEGER, quality_worse INTEGER, quality_tie_or_same INTEGER)"
        )
        connection.executemany(
            "INSERT INTO learned_comparisons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row.values()) for row in comparison_rows],
        )
        connection.execute(
            "CREATE TABLE comparison_chart (baseline TEXT, metric TEXT, difference REAL)"
        )
        connection.executemany(
            "INSERT INTO comparison_chart VALUES (?, ?, ?)",
            [tuple(row.values()) for row in comparison_chart_rows],
        )
        connection.execute(
            "CREATE TABLE gate_results (gate TEXT, value TEXT, threshold TEXT, result TEXT)"
        )
        connection.executemany(
            "INSERT INTO gate_results VALUES (?, ?, ?, ?)",
            [tuple(row.values()) for row in gate_rows],
        )
        connection.execute(
            "CREATE TABLE component_diagnostics (component TEXT, seed_rows INTEGER, "
            "learned_on_fraction REAL, quality_mean_vs_off REAL, learned_risk_rate REAL, "
            "fixed_high_risk_rate REAL, interpretation TEXT)"
        )
        connection.executemany(
            "INSERT INTO component_diagnostics VALUES (?, ?, ?, ?, ?, ?, ?)",
            [tuple(row.values()) for row in component_rows],
        )
        connection.execute(
            "CREATE TABLE critical_cases (component TEXT, states INTEGER, seed_level_events INTEGER, "
            "card TEXT, observed_failure TEXT, responsibility TEXT)"
        )
        connection.executemany(
            "INSERT INTO critical_cases VALUES (?, ?, ?, ?, ?, ?)",
            [tuple(row.values()) for row in critical_rows],
        )
        connection.execute(
            "CREATE TABLE label_sensitivity (label_version TEXT, quality_vs_fixed_high REAL, "
            "quality_ci_low REAL, critical_events INTEGER, result TEXT)"
        )
        connection.executemany(
            "INSERT INTO label_sensitivity VALUES (?, ?, ?, ?, ?)",
            [tuple(row.values()) for row in sensitivity_rows],
        )

    copies = {
        "confirmation_analysis.json": ANALYSIS,
        "primary_only_sensitivity.json": SENSITIVITY,
        "label_resolution_manifest.json": LABELS,
        "confirmation_contract.json": CONTRACT,
    }
    for target, source in copies.items():
        shutil.copy2(source, args.out_dir / target)

    sql_sources = {
        "policy": (
            "Frozen four-policy outcome summary",
            "SELECT * FROM policy_outcomes ORDER BY policy_key;",
            ["policy_outcomes"],
        ),
        "comparison": (
            "Learned PM contrasts against frozen baselines",
            "SELECT * FROM learned_comparisons ORDER BY baseline;",
            ["learned_comparisons"],
        ),
        "comparison_chart": (
            "Quality and risk differences against frozen baselines",
            "SELECT baseline, metric, difference FROM comparison_chart ORDER BY baseline, metric;",
            ["comparison_chart"],
        ),
        "gates": (
            "Pre-frozen V5.2 confirmation gates",
            "SELECT gate, value, threshold, result FROM gate_results ORDER BY gate;",
            ["gate_results"],
        ),
        "components": (
            "Per-component diagnostic outcomes",
            "SELECT * FROM component_diagnostics ORDER BY component;",
            ["component_diagnostics"],
        ),
        "critical": (
            "Critical grounding root-cause audit",
            "SELECT * FROM critical_cases;",
            ["critical_cases"],
        ),
        "sensitivity": (
            "Label-resolution sensitivity analysis",
            "SELECT * FROM label_sensitivity ORDER BY label_version;",
            ["label_sensitivity"],
        ),
    }
    sources = [
        {
            "id": source_id,
            "label": label,
            "path": "report_source.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": sql,
                "description": label,
                "tables_used": tables,
            },
        }
        for source_id, (label, sql, tables) in sql_sources.items()
    ]
    sources.extend(
        [
            {"id": "analysis", "label": "Frozen V5.2 confirmation analysis", "path": "confirmation_analysis.json"},
            {"id": "primary_sensitivity", "label": "Primary-only label sensitivity replay", "path": "primary_only_sensitivity.json"},
            {"id": "label_resolution", "label": "Final adjudication and construct-validity lineage", "path": "label_resolution_manifest.json"},
            {"id": "contract", "label": "Pre-frozen V5.2 one-shot confirmation contract", "path": "confirmation_contract.json"},
        ]
    )

    fixed = analysis["learned_comparisons"]["component_fixed_high"]
    rule = analysis["learned_comparisons"]["transparent_rule"]
    off = analysis["learned_comparisons"]["always_off"]
    prompt_reduction = analysis["gate_results"]["prompt_cost_reduction_vs_fixed_high"]["value"]
    title = "PM V1.5 V5.2 最终确认：学到了有意义的取舍，但完整系统门未通过"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": title,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "analysis",
                    "body": (
                        "## 技术摘要：应报告为 bounded / negative confirmation，而不是继续调到通过\n\n"
                        "冻结的一次性内容独立确认**未通过完整系统门**。失败只有两项：相对fixed-high的质量"
                        f"非劣下界为**{pct(fixed['quality_cluster_bootstrap_95_ci'][0])}**，低于预设-5%；"
                        "learned PM发生**4次**critical fabricated recall，高于上限2。\n\n"
                        "这不是PM没有学会。相对transparent rule，learned PM质量净差"
                        f"**+{pct(rule['quality_mean_difference'])}**、risk差**{pct(rule['material_risk_rate_difference'])}**、"
                        f"每回复少**{abs(rule['mean_prompt_token_difference']):.5g}**个prompt tokens；相对always-off质量"
                        f"**+{pct(off['quality_mean_difference'])}**。相对fixed-high，质量点估计仍为"
                        f"**+{pct(fixed['quality_mean_difference'])}**，risk下降**{pct(-fixed['material_risk_rate_difference'])}**，"
                        f"prompt减少**{pct(prompt_reduction)}**，但不确定性和critical risk阻止完整主张成立。"
                    ),
                },
                {
                    "id": "policy_finding",
                    "type": "markdown",
                    "sourceId": "policy",
                    "body": (
                        "## Learned PM形成了非退化的质量—风险—成本折中\n\n"
                        "learned PM在256个seed-level决策中开启130次（50.8%），既不是全开也不是全关。"
                        "它获得四套策略中最高的相对always-off质量均值，同时把fixed-high的material-risk率"
                        "从21.9%降到15.6%，并把平均prompt从236.2降到219.2 tokens。代价是仍高于"
                        "always-off的3.5%风险率；因此论文主张必须是平衡与选择，而不是绝对安全。"
                    ),
                },
                {"id": "quality_chart_block", "type": "chart", "chartId": "policy_quality", "layout": "full"},
                {
                    "id": "quality_chart_note",
                    "type": "markdown",
                    "body": (
                        "图中的质量值以同一状态的always-off回复为零点；正值代表策略选择的回复更常实质胜出。"
                        "Learned PM高于三个基线的点估计，但与fixed-high的64组cluster bootstrap区间仍跨过"
                        "预注册非劣界，因此只能称方向积极，不能称已证实非劣。"
                    ),
                },
                {"id": "risk_chart_block", "type": "chart", "chartId": "policy_risk", "layout": "full"},
                {
                    "id": "risk_chart_note",
                    "type": "markdown",
                    "body": (
                        "风险是interaction-and-grounding material-risk proxy，不是临床安全率。Learned PM"
                        "显著低于fixed-high，但仍选择了4条critical fabricated recall；低平均风险不能抵消"
                        "预冻结的严重事件上限。"
                    ),
                },
                {"id": "policy_table_block", "type": "table", "tableId": "policy_outcomes", "layout": "full"},
                {
                    "id": "baseline_contrasts",
                    "type": "markdown",
                    "sourceId": "comparison",
                    "body": (
                        "## 相比透明规则已经及格；相比全开仍缺稳定的质量证据\n\n"
                        "Learned PM相对transparent rule在质量、risk和prompt cost三个点估计上同时改善，且质量"
                        "95%区间下界为+3.9%。相对fixed-high则是一个更难的结论：67胜、59负、130平或同臂，"
                        "点估计略好，但MS漏开使部分反事实组出现系统性质量损失，区间下界降至-10.9%。"
                    ),
                },
                {"id": "comparison_chart_block", "type": "chart", "chartId": "learned_differences", "layout": "full"},
                {"id": "comparison_table_block", "type": "table", "tableId": "learned_comparisons", "layout": "full"},
                {
                    "id": "gate_finding",
                    "type": "markdown",
                    "sourceId": "gates",
                    "body": (
                        "## 预冻结系统门有11项通过、2项失败\n\n"
                        "通过项覆盖非退化活动、always-off质量、transparent-rule三指标、fixed-high风险和成本。"
                        "失败项不能用其他外测或事后阈值补救：fixed-high质量非劣下界和critical grounding"
                        "事件上限是生成与人评前已经写死的门。"
                    ),
                },
                {"id": "gate_table_block", "type": "table", "tableId": "gate_results", "layout": "full"},
                {
                    "id": "component_finding",
                    "type": "markdown",
                    "sourceId": "components",
                    "body": (
                        "## 失败不是四个组件轮流随机坏：MS漏开与RS执行失真是两个不同责任层\n\n"
                        "MS的ON回复在本确认中常带来质量收益，但Step1过度保守，关闭了fixed-high会赢的状态；"
                        "这是路由泛化问题。RS则相反：Step1和人工规则都允许同一张卡，Step2却把‘用户已识别"
                        "或愿意考虑的可信联系人’自由补成不存在的Sarah；这是card-executor交互和候选资格"
                        "问题。ME保持低风险与正质量，是当前最稳定的记忆组件；MP仍缺可区分增量。"
                    ),
                },
                {"id": "component_table_block", "type": "table", "tableId": "component_diagnostics", "layout": "full"},
                {
                    "id": "critical_finding",
                    "type": "markdown",
                    "sourceId": "critical",
                    "body": (
                        "## 四次critical事件来自两个状态、一个RS卡片和两个seed\n\n"
                        "四条回复都虚构用户过去提到过Sarah，而可见对话和授权资源均没有该人物；卡片本身还"
                        "明确禁止在没有证据时假设具体联系人。Learned PM、fixed-high和transparent rule都开启"
                        "了这张卡，所以这既是冻结Step2的真实失败，也说明Step1没有表达或学到该执行风险。"
                    ),
                },
                {"id": "critical_table_block", "type": "table", "tableId": "critical_cases", "layout": "full"},
                {
                    "id": "definitions",
                    "type": "markdown",
                    "sourceId": "contract",
                    "body": (
                        "## 范围、分析单位与指标定义\n\n"
                        "确认集含128个state × exact Rank-1 candidate单位、64个二元counterfactual group、"
                        "四组件各32个单位；每状态两个固定seed，ON/OFF同seed配对。策略在已生成的同一对回复"
                        "间选臂，不各自重新生成。质量得分为ON相对OFF的material A/B/tie方向；risk只覆盖"
                        "预定义interaction-and-grounding事件；cost取策略实际选中回复的API token usage。"
                        "95%区间对64个counterfactual group做50,000次percentile cluster bootstrap。"
                    ),
                },
                {
                    "id": "label_robustness",
                    "type": "markdown",
                    "sourceId": "label_resolution",
                    "body": (
                        "## 第三裁决含5处构念修正，但结论对修正不敏感\n\n"
                        "三条质量裁决把盲页未展示的授权历史误当成虚构；两条risk裁决把自然的来源说明误标为"
                        "内部标签泄漏。最终只做列举式构念修正，没有新评审、没有改回复或策略。使用未修正的"
                        "primary-only标签重放时，系统仍因同样两项门失败；因此正式结论不依赖这5处处理。"
                    ),
                },
                {"id": "sensitivity_table_block", "type": "table", "tableId": "label_sensitivity", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制与可支持的论文表述\n\n"
                        "本结果是内容独立的合成内部确认，不是ESConv/EvoEmo untouched external lockbox；后两者"
                        "已在旧版本暴露，只能作post-hoc repaired replication。来源说明使资源臂有时可被评审者"
                        "识别，因此只能称A/B位置盲化、策略选择盲化，不能声称完全treatment-blind。风险指标也"
                        "不是临床安全评价。当前可支持的主张是：低容量、可审计的PM学到了优于透明规则的保守"
                        "资源取舍，并呈现相对全开的风险/成本优势；不能主张完整系统已经通过预注册确认。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐下一步：停止内部调试循环，转入结果写作和隔离的未来修复\n\n"
                        "1. 将V5.2作为最终bounded/negative confirmation写入论文，不建立第二确认集。\n"
                        "2. 保留ESConv/EvoEmo仅作同一冻结系统的探索性或修复后replication，不用于救活本门。\n"
                        "3. 把MS漏开定义为Step1未来改进，把Sarah事件定义为RS card-executor安全缺陷；任何"
                        "修复必须升新版本并使用新考卷，不能回写V5.2。\n"
                        "4. 论文主比较保留always-off、fixed-high、transparent-rule和learned PM；V1.0只作"
                        "历史方法基线，不与冻结V5.2的独立确认统计混在同一主门里。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续研究问题\n\n"
                        "- 若新版本显式建模candidate × executor风险，能否在不牺牲MS收益的情况下消除RS严重事件？\n"
                        "- 在真正未暴露的纵向外部数据上，ME的低风险正收益能否复现？\n"
                        "- 更强语义表示是否只改善goal/function泛化，还是会增加不可审计捷径？"
                    ),
                },
            ],
            "charts": [
                {
                    "id": "policy_quality",
                    "title": "四种策略的质量均值（相对always-off）",
                    "subtitle": "256个seed-level策略选择；正值代表更常选择实质更好的回复。",
                    "type": "bar",
                    "dataset": "policy_outcomes",
                    "sourceId": "policy",
                    "encodings": {
                        "x": {"field": "policy", "type": "nominal", "label": "策略"},
                        "y": {"field": "quality_mean_vs_off", "type": "quantitative", "label": "质量均值", "format": "percent"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "policy_risk",
                    "title": "四种策略的material-risk率",
                    "subtitle": "interaction-and-grounding proxy；分母为每策略选中的256条回复。",
                    "type": "bar",
                    "dataset": "policy_outcomes",
                    "sourceId": "policy",
                    "encodings": {
                        "x": {"field": "policy", "type": "nominal", "label": "策略"},
                        "y": {"field": "material_risk_rate", "type": "quantitative", "label": "Risk率", "format": "percent"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "learned_differences",
                    "title": "Learned PM相对三个基线的质量与risk差",
                    "subtitle": "差值单位为比例点；risk负值更好，quality正值更好。",
                    "type": "bar",
                    "dataset": "comparison_chart",
                    "sourceId": "comparison_chart",
                    "encodings": {
                        "x": {"field": "baseline", "type": "nominal", "label": "比较基线"},
                        "y": {"field": "difference", "type": "quantitative", "label": "差值", "format": "percent"},
                        "color": {"field": "metric", "type": "nominal", "label": "指标"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "policy_outcomes",
                    "title": "冻结四策略的完整结果",
                    "subtitle": "质量以always-off为零点；risk与token均取策略实际选择的回复。",
                    "dataset": "policy_outcomes",
                    "sourceId": "policy",
                    "defaultSort": {"field": "policy", "direction": "asc"},
                    "columns": [
                        {"field": "policy", "label": "策略", "type": "text"},
                        {"field": "selected_on_fraction", "label": "ON率", "format": "percent"},
                        {"field": "quality_mean_vs_off", "label": "质量vs off", "format": "percent"},
                        {"field": "material_risk_rate", "label": "Risk率", "format": "percent"},
                        {"field": "mean_prompt_tokens", "label": "Prompt tokens", "format": "number"},
                        {"field": "critical_grounding_events", "label": "Critical事件", "format": "number"},
                    ],
                },
                {
                    "id": "learned_comparisons",
                    "title": "Learned PM相对基线的成对差异",
                    "subtitle": "质量与risk区间按64个反事实组聚类bootstrap；token为均值差。",
                    "dataset": "learned_comparisons",
                    "sourceId": "comparison",
                    "defaultSort": {"field": "baseline", "direction": "asc"},
                    "columns": [
                        {"field": "baseline", "label": "基线", "type": "text"},
                        {"field": "quality_difference", "label": "质量差", "format": "percent"},
                        {"field": "quality_ci_low", "label": "质量CI下界", "format": "percent"},
                        {"field": "quality_ci_high", "label": "质量CI上界", "format": "percent"},
                        {"field": "risk_rate_difference", "label": "Risk差", "format": "percent"},
                        {"field": "prompt_token_difference", "label": "Prompt差", "format": "number"},
                    ],
                },
                {
                    "id": "gate_results",
                    "title": "一次性V5.2系统门",
                    "subtitle": "所有门均在确认回复和人评结果产生前冻结。",
                    "dataset": "gate_results",
                    "sourceId": "gates",
                    "defaultSort": {"field": "result", "direction": "asc"},
                    "columns": [
                        {"field": "gate", "label": "门", "type": "text"},
                        {"field": "value", "label": "实际值", "type": "text"},
                        {"field": "threshold", "label": "阈值", "type": "text"},
                        {"field": "result", "label": "结果", "type": "text"},
                    ],
                },
                {
                    "id": "component_diagnostics",
                    "title": "四组件诊断（不作为单独晋级门）",
                    "subtitle": "每组件64个seed-level行；用于责任定位，不重新训练head。",
                    "dataset": "component_diagnostics",
                    "sourceId": "components",
                    "defaultSort": {"field": "component", "direction": "asc"},
                    "columns": [
                        {"field": "component", "label": "组件", "type": "text"},
                        {"field": "learned_on_fraction", "label": "Learned ON率", "format": "percent"},
                        {"field": "quality_mean_vs_off", "label": "质量vs off", "format": "percent"},
                        {"field": "learned_risk_rate", "label": "Learned risk", "format": "percent"},
                        {"field": "fixed_high_risk_rate", "label": "Fixed risk", "format": "percent"},
                        {"field": "interpretation", "label": "解释", "type": "text"},
                    ],
                },
                {
                    "id": "critical_cases",
                    "title": "Critical fabricated-recall根因",
                    "subtitle": "四个seed-level事件来自两个状态和同一RS卡。",
                    "dataset": "critical_cases",
                    "sourceId": "critical",
                    "defaultSort": {"field": "component", "direction": "asc"},
                    "columns": [
                        {"field": "component", "label": "组件", "type": "text"},
                        {"field": "states", "label": "状态", "format": "number"},
                        {"field": "seed_level_events", "label": "事件", "format": "number"},
                        {"field": "card", "label": "卡片", "type": "text"},
                        {"field": "observed_failure", "label": "观察到的失败", "type": "text"},
                        {"field": "responsibility", "label": "责任层", "type": "text"},
                    ],
                },
                {
                    "id": "label_sensitivity",
                    "title": "构念修正敏感性",
                    "subtitle": "修正后与primary-only重放均因相同两项门失败。",
                    "dataset": "label_sensitivity",
                    "sourceId": "sensitivity",
                    "defaultSort": {"field": "label_version", "direction": "asc"},
                    "columns": [
                        {"field": "label_version", "label": "标签版本", "type": "text"},
                        {"field": "quality_vs_fixed_high", "label": "质量差", "format": "percent"},
                        {"field": "quality_ci_low", "label": "CI下界", "format": "percent"},
                        {"field": "critical_events", "label": "Critical事件", "format": "number"},
                        {"field": "result", "label": "系统门", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "policy_outcomes": policy_rows,
                "learned_comparisons": comparison_rows,
                "comparison_chart": comparison_chart_rows,
                "gate_results": gate_rows,
                "component_diagnostics": component_rows,
                "critical_cases": critical_rows,
                "label_sensitivity": sensitivity_rows,
            },
        },
        "sources": sources,
        "package_info": {
            "generated_by": "scripts/v1_5/27q_build_v5_2_confirmation_final_report_v1_5.py",
            "source_notes": (
                "Audience=technical. Required structure mapped to technical summary, key findings, "
                "scope/definitions, methodology, robustness/limitations, next steps, and further questions. "
                "Chart map: policy quality and policy risk use separate categorical bars because the analytical "
                "questions differ and units should not share one axis; learned differences uses a grouped signed "
                "bar for quality versus risk-rate changes. Palette policies are single-root for single-series "
                "charts and hard two-root for the comparator chart. Exact gates, costs, components, critical "
                "cases, and sensitivity remain tables because audit lookup is the primary task."
            ),
        },
    }
    write_json(args.out_dir / "artifact.json", artifact)
    print({"status": "FINAL_REPORT_ARTIFACT_READY", "out_dir": str(args.out_dir)})


if __name__ == "__main__":
    main()
