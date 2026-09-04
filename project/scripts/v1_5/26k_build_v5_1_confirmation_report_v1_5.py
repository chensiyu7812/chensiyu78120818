#!/usr/bin/env python3
"""Build the portable technical report for the frozen V5.1 confirmation."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pm_v1_5_v5_1_confirmation_analysis_v1"
ANALYSIS = OUT / "confirmation_analysis.json"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    result = json.loads(ANALYSIS.read_text())
    policies = result["policy_summary"]
    comparisons = result["learned_comparisons"]
    gates = result["gate_results"]
    components = result["component_diagnostic_not_a_gate"]

    learned = policies["learned_pm"]
    fixed = policies["component_fixed_high"]
    off = policies["always_off"]
    rule = policies["transparent_rule"]
    vs_fixed = comparisons["component_fixed_high"]
    vs_off = comparisons["always_off"]
    vs_rule = comparisons["transparent_rule"]
    prompt_reduction = 1 - learned["mean_prompt_tokens"] / fixed["mean_prompt_tokens"]

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    source_summary = {
        "id": "confirmation_summary",
        "label": "V5.1 frozen confirmation summary snapshot",
        "path": "report_source.sqlite",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT * FROM confirmation_summary;",
            "description": "Reads the precommitted system-gate summary derived from the verified blinded human exports.",
            "tables_used": ["confirmation_summary"],
        },
    }
    source_policy = {
        "id": "policy_summary",
        "label": "V5.1 policy-level confirmation snapshot",
        "path": "report_source.sqlite",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT policy, on_fraction, quality_vs_off, risk_rate, risk_count, mean_prompt_tokens, mean_total_tokens, fallback_count, critical_events FROM policy_summary;",
            "description": "Reads the selected-arm quality, grounding-risk, cost, fallback, and activity metrics for all four frozen policies.",
            "tables_used": ["policy_summary"],
        },
    }
    source_comparison = {
        "id": "quality_comparison",
        "label": "V5.1 paired policy-comparison snapshot",
        "path": "report_source.sqlite",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT comparator, better, worse, tie_or_same, quality_difference, quality_ci, risk_rate_difference, risk_ci, prompt_token_difference FROM quality_comparison;",
            "description": "Reads learned-PM minus comparator paired quality, risk, and prompt-token estimates.",
            "tables_used": ["quality_comparison"],
        },
    }
    source_component = {
        "id": "component_diagnostic",
        "label": "V5.1 component diagnostic snapshot",
        "path": "report_source.sqlite",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT component, learned_on_fraction, quality_vs_off, learned_risk_rate, fixed_risk_rate, learned_prompt_tokens, fixed_prompt_tokens FROM component_diagnostic;",
            "description": "Reads non-gating component-level diagnostics from the frozen system confirmation.",
            "tables_used": ["component_diagnostic"],
        },
    }
    source_gates = {
        "id": "gate_results",
        "label": "V5.1 pre-outcome gate result snapshot",
        "path": "report_source.sqlite",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": "SELECT gate_order, gate, observed, threshold_value, result FROM gate_results ORDER BY gate_order;",
            "description": "Reads every frozen single-use system gate, observed value, threshold, and pass/fail result.",
            "tables_used": ["gate_results"],
        },
    }
    source_contract = {
        "id": "frozen_contract",
        "label": "V5.1 pre-outcome system gate",
        "path": "frozen_contract.json",
    }
    sources = [
        source_summary,
        source_policy,
        source_comparison,
        source_component,
        source_gates,
        source_contract,
    ]

    policy_rows = []
    labels = {
        "learned_pm": "Learned PM",
        "component_fixed_high": "Fixed-high",
        "transparent_rule": "Transparent rule",
        "always_off": "Always-off",
    }
    for name in ("learned_pm", "component_fixed_high", "transparent_rule", "always_off"):
        row = policies[name]
        policy_rows.append(
            {
                "policy": labels[name],
                "on_fraction": row["selected_on_fraction"],
                "quality_vs_off": row["quality_mean_vs_always_off"],
                "risk_rate": row["material_risk_rate"],
                "risk_count": row["material_risk_count"],
                "mean_prompt_tokens": row["mean_prompt_tokens"],
                "mean_total_tokens": row["mean_total_tokens"],
                "fallback_count": row["fallback_count"],
                "critical_events": row["critical_grounding_event_count"],
            }
        )

    comparison_rows = []
    for name, comparison in (
        ("Fixed-high", vs_fixed),
        ("Always-off", vs_off),
        ("Transparent rule", vs_rule),
    ):
        comparison_rows.append(
            {
                "comparator": name,
                "better": comparison["quality_better"],
                "worse": comparison["quality_worse"],
                "tie_or_same": comparison["quality_tie_or_same_arm"],
                "quality_difference": comparison["quality_mean_difference"],
                "quality_ci": (
                    f"[{comparison['quality_cluster_bootstrap_95_ci'][0]:+.3f}, "
                    f"{comparison['quality_cluster_bootstrap_95_ci'][1]:+.3f}]"
                ),
                "risk_rate_difference": comparison["material_risk_rate_difference"],
                "risk_ci": (
                    f"[{comparison['risk_cluster_bootstrap_95_ci'][0]:+.3f}, "
                    f"{comparison['risk_cluster_bootstrap_95_ci'][1]:+.3f}]"
                ),
                "prompt_token_difference": comparison["mean_prompt_token_difference"],
            }
        )

    gate_rows = []
    for index, (name, gate) in enumerate(gates.items(), start=1):
        gate_rows.append(
            {
                "order": index,
                "gate": name,
                "observed": json.dumps(gate["value"], ensure_ascii=False),
                "threshold": json.dumps(gate["threshold"], ensure_ascii=False),
                "result": "PASS" if gate["pass"] else "FAIL",
            }
        )

    component_rows = []
    for component in ("MP", "MS", "ME", "RS"):
        row = components[component]
        component_rows.append(
            {
                "component": component,
                "learned_on_fraction": row["learned_on_fraction"],
                "quality_vs_off": row["learned_quality_mean_vs_always_off"],
                "learned_risk_rate": row["learned_material_risk_rate"],
                "fixed_risk_rate": row["fixed_high_material_risk_rate"],
                "learned_prompt_tokens": row["learned_mean_prompt_tokens"],
                "fixed_prompt_tokens": row["fixed_high_mean_prompt_tokens"],
            }
        )

    charts = [
        {
            "id": "risk_by_policy",
            "title": "四套策略选中回复的 material grounding-risk 率",
            "subtitle": "每套策略256个seed-level选中回复；风险为预定义interaction-and-grounding proxy。",
            "type": "bar",
            "dataset": "policy_summary",
            "sourceId": "policy_summary",
            "encodings": {
                "x": {"field": "policy", "type": "nominal", "label": "策略"},
                "y": {"field": "risk_rate", "type": "quantitative", "label": "Material risk率", "format": "percent"},
            },
            "yAxisTitle": "Material risk率",
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "prompt_by_policy",
            "title": "四套策略的平均 prompt token 成本",
            "subtitle": "实际选中臂，包含合法fallback造成的真实执行成本。",
            "type": "bar",
            "dataset": "policy_summary",
            "sourceId": "policy_summary",
            "encodings": {
                "x": {"field": "policy", "type": "nominal", "label": "策略"},
                "y": {"field": "mean_prompt_tokens", "type": "quantitative", "label": "平均prompt tokens", "format": "number"},
            },
            "yAxisTitle": "平均prompt tokens",
            "valueFormat": "number",
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "quality_comparison",
            "title": "Learned PM相对各对照的配对质量、风险与成本差",
            "subtitle": "质量与风险置信区间按64个counterfactual group聚类bootstrap；差值均为learned减对照。",
            "dataset": "quality_comparison",
            "sourceId": "quality_comparison",
            "layout": "full",
            "columns": [
                {"field": "comparator", "label": "对照", "type": "text"},
                {"field": "better", "label": "质量更好", "format": "number"},
                {"field": "worse", "label": "质量更差", "format": "number"},
                {"field": "tie_or_same", "label": "平局/同臂", "format": "number"},
                {"field": "quality_difference", "label": "平均质量差", "format": "number", "movement": True},
                {"field": "quality_ci", "label": "质量95% CI", "type": "text"},
                {"field": "risk_rate_difference", "label": "Risk率差", "format": "percent", "movement": True},
                {"field": "risk_ci", "label": "Risk 95% CI", "type": "text"},
                {"field": "prompt_token_difference", "label": "Prompt token差", "format": "number", "movement": True},
            ],
        },
        {
            "id": "gate_results",
            "title": "结果前冻结的全部系统门",
            "subtitle": "所有检查均须通过；本表未在读取人评结果后更改阈值。",
            "dataset": "gate_results",
            "sourceId": "gate_results",
            "layout": "full",
            "columns": [
                {"field": "order", "label": "顺序", "format": "number"},
                {"field": "gate", "label": "冻结检查", "type": "text"},
                {"field": "observed", "label": "观测值", "type": "text"},
                {"field": "threshold", "label": "冻结门槛", "type": "text"},
                {"field": "result", "label": "结果", "type": "text"},
            ],
        },
        {
            "id": "component_diagnostic",
            "title": "组件级结果仅作为诊断，不覆盖预注册系统结论",
            "subtitle": "每组件64个seed-level结果；质量为learned策略相对always-off的平均配对分。",
            "dataset": "component_diagnostic",
            "sourceId": "component_diagnostic",
            "layout": "full",
            "columns": [
                {"field": "component", "label": "组件", "type": "text"},
                {"field": "learned_on_fraction", "label": "Learned ON率", "format": "percent"},
                {"field": "quality_vs_off", "label": "质量 vs OFF", "format": "number", "movement": True},
                {"field": "learned_risk_rate", "label": "Learned risk率", "format": "percent"},
                {"field": "fixed_risk_rate", "label": "Fixed-high risk率", "format": "percent"},
                {"field": "learned_prompt_tokens", "label": "Learned prompt", "format": "number"},
                {"field": "fixed_prompt_tokens", "label": "Fixed-high prompt", "format": "number"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# PM V1.5 V5.1：内部冻结确认通过"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "confirmation_summary",
            "body": (
                "## 技术结论：冻结learned PM首次在未触碰确认集上通过完整QRC系统门\n\n"
                f"相对fixed-high，learned PM质量为**59胜、40负、157平/同臂**，平均差"
                f"**{signed(vs_fixed['quality_mean_difference'])}**，64组聚类bootstrap 95% CI "
                f"**[{vs_fixed['quality_cluster_bootstrap_95_ci'][0]:+.3f}, {vs_fixed['quality_cluster_bootstrap_95_ci'][1]:+.3f}]**；"
                f"material risk为**{learned['material_risk_count']}/256**，fixed-high为**{fixed['material_risk_count']}/256**；"
                f"平均prompt tokens从**{fixed['mean_prompt_tokens']:.1f}**降到**{learned['mean_prompt_tokens']:.1f}**，减少**{pct(prompt_reduction)}**。\n\n"
                f"Learned PM开启**{pct(learned['selected_on_fraction'])}**，没有退化成全开或全关；"
                f"关键grounding事件为**{learned['critical_grounding_event_count']}**，fixed-high为**{fixed['critical_grounding_event_count']}**。"
                "因此可以冻结完整16动作运行时并进入sealed internal、ESConv和EvoEmo。"
            ),
        },
        {
            "id": "quality_result",
            "type": "markdown",
            "sourceId": "quality_comparison",
            "body": (
                "## Learned PM保住质量，但对fixed-high的非劣结论接近预设边界\n\n"
                f"相对fixed-high的质量95% CI下界为**{vs_fixed['quality_cluster_bootstrap_95_ci'][0]:+.3f}**，"
                "刚高于预设的−0.050门槛；点估计为正，但不能据此声称明确优于fixed-high。"
                f"相对always-off则为**{vs_off['quality_mean_difference']:+.3f}**，95% CI "
                f"**[{vs_off['quality_cluster_bootstrap_95_ci'][0]:+.3f}, {vs_off['quality_cluster_bootstrap_95_ci'][1]:+.3f}]**，"
                "说明资源并非整体无用。相对transparent rule的质量差为"
                f"**{vs_rule['quality_mean_difference']:+.3f}**，95% CI "
                f"**[{vs_rule['quality_cluster_bootstrap_95_ci'][0]:+.3f}, {vs_rule['quality_cluster_bootstrap_95_ci'][1]:+.3f}]**。"
            ),
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_comparison", "layout": "full"},
        {
            "id": "risk_result",
            "type": "markdown",
            "sourceId": "confirmation_summary",
            "body": (
                "## Risk方向更低并满足非劣，但当前样本不足以证明统计显著下降\n\n"
                f"Learned PM的material-risk率为**{pct(learned['material_risk_rate'])}**，fixed-high为"
                f"**{pct(fixed['material_risk_rate'])}**，点差**{pct(vs_fixed['material_risk_rate_difference'])}**；"
                f"95% CI为**[{pct(vs_fixed['risk_cluster_bootstrap_95_ci'][0])}, {pct(vs_fixed['risk_cluster_bootstrap_95_ci'][1])}]**。"
                "区间跨0，所以准确表述是‘方向更低且未见风险增加’，不是‘已证明显著降低’。"
                "Learned PM仅出现1个fabricated-recall或内部标签泄漏事件，低于预设上限2，也低于fixed-high的6个。"
            ),
        },
        {"id": "risk_chart_block", "type": "chart", "chartId": "risk_by_policy", "layout": "full"},
        {
            "id": "cost_result",
            "type": "markdown",
            "sourceId": "confirmation_summary",
            "body": (
                "## 成本收益清晰，并非通过全关获得\n\n"
                f"Learned PM平均prompt tokens比fixed-high少**{fixed['mean_prompt_tokens'] - learned['mean_prompt_tokens']:.1f}**，"
                f"相对减少**{pct(prompt_reduction)}**；比transparent rule少**{-vs_rule['mean_prompt_token_difference']:.1f}**。"
                f"同时learned策略仍开启**{pct(learned['selected_on_fraction'])}**的候选，且相对always-off质量为正，"
                "因此成本下降不是‘所有资源永远关闭’造成的伪收益。"
            ),
        },
        {"id": "cost_chart_block", "type": "chart", "chartId": "prompt_by_policy", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## 范围、数据与指标定义\n\n"
                "本次是**内部FRESH_CONFIRMATION**：128个state-candidate unit、64个counterfactual group、每个unit两个冻结生成seed，共256个ON/OFF质量对和512个单臂risk审核。"
                "四套策略从同一批冻结回复中选择各自的臂。Quality采用同状态盲态A/B/tie；material risk仅指预定义的interaction-and-grounding事件，不是临床安全总风险；cost采用API报告的实际tokens。"
                "质量和risk由不同主评分别完成；成本、路由、fallback、重复和数据泄漏由程序测量。"
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "sourceId": "frozen_contract",
            "body": (
                "## 方法与停止规则\n\n"
                "特征、四个L2 logistic head、scaler、0.5阈值、候选检索器、typed executor、Llama-3.1-8B生成器、确认集和通过门均在读取确认回复与人评前冻结。"
                "不确定性通过对64个counterfactual group进行50,000次percentile cluster bootstrap估计。"
                "确认结果不得用于修阈值、换encoder、补训练样本或改prompt；通过后只允许进入一次sealed/internal/external比较。"
            ),
        },
        {"id": "gates_table_block", "type": "table", "tableId": "gate_results", "layout": "full"},
        {
            "id": "component_limit",
            "type": "markdown",
            "sourceId": "component_diagnostic",
            "body": (
                "## 系统通过不等于四个单头都已成为优秀语义分类器\n\n"
                "系统级QRC门是预注册主结论，组件拆分仅用于解释边界。MS贡献了最强的内部质量信号；ME在本确认集相对always-off的组件级质量均值为负。"
                "这不推翻系统门，但禁止把论文写成‘每种记忆在各自场景均稳定提高质量’，也禁止重新依据ME结果修改模型。"
            ),
        },
        {"id": "component_table_block", "type": "table", "tableId": "component_diagnostic", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限、稳健性与可主张边界\n\n"
                "质量和risk仍是人类测量代理，不是客观gold；本确认集没有另做重复双标，量表可靠性来自此前FIT overlap与集中裁决。"
                "风险事件只有23条，统计功效有限；quality-vs-fixed-high下界距门槛仅0.007。"
                "当前只证明内部构造环境中的一次冻结系统确认，没有证明ESConv/EvoEmo外部泛化、长期用户获益、临床安全或任意未来Bank/记忆库无需重训。"
                "后续可在完全冻结后加入独立LLM judge作为敏感性分析，但不能改变本次人评主结论。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 下一步：停止内部修补，进入一次外部泛化检验\n\n"
                "1. 冻结当前完整16动作runtime和本次PASS记录。\n"
                "2. 运行SEALED_INTERNAL_TEST，验证未参与任何开发的人造能力格子。\n"
                "3. 在ESConv上评估RS与memory-unavailable/off路径。\n"
                "4. 在EvoEmo上评估MP_PROFILE、MS_SESSION、ME_REUSABLE_OUTCOME及in-support coverage。\n"
                "5. 全量报告自动cost/coverage/routing，固定LLM judge作次要敏感性分析，并对外部样本做分层盲态人评。\n"
                "6. 所有sealed和外部结果只进入论文，不再回流修改PM。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 外部阶段仍需回答的问题\n\n"
                "- 质量非劣的窄裕量能否在ESConv/EvoEmo复现？\n"
                "- EvoEmo中各记忆组件的in-support coverage有多高，OOD时是否正确abstain/off？\n"
                "- ME的内部负向诊断是样本偶然、executor利用不足，还是可迁移性边界？\n"
                "- 人评、独立LLM judge与客观cost是否给出同方向结论？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "PM V1.5 V5.1：内部冻结确认通过",
            "description": "冻结learned PM相对fixed-high、always-off与transparent rule的quality-risk-cost确认结果。",
            "generatedAt": generated_at,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "policy_summary": policy_rows,
                "quality_comparison": comparison_rows,
                "gate_results": gate_rows,
                "component_diagnostic": component_rows,
            },
        },
        "sources": sources,
        "package_info": {},
    }

    database = OUT / "report_source.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE confirmation_summary (status TEXT, learned_on_fraction REAL, quality_diff_vs_fixed REAL, quality_lower_ci_vs_fixed REAL, quality_diff_vs_off REAL, quality_lower_ci_vs_off REAL, learned_risk_rate REAL, fixed_risk_rate REAL, risk_diff_vs_fixed REAL, risk_upper_ci_vs_fixed REAL, prompt_reduction_vs_fixed REAL, learned_critical_events INTEGER, fixed_critical_events INTEGER)"
        )
        connection.execute(
            "INSERT INTO confirmation_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result["status"],
                learned["selected_on_fraction"],
                vs_fixed["quality_mean_difference"],
                vs_fixed["quality_cluster_bootstrap_95_ci"][0],
                vs_off["quality_mean_difference"],
                vs_off["quality_cluster_bootstrap_95_ci"][0],
                learned["material_risk_rate"],
                fixed["material_risk_rate"],
                vs_fixed["material_risk_rate_difference"],
                vs_fixed["risk_cluster_bootstrap_95_ci"][1],
                prompt_reduction,
                learned["critical_grounding_event_count"],
                fixed["critical_grounding_event_count"],
            ),
        )
        connection.execute(
            "CREATE TABLE policy_summary (policy TEXT, on_fraction REAL, quality_vs_off REAL, risk_rate REAL, risk_count INTEGER, mean_prompt_tokens REAL, mean_total_tokens REAL, fallback_count INTEGER, critical_events INTEGER)"
        )
        connection.executemany(
            "INSERT INTO policy_summary VALUES (:policy, :on_fraction, :quality_vs_off, :risk_rate, :risk_count, :mean_prompt_tokens, :mean_total_tokens, :fallback_count, :critical_events)",
            policy_rows,
        )
        connection.execute(
            "CREATE TABLE quality_comparison (comparator TEXT, better INTEGER, worse INTEGER, tie_or_same INTEGER, quality_difference REAL, quality_ci TEXT, risk_rate_difference REAL, risk_ci TEXT, prompt_token_difference REAL)"
        )
        connection.executemany(
            "INSERT INTO quality_comparison VALUES (:comparator, :better, :worse, :tie_or_same, :quality_difference, :quality_ci, :risk_rate_difference, :risk_ci, :prompt_token_difference)",
            comparison_rows,
        )
        connection.execute(
            "CREATE TABLE component_diagnostic (component TEXT, learned_on_fraction REAL, quality_vs_off REAL, learned_risk_rate REAL, fixed_risk_rate REAL, learned_prompt_tokens REAL, fixed_prompt_tokens REAL)"
        )
        connection.executemany(
            "INSERT INTO component_diagnostic VALUES (:component, :learned_on_fraction, :quality_vs_off, :learned_risk_rate, :fixed_risk_rate, :learned_prompt_tokens, :fixed_prompt_tokens)",
            component_rows,
        )
        connection.execute(
            "CREATE TABLE gate_results (gate_order INTEGER, gate TEXT, observed TEXT, threshold_value TEXT, result TEXT)"
        )
        connection.executemany(
            "INSERT INTO gate_results VALUES (:order, :gate, :observed, :threshold, :result)",
            gate_rows,
        )
        connection.commit()
    finally:
        connection.close()

    (OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(CONTRACT, OUT / "frozen_contract.json")
    print(json.dumps({"status": "READY_FOR_PORTABLE_REPORT_DELIVERY", "artifact": str(OUT / "artifact.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
