#!/usr/bin/env python3
"""Build the canonical artifact for the V5.2 confirmation freeze report."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sqlite3

from metacom_pm.io import iter_jsonl, read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
FREEZE = (
    ROOT
    / "outputs/pm_v1_5_v5_2_confirmation_freeze_v1/freeze_and_data_quality_report.json"
)
CONTRACT = (
    ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json"
)
PLAN = ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1/freeze_manifest.json"
BINDINGS = (
    ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1/state_policy_bindings_private.jsonl"
)


def rows(path: Path) -> list[dict]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_freeze_report_v1",
    )
    args = parser.parse_args()
    freeze = read_json(FREEZE)
    contract = read_json(CONTRACT)
    plan = read_json(PLAN)
    bindings = rows(BINDINGS)
    if freeze["status"] != "PASS_V5_2_CONTENT_DISJOINT_CONFIRMATION_IDENTITY_FROZEN":
        raise RuntimeError("confirmation identity did not pass")
    if plan["status"] != "READY_FOR_SINGLE_V5_2_CONTENT_DISJOINT_CONFIRMATION_EXECUTION":
        raise RuntimeError("confirmation plan is not ready")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy_rows = []
    for component in ("MP", "MS", "ME", "RS"):
        subset = [row for row in bindings if row["component"] == component]
        policy_rows.append(
            {
                "component": component,
                "units": len(subset),
                "learned_on": sum(row["policy_decisions"]["learned_pm"] for row in subset),
                "transparent_on": sum(
                    row["policy_decisions"]["transparent_rule"] for row in subset
                ),
                "fixed_high_on": len(subset),
                "always_off_on": 0,
            }
        )
    policy_chart_rows = [
        {
            "component": row["component"],
            "policy": policy,
            "on_fraction": row[field] / row["units"],
            "on_count": row[field],
            "unit_count": row["units"],
        }
        for row in policy_rows
        for policy, field in (
            ("Learned PM", "learned_on"),
            ("Transparent rule", "transparent_on"),
        )
    ]
    exposure_rows = [
        {
            "old_split": split,
            "states": value["expected_states"],
            "exposed_states": value["exposed_states"],
            "reusable_as_v5_2_confirmation": "否",
        }
        for split, value in freeze["historical_exposure_audit"].items()
    ]
    shape_rows = [
        {
            "dimension": "State-candidate units",
            "count": 128,
            "rule": "四组件各32",
        },
        {
            "dimension": "Counterfactual groups",
            "count": 64,
            "rule": "每组HIGH与LOW_OR_NEUTRAL各1",
        },
        {
            "dimension": "Logic families",
            "count": 32,
            "rule": "每族4条",
        },
        {
            "dimension": "New topic families",
            "count": 8,
            "rule": "每主题16条",
        },
        {
            "dimension": "Paired ON/OFF calls",
            "count": 512,
            "rule": "128状态×2 seeds×2 arms",
        },
    ]
    sqlite_path = args.out_dir / "report_source.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE policy_on_fraction "
            "(component TEXT, policy TEXT, on_fraction REAL, on_count INTEGER, unit_count INTEGER)"
        )
        connection.executemany(
            "INSERT INTO policy_on_fraction VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row["component"],
                    row["policy"],
                    row["on_fraction"],
                    row["on_count"],
                    row["unit_count"],
                )
                for row in policy_chart_rows
            ],
        )
        connection.execute(
            "CREATE TABLE old_exposure "
            "(old_split TEXT, states INTEGER, exposed_states INTEGER, reusable_as_v5_2_confirmation TEXT)"
        )
        connection.executemany(
            "INSERT INTO old_exposure VALUES (?, ?, ?, ?)",
            [tuple(row.values()) for row in exposure_rows],
        )
        connection.execute(
            "CREATE TABLE confirmation_shape (dimension TEXT, count INTEGER, rule TEXT)"
        )
        connection.executemany(
            "INSERT INTO confirmation_shape VALUES (?, ?, ?)",
            [tuple(row.values()) for row in shape_rows],
        )
        connection.execute(
            "CREATE TABLE policy_distribution "
            "(component TEXT, units INTEGER, learned_on INTEGER, transparent_on INTEGER, "
            "fixed_high_on INTEGER, always_off_on INTEGER)"
        )
        connection.executemany(
            "INSERT INTO policy_distribution VALUES (?, ?, ?, ?, ?, ?)",
            [tuple(row.values()) for row in policy_rows],
        )
    sources = [
        {
            "id": "freeze",
            "label": "V5.2 confirmation identity and data-quality freeze",
            "path": "freeze_and_data_quality_report.json",
        },
        {
            "id": "contract",
            "label": "V5.2 one-shot confirmation contract",
            "path": "confirmation_contract.json",
        },
        {
            "id": "plan",
            "label": "V5.2 frozen confirmation call and policy manifest",
            "path": "confirmation_plan_manifest.json",
        },
        {
            "id": "policy_chart",
            "label": "Frozen learned and transparent policy ON distribution",
            "path": "report_source.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT component, policy, on_fraction, on_count, unit_count "
                    "FROM policy_on_fraction ORDER BY component, policy;"
                ),
                "description": "Reads the frozen per-component ON distribution used in the chart.",
                "tables_used": ["policy_on_fraction"],
            },
        },
        {
            "id": "old_exposure_table",
            "label": "Historical internal split exposure audit",
            "path": "report_source.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": "SELECT old_split, states, exposed_states, reusable_as_v5_2_confirmation FROM old_exposure ORDER BY old_split;",
                "description": "Reads the audited exposure status of prior internal experiment identities.",
                "tables_used": ["old_exposure"],
            },
        },
        {
            "id": "confirmation_shape_table",
            "label": "Frozen V5.2 confirmation identity shape",
            "path": "report_source.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": "SELECT dimension, count, rule FROM confirmation_shape ORDER BY dimension;",
                "description": "Reads the reviewed static shape of the new confirmation identity.",
                "tables_used": ["confirmation_shape"],
            },
        },
        {
            "id": "policy_distribution_table",
            "label": "Frozen per-component policy decision counts",
            "path": "report_source.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT component, units, learned_on, transparent_on, fixed_high_on, "
                    "always_off_on FROM policy_distribution ORDER BY component;"
                ),
                "description": "Reads the frozen ON counts for all four comparison policies.",
                "tables_used": ["policy_distribution"],
            },
        },
    ]
    title = "PM V1.5 V5.2：唯一内容独立确认实验已冻结"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": title,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "plan",
                    "body": (
                        "## 技术结论：可以开始一次性确认，但尚未产生确认结果\n\n"
                        "V5.2的新确认身份已通过零API静态门：**128个state-candidate unit、64个"
                        "counterfactual group、四组件各32条**；冻结策略在其中开启**65/128（50.8%）**，"
                        "不存在全开或全关退化。已经封存**512次**同状态、同seed的ON/OFF调用，成本上界"
                        "代理约**$0.122**。这些数字只证明实验可执行、数据不泄漏，不是质量、risk或cost结论。"
                    ),
                },
                {
                    "id": "old_identity",
                    "type": "markdown",
                    "sourceId": "freeze",
                    "body": (
                        "## 旧内部split均已暴露，不能再改名冒充新确认\n\n"
                        "EFFECT_FIT的256条、旧FRESH_CONFIRMATION的128条、旧SEALED_INTERNAL_TEST的128条"
                        "都已经进入保存的生成结果。早期关于EvoEmo p7–p18‘零暴露’的判断也被后来的V5.1 T5"
                        "执行覆盖。因此，本轮没有复用任何旧身份；旧ESConv/EvoEmo只能作为修复后外部replication。"
                    ),
                },
                {
                    "id": "exposure_table_block",
                    "type": "table",
                    "tableId": "old_exposure",
                    "layout": "full",
                },
                {
                    "id": "new_identity",
                    "type": "markdown",
                    "sourceId": "freeze",
                    "body": (
                        "## 新确认集只改变内容，不改变研究方法\n\n"
                        "32个既有逻辑族、四种history scale、80-card Strategy Bank、同一Rank-1检索器、"
                        "同一5–7维特征、四个已训练logistic head、0.5阈值和V5.2 backend-locked executor"
                        "全部保持不变。新内容来自8个普通非临床主题，协议字符串直接确定ID和seed，没有尝试多个"
                        "seed挑结果。与全部旧内部split的state ID、group、可见表面及完整决策表面均为零重叠。"
                    ),
                },
                {
                    "id": "shape_table_block",
                    "type": "table",
                    "tableId": "confirmation_shape",
                    "layout": "full",
                },
                {
                    "id": "policy",
                    "type": "markdown",
                    "sourceId": "plan",
                    "body": (
                        "## 四套策略已经在看结果前绑定到同一批ON/OFF回复\n\n"
                        "always-off、component fixed-high、transparent rule和learned PM只负责在同一对回复中"
                        "选择一侧；它们不各自重新生成。learned PM总ON率为50.8%，四个组件都同时产生ON和OFF。"
                        "因此，后续差异不能由seed、generator版本或不同考题解释。"
                    ),
                },
                {
                    "id": "policy_chart_block",
                    "type": "chart",
                    "chartId": "policy_on_fraction",
                    "layout": "full",
                },
                {
                    "id": "policy_table_block",
                    "type": "table",
                    "tableId": "policy_distribution",
                    "layout": "full",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 测量范围与通过标准\n\n"
                        "分析单位是state-candidate unit，不是声称PM已经理解复杂用户需求。质量采用盲态"
                        "material A/B/tie；material risk只覆盖预定义interaction-and-grounding事件；cost取实际"
                        "selected arm的prompt与总tokens。95%区间按counterfactual group聚类bootstrap。主门要求"
                        "相对fixed-high质量非劣、risk点估计不增加、prompt至少下降5%；相对transparent rule不更差"
                        "且至少一项点估计严格改善；相对always-off质量点估计非负。"
                    ),
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "sourceId": "contract",
                    "body": (
                        "## 实验设计把Step1、Step2与评测责任分开\n\n"
                        "Step1只输出四个资源bit；Step2使用冻结的typed/backend-locked执行器；generator只写当前"
                        "回复。每状态两个固定seed并同时生成ON/OFF。完整主评覆盖质量与所有selected arm risk；"
                        "协议hash确定20%独立重叠，分歧只集中裁决一次。LLM judge仅作独立稳健性分析，不替代主"
                        "人评，也不能修改策略。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 这仍是合成域内确认，不是外部泛化证明\n\n"
                        "本批检验在新内容表面上能否复现内部QRC方向；它不证明复杂语义理解、任意新RAG/记忆库"
                        "无需重训，也不替代ESConv/EvoEmo。另一个限制是确认数据在看到FIT结果后才创建；通过固定"
                        "逻辑族、协议hash seed、零结果读取和只允许一批来约束研究者自由度，但不能把它描述为项目"
                        "开始时就预注册的lockbox。"
                    ),
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "body": (
                        "## 下一步只有一次执行和一次评测\n\n"
                        "1. 运行已封存的512次调用，不改prompt、模型、head、阈值或样本。\n"
                        "2. 一次性生成质量、risk、cost与routing面板；不再建立10/16/32条调试小包。\n"
                        "3. 无论通过或失败，都冻结内部结论；通过后才运行修复后的ESConv/EvoEmo replication，"
                        "失败则写bounded/negative V1.5结果。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需由确认结果回答的问题\n\n"
                        "- learned PM的FIT Pareto方向能否在新内容表面复现？\n"
                        "- MP/ME/RS单头较弱时，组合策略是否仍能优于透明规则？\n"
                        "- 低risk是否来自合理关闭，而不是质量退化或always-off捷径？\n"
                        "- 修复后的外部replication是否与内部确认方向一致？"
                    ),
                },
            ],
            "charts": [
                {
                    "id": "policy_on_fraction",
                    "title": "冻结策略在各组件上的开启比例",
                    "subtitle": "每组件32个state-candidate unit；同一组件内比较learned PM与透明规则。",
                    "type": "bar",
                    "dataset": "policy_on_fraction",
                    "sourceId": "policy_chart",
                    "encodings": {
                        "x": {
                            "field": "component",
                            "type": "nominal",
                            "label": "组件",
                        },
                        "y": {
                            "field": "on_fraction",
                            "type": "quantitative",
                            "label": "开启比例",
                            "format": "percent",
                        },
                        "color": {
                            "field": "policy",
                            "type": "nominal",
                            "label": "策略",
                        },
                    },
                    "yAxisTitle": "开启比例",
                    "valueFormat": "percent",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "old_exposure",
                    "title": "旧内部实验身份的结果暴露",
                    "subtitle": "暴露按唯一state ID计算；三者均不可再作为V5.2 untouched confirmation。",
                    "dataset": "old_exposure",
                    "sourceId": "old_exposure_table",
                    "defaultSort": {"field": "old_split", "direction": "asc"},
                    "columns": [
                        {"field": "old_split", "label": "旧split", "type": "text"},
                        {"field": "states", "label": "状态数", "format": "number"},
                        {"field": "exposed_states", "label": "已暴露", "format": "number"},
                        {
                            "field": "reusable_as_v5_2_confirmation",
                            "label": "可复用为确认",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "confirmation_shape",
                    "title": "V5.2新确认身份的静态结构",
                    "subtitle": "所有结构与seed均在任何确认API调用前冻结。",
                    "dataset": "confirmation_shape",
                    "sourceId": "confirmation_shape_table",
                    "defaultSort": {"field": "dimension", "direction": "asc"},
                    "columns": [
                        {"field": "dimension", "label": "维度", "type": "text"},
                        {"field": "count", "label": "数量", "format": "number"},
                        {"field": "rule", "label": "结构规则", "type": "text"},
                    ],
                },
                {
                    "id": "policy_distribution",
                    "title": "冻结策略按组件的ON决策",
                    "subtitle": "每组件32个state-candidate unit；fixed-high与always-off为确定基线。",
                    "dataset": "policy_distribution",
                    "sourceId": "policy_distribution_table",
                    "defaultSort": {"field": "component", "direction": "asc"},
                    "columns": [
                        {"field": "component", "label": "组件", "type": "text"},
                        {"field": "units", "label": "单位数", "format": "number"},
                        {"field": "learned_on", "label": "Learned ON", "format": "number"},
                        {
                            "field": "transparent_on",
                            "label": "规则 ON",
                            "format": "number",
                        },
                        {
                            "field": "fixed_high_on",
                            "label": "Fixed-high ON",
                            "format": "number",
                        },
                        {
                            "field": "always_off_on",
                            "label": "Always-off ON",
                            "format": "number",
                        },
                    ],
                },
            ],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "old_exposure": exposure_rows,
                "confirmation_shape": shape_rows,
                "policy_distribution": policy_rows,
                "policy_on_fraction": policy_chart_rows,
            },
        },
        "sources": sources,
        "package_info": {
            "generated_by": "scripts/v1_5/27l_build_v5_2_confirmation_freeze_report_v1_5.py",
            "source_notes": (
                "Technical-report required structure is mapped to visible sections. "
                "Chart map: policy section; question=does learned routing remain non-degenerate "
                "by component relative to the transparent rule; family=grouped categorical bar; "
                "fields=component, policy, on_fraction; takeaway=both policies open and close in "
                "every component; palette=two-root comparator policy; delivery=portable HTML. "
                "Exposure and identity evidence remain tables because exact audit lookup is the point."
            ),
        },
    }
    write_json(args.out_dir / "freeze_and_data_quality_report.json", freeze)
    write_json(args.out_dir / "confirmation_contract.json", contract)
    write_json(args.out_dir / "confirmation_plan_manifest.json", plan)
    write_json(args.out_dir / "artifact.json", artifact)
    print({"status": "ARTIFACT_READY_FOR_PORTABLE_DELIVERY", "out_dir": str(args.out_dir)})


if __name__ == "__main__":
    main()
