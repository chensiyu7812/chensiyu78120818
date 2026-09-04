#!/usr/bin/env python3
"""Build the canonical technical-report artifact for the Evo mimic audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
TITLE = "PM V1.5 内部记忆对 EvoEmo 同构审计"


def _source(path: str) -> dict:
    return {
        "id": "src-equivalence-audit",
        "label": "Outcome-blind Evo mimic equivalence audit",
        "path": path,
        "query": {
            "engine": "portable SQL",
            "language": "sql",
            "description": (
                "Materialize reviewed exact-mechanism checks and "
                "candidate-feature support rates from the saved audit."
            ),
            "sql": (
                "SELECT * FROM pm_v1_5_evo_mimic_equivalence_audit_v1"
            ),
            "metric_definitions": [
                (
                    "exact mechanism pass means the internal and external "
                    "paths use the same executable layer and frozen settings."
                ),
                (
                    "candidate support coverage is the share of 204 "
                    "outcome-blind EvoEmo candidate-present rows whose "
                    "complete bounded PM model-feature vector lies within "
                    "internal train-plus-calibration support."
                ),
            ],
        },
    }


def build_artifact(audit_path: Path) -> dict:
    audit = read_json(audit_path)
    checks = [
        {
            "layer": row["layer"],
            "status": row["status"],
            "evidence": row["evidence"],
        }
        for row in audit["exact_mechanism_checks"]
    ]
    coverage = [
        {
            "source": source,
            "coverage_percent": round(
                100.0
                * float(
                    audit["candidate_descriptor_comparison"][source][
                        "all_primary_fields_within_internal_min_max_rate"
                    ]
                ),
                1,
            ),
            "external_rows": int(
                audit["candidate_descriptor_comparison"][source][
                    "external_rows"
                ]
            ),
            "candidate_present_rows": int(
                audit["candidate_descriptor_comparison"][source][
                    "external_present_candidate_rows"
                ]
            ),
        }
        for source in ("MP", "MS", "ME")
    ]
    coverage_by_source = {
        row["source"]: row["coverage_percent"] for row in coverage
    }
    internal = audit["content_profiles"]["repaired_internal"]
    external = audit["content_profiles"]["evoemo_external"]
    structure = [
        {
            "metric": "Profile fields per user",
            "repaired_internal": internal["profile_fields_per_user"][
                "median"
            ],
            "evoemo_external": external["profile_fields_per_user"][
                "median"
            ],
            "interpretation": "Ontology differs: support preferences vs demographic/profile facts.",
        },
        {
            "metric": "Historical sessions per user",
            "repaired_internal": (
                internal["sessions"] / internal["users"]
            ),
            "evoemo_external": (
                external["sessions"] / external["users"]
            ),
            "interpretation": "Both paths compile all causal prior sessions; the natural history lengths differ.",
        },
        {
            "metric": "Full-history MP catalog items per user",
            "repaired_internal": internal["catalog_count_per_user"]["MP"][
                "median"
            ],
            "evoemo_external": external["catalog_count_per_user"]["MP"][
                "median"
            ],
            "interpretation": "Count gap remains, while raw count is diagnostic-only and not a PM feature.",
        },
        {
            "metric": "Full-history MS catalog items per user",
            "repaired_internal": internal["catalog_count_per_user"]["MS"][
                "median"
            ],
            "evoemo_external": external["catalog_count_per_user"]["MS"][
                "median"
            ],
            "interpretation": "Same compiler, naturally longer external history; raw count is diagnostic-only.",
        },
        {
            "metric": "Full-history ME catalog items per user",
            "repaired_internal": internal["catalog_count_per_user"]["ME"][
                "median"
            ],
            "evoemo_external": external["catalog_count_per_user"]["ME"][
                "median"
            ],
            "interpretation": "External Top-3 retrieval faces substantially more item-level competition: about 68 versus 9.",
        },
        {
            "metric": "Summary tokens",
            "repaired_internal": internal["summary_tokens"]["median"],
            "evoemo_external": external["summary_tokens"]["median"],
            "interpretation": "Both runtime catalogs have complete MS coverage; half of internal summaries use the shared fallback.",
        },
        {
            "metric": "Dialogue turns per historical session",
            "repaired_internal": internal[
                "dialogue_turns_per_session"
            ]["median"],
            "evoemo_external": external[
                "dialogue_turns_per_session"
            ]["median"],
            "interpretation": "EvoEmo sessions contain substantially richer histories.",
        },
        {
            "metric": "MP item tokens",
            "repaired_internal": internal["item_tokens"]["MP"]["median"],
            "evoemo_external": external["item_tokens"]["MP"]["median"],
            "interpretation": "Internal preferences are much longer than external profile facts.",
        },
        {
            "metric": "MS item tokens",
            "repaired_internal": internal["item_tokens"]["MS"]["median"],
            "evoemo_external": external["item_tokens"]["MS"]["median"],
            "interpretation": "The closest source by item length.",
        },
        {
            "metric": "ME item tokens",
            "repaired_internal": internal["item_tokens"]["ME"]["median"],
            "evoemo_external": external["item_tokens"]["ME"]["median"],
            "interpretation": "Same chunker, but longer external source sessions still shift item length.",
        },
    ]
    audit_relative = str(audit_path.relative_to(ROOT))
    source = _source(audit_relative)
    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": (
            "Technical audit of exact pipeline equivalence and "
            "outcome-blind distribution support."
        ),
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": f"# {TITLE}",
            },
            {
                "id": "technical-summary",
                "type": "markdown",
                "body": (
                    "## 结论：内外运行栈已同构，训练内容仍保持隔离\n\n"
                    "初审发现的四个代码分叉已修复并重建全链：13/13 个端到端机制层"
                    "通过，EvoEmo 有候选状态的有界 PM 特征支持为 "
                    f"MP {coverage_by_source['MP']}%、"
                    f"MS {coverage_by_source['MS']}%、"
                    f"ME {coverage_by_source['ME']}%。"
                    "内外 exact memory text overlap"
                    "仍为 0，且没有读取外部回复、reference 或 outcome。"
                ),
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "coverage-finding",
                "type": "markdown",
                "body": (
                    "## 候选特征支持门已通过\n\n"
                    "下图按 MP、MS、ME 评估204个 outcome-blind EvoEmo状态，"
                    "支持率分母只包含真实存在候选的状态（MP 114，MS/ME各204）。"
                    "指标要求一个候选的全部有界 model features 都落在内部"
                    "训练加校准样本的支持范围；它是部署前输入支持检查，不是模型"
                    "准确率，也不代表外部回复质量已经通过。"
                ),
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "coverage-chart-block",
                "type": "chart",
                "chartId": "chart-candidate-support",
            },
            {
                "id": "mechanism-finding",
                "type": "markdown",
                "body": (
                    "## 四个代码分叉已闭合\n\n"
                    "内部和外部现在共同调用V4 guidance compiler、candidate "
                    "discovery/descriptor、post-candidate PM接口和相对session-age"
                    "时间表面。runner继续以13/13和每source支持门fail-closed。"
                ),
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "checks-table-block",
                "type": "table",
                "tableId": "table-exact-checks",
            },
            {
                "id": "semantic-gap",
                "type": "markdown",
                "body": (
                    "## 仍需披露的输入语义差异\n\n"
                    "内部MP的三个字段是支持偏好，EvoEmo的七个字段是年龄、工作、"
                    "地点等身份资料，字段名交集为零。内部一半MS由当前user text"
                    "回填，而EvoEmo全历史的401个session全部带summary。共同的"
                    "all-causal-history规则和summary compiler解决了执行分叉，但没有把"
                    "这些自然的内容subtype差异伪装成相同。MP外部结论必须保持粗粒度。"
                ),
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "raw-versus-runtime",
                "type": "markdown",
                "body": (
                    "## 全历史外测保留，规模差由可迁移特征处理\n\n"
                    "正式两条路径都编译该用户全部合法过去历史。每用户中位目录为"
                    "内部MP/MS/ME=3/8/9、EvoEmo=7/22/68；这是真实考题差异，不再"
                    "用recent-8截断。正式PM不读取raw catalog count，age改为相对"
                    "current session的位置，再与token、relevance、margin和Top-k "
                    "capacity共同桶化。204个外部状态中，candidate-present联合特征"
                    f"支持为MP {coverage_by_source['MP']}%、"
                    f"MS {coverage_by_source['MS']}%、"
                    f"ME {coverage_by_source['ME']}%；低于内部支持范围的状态按"
                    "冻结OOD规则关闭。另有512/512个既有生成prompt精确重算通过，"
                    "无需为了这次修订重新调用模型。"
                ),
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "structure-table-block",
                "type": "table",
                "tableId": "table-structure",
            },
            {
                "id": "definitions",
                "type": "markdown",
                "body": (
                    "## “100%模仿”的严格定义\n\n"
                    "**必须100%相同：** source定义、只读过去历史、MemoryItem schema、"
                    "全部因果历史规则、summary/episode compiler、query、retriever、阈值、"
                    "Top-k、candidate descriptor、PM决策时点、filter、prompt和generator。"
                    "\n\n**不得100%相同：** 用户文本、具体记忆、主题和外测outcome。"
                    "这些内容必须隔离；只对其无标签特征支持做量化。"
                ),
            },
            {
                "id": "method",
                "type": "markdown",
                "body": (
                    "## 审计范围与方法\n\n"
                    "机制检查读取共享代码、配置、冻结call plan和实际runner顺序；"
                    "分布检查使用内部train+calibration的288个状态/每source候选，"
                    "以及固定EvoEmo轨迹turn 3与8形成的204个外部状态。未读取回复、"
                    "human/judge标签、reference answer或任何quality/risk outcome。"
                ),
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制：这是开发知情的结构审计\n\n"
                    "EvoEmo已经被用于既往诊断，因此这些无标签结构统计可以用于修复"
                    "V1.5，但之后的EvoEmo只能称development-informed stress test。"
                    "真正pristine的泛化仍需要协议冻结后才打开的新纵向语料。绝对"
                    "min-max联合覆盖也较保守，后续需同时报告逐特征覆盖和最终OOD率。"
                    "此外，相同lexical retriever和候选特征支持不等于Top-k语义正确；"
                    "外部ME约68条对内部约9条，item-level竞争显著更强。模型冻结后必须"
                    "把retrieval relevance/consistency作为独立诊断，不能声称PM学会"
                    "选择具体记忆。正式训练不把无关事件拼成同一用户的假大目录；"
                    "那会把规模问题替换为人格一致性或wrong-person污染。"
                ),
            },
            {
                "id": "next-steps",
                "type": "markdown",
                "body": (
                    "## 下一步：完成一次盲评并训练四个head\n\n"
                    "1. 384个memory与128个同栈RS response arms均已完成。\n"
                    "2. 唯一正式包含RS/MP/MS/ME各64个独立contrast，旧pairs不进入标签。\n"
                    "3. 完成一次质量盲评；tie按无实质收益关闭，只复核component-on"
                    "胜例的material risk。\n"
                    "4. 按user-group OOF训练RS/MP/MS/ME四个L2 logistic heads，并与"
                    "always-off、always-on及透明规则比较。\n"
                    "5. internal gate通过后冻结模型，再运行EvoEmo的OOD、retrieval"
                    "与端到端压力测试；不得用外部结果回调方法。"
                ),
            },
            {
                "id": "further-questions",
                "type": "markdown",
                "body": (
                    "## 仍需由效果数据回答\n\n"
                    "- 四个组件是否各自同时产生足够的on/off标签？\n"
                    "- state×candidate×background是否能稳定优于prevalence prior和"
                    "透明规则，而不是记住用户或component？\n"
                    "- MP从support preference到demographic/profile subtype的外部"
                    "transport是否只支持有限结论？"
                ),
            },
        ],
        "charts": [
            {
                "id": "chart-candidate-support",
                "title": "EvoEmo candidate-feature support coverage",
                "description": (
                    "Share of external candidate-present rows whose bounded "
                    "PM model features lie within internal train+calibration "
                    "support."
                ),
                "dataset": "candidate_support",
                "type": "bar",
                "encodings": {
                    "x": {
                        "field": "source",
                        "type": "nominal",
                        "title": "Memory source",
                    },
                    "y": {
                        "field": "coverage_percent",
                        "type": "quantitative",
                        "title": "Coverage (%)",
                    },
                },
                "options": {
                    "orientation": "vertical",
                    "grouping": "single",
                    "legend": False,
                },
                "sourceId": "src-equivalence-audit",
            }
        ],
        "tables": [
            {
                "id": "table-exact-checks",
                "title": "Exact mechanism checks",
                "description": "Thirteen executable layers audited before any new response call.",
                "dataset": "exact_checks",
                "columns": [
                    {"field": "layer", "label": "Layer", "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {
                        "field": "evidence",
                        "label": "Evidence",
                        "type": "text",
                    },
                ],
                "defaultSort": {"field": "status", "direction": "asc"},
                "sourceId": "src-equivalence-audit",
            },
            {
                "id": "table-structure",
                "title": "Internal and EvoEmo structural profiles",
                "description": "Medians unless the metric is a fixed per-user count.",
                "dataset": "structure_comparison",
                "columns": [
                    {"field": "metric", "label": "Metric", "type": "text"},
                    {
                        "field": "repaired_internal",
                        "label": "Repaired internal",
                        "type": "number",
                        "format": "number",
                    },
                    {
                        "field": "evoemo_external",
                        "label": "EvoEmo",
                        "type": "number",
                        "format": "number",
                    },
                    {
                        "field": "interpretation",
                        "label": "Interpretation",
                        "type": "text",
                    },
                ],
                "defaultSort": {"field": "metric", "direction": "asc"},
                "sourceId": "src-equivalence-audit",
            },
        ],
        "sources": [source],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": "2026-07-31T00:00:00+09:00",
            "datasets": {
                "candidate_support": coverage,
                "exact_checks": checks,
                "structure_comparison": structure,
            },
        },
        "sources": [source],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/"
        "equivalence_audit.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_report_v1",
    )
    args = parser.parse_args()
    artifact = build_artifact(args.audit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "artifact.json", artifact)
    print(
        {
            "title": TITLE,
            "status": "ARTIFACT_READY_FOR_VALIDATION",
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
