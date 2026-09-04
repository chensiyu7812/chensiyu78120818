#!/usr/bin/env python3
"""Build and audit the zero-API V3 eligibility/effect construction blueprint."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_v3_effect_blueprint import PROTOCOL, audit_blueprint, build_blueprint


ROOT = Path(__file__).resolve().parents[2]


def _report_html(report: dict[str, object]) -> str:
    status = html.escape(str(report["status"]))
    failures = report.get("failures", [])
    failure_html = "<li>None</li>" if not failures else "".join(
        f"<li>{html.escape(str(item))}</li>" for item in failures  # type: ignore[arg-type]
    )
    summary = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>PM V1.5 V3 P1 Blueprint Audit</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:32px auto;line-height:1.55}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:16px}}.pass{{color:#137333}}</style></head>
<body><h1>PM V1.5 V3-P1 蓝图静态审计</h1><p class="pass"><strong>Status: {status}</strong></p>
<p>本报告只证明蓝图结构、独立性和反捷径约束；不证明候选、人评、Step2、PM 或 QRC 已通过。</p>
<h2>Failures</h2><ul>{failure_html}</ul><h2>完整机器证据</h2><pre>{summary}</pre></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p1_effect_blueprint_static_audit_v1",
    )
    args = parser.parse_args()
    rows = build_blueprint()
    report = audit_blueprint(rows)
    args.blueprint_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blueprint_path = args.blueprint_dir / "construction_blueprint.jsonl"
    write_jsonl(blueprint_path, rows)
    report.update(
        {
            "blueprint_protocol": PROTOCOL,
            "blueprint_path": str(blueprint_path.relative_to(ROOT)),
            "blueprint_sha256": sha256_file(blueprint_path),
            "api_calls": 0,
            "human_labels_read": 0,
            "external_lockbox_read": False,
            "next_step_if_pass": "REALIZE_VISIBLE_STATES_AND_FORMAL_EXACT_RANK1_CANDIDATES_WITHOUT_GENERATING_RESPONSES",
        }
    )
    write_json(args.blueprint_dir / "blueprint_report.json", report)
    write_json(args.out_dir / "audit_report.json", report)
    (args.out_dir / "report.html").write_text(_report_html(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
