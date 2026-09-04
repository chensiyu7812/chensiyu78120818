#!/usr/bin/env python3
"""Freeze the V5.3 accountability row schema before any new outcome.

This is a zero-API, outcome-blind construction check.  It verifies that the
typed row model can represent every field promised by the machine contract,
including the same-state policy/seed identity needed by paired experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_accountability import (  # noqa: E402
    ACCOUNTABILITY_PROTOCOL,
    StagewiseAccountabilityRow,
)


CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_stagewise_accountability_schema_v1"


def build_freeze() -> tuple[dict, dict]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    accountability = contract["stagewise_accountability"]
    schema = StagewiseAccountabilityRow.model_json_schema()
    schema_fields = set(schema["properties"])
    required_fields = set(accountability["required_fields"])
    missing = sorted(required_fields - schema_fields)
    if missing:
        raise RuntimeError(f"typed accountability schema is missing contract fields: {missing}")
    row_identity = accountability.get("row_identity_fields")
    if row_identity != ["state_id", "policy_condition", "seed_label"]:
        raise RuntimeError("machine contract row identity is not the frozen paired-experiment key")
    if accountability.get("generator_self_trace_is_gold") is not False:
        raise RuntimeError("generator self trace must remain telemetry-only")

    schema_sha = sha256_text(canonical_json(schema))
    report = {
        "protocol": "pm-v1.5-v5.3-stagewise-accountability-schema-freeze-v1",
        "status": "COMPLETE_ZERO_API_PRE_OUTCOME_SCHEMA_FREEZE",
        "ledger_protocol": ACCOUNTABILITY_PROTOCOL,
        "machine_contract_sha256": sha256_file(CONTRACT),
        "typed_schema_sha256": schema_sha,
        "row_identity_fields": row_identity,
        "required_field_count": len(required_fields),
        "typed_property_count": len(schema_fields),
        "missing_required_fields": missing,
        "generator_self_trace_role": "telemetry_only",
        "same_state_policy_seed_rows_addressable": True,
        "quality_risk_or_function_outcomes_read": False,
        "api_calls": 0,
    }
    return schema, report


def main() -> None:
    schema, report = build_freeze()
    write_json(OUT_DIR / "stagewise_accountability_row.schema.json", schema)
    write_json(OUT_DIR / "schema_freeze_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
