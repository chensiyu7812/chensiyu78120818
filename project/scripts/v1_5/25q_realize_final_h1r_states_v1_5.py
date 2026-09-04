#!/usr/bin/env python3
"""Realize and audit the final 96 H1R states without API or H1 labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import H1R_BLUEPRINT_PROTOCOL
from metacom_pm.v1_5_h1r_realization import (
    H1R_RAW_PROTOCOL,
    audit_h1r_raw_states,
    compile_h1r_raw_state,
    realize_h1r_draft,
    validate_h1r_draft,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-p2r-zero-api-state-realization-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_h1r_v1/private",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_zero_api_raw_v1",
    )
    args = parser.parse_args()
    blueprint_path = args.blueprint_dir / "construction_blueprint.jsonl"
    rows = [dict(row) for row in iter_jsonl(blueprint_path)]
    report = read_json(args.blueprint_dir / "blueprint_report.json")
    if (
        report.get("status") != "PASS"
        or report.get("blueprint_protocol") != H1R_BLUEPRINT_PROTOCOL
        or report.get("blueprint_sha256") != sha256_file(blueprint_path)
        or len(rows) != 96
    ):
        raise RuntimeError("H1R blueprint is stale or invalid")
    raw_states = []
    errors: Counter[str] = Counter()
    for row in rows:
        draft = realize_h1r_draft(row)
        validation = validate_h1r_draft(draft=draft, row=row)
        errors.update(validation["errors"])
        if validation["status"] == "PASS":
            raw_states.append(compile_h1r_raw_state(draft=draft, row=row))
    audit = audit_h1r_raw_states(rows=rows, raw_states=raw_states)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_states.jsonl"
    write_jsonl(raw_path, raw_states)
    output = {
        **audit,
        "protocol": PROTOCOL,
        "raw_state_protocol": H1R_RAW_PROTOCOL,
        "validation_errors": dict(errors),
        "blueprint_sha256": sha256_file(blueprint_path),
        "raw_states_sha256": sha256_file(raw_path),
        "api_calls_made": 0,
    }
    if errors:
        output["status"] = "FAIL"
    write_json(args.out_dir / "realization_report.json", output)
    if output["status"] != "PASS":
        raise RuntimeError(output)
    print(output)


if __name__ == "__main__":
    main()
