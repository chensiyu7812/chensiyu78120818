#!/usr/bin/env python3
"""Zero-API V5.3 EvoEmo/ES-MemEval source-leakage scaffold.

This script audits only deterministic input projection and content overlap.
It never reads generated responses, quality/risk outcomes, or evaluator
scores.  Unless ``--internal-superdomain`` names a real frozen V5.3 data
surface, the overlap section is explicitly PENDING and the report does not
claim a formal leakage PASS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.evoemo import load_evoemo  # noqa: E402
from metacom_pm.io import sha256_file, write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_external_leakage_audit import (  # noqa: E402
    AUDIT_PROTOCOL,
    audit_evoemo_projection_and_qa,
    compare_text_surface_overlap,
    external_overlap_surfaces,
    load_internal_text_surfaces,
)


EVOEMO = ROOT / "data/external/evo_emo.json"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_external_source_leakage_audit_v1"
AUDIT_MODULE = ROOT / "src/metacom_pm/v1_5_v5_3_external_leakage_audit.py"
THIS_SCRIPT = Path(__file__).resolve()


def build_report(
    *,
    evoemo_path: Path,
    internal_superdomain: Path | None,
    ngram_size: int,
) -> dict:
    users = load_evoemo(evoemo_path)
    projection = audit_evoemo_projection_and_qa(users)
    mechanical_failures = sum(
        int(projection[key])
        for key in (
            "response_projection_key_violations",
            "response_current_or_future_session_exposures",
            "response_cross_user_session_exposures",
            "response_forbidden_gold_canary_failures",
            "qa_answer_evidence_canary_failures",
            "qa_cross_user_session_exposures",
        )
    )

    if internal_superdomain is None:
        overlap = {
            "status": "PENDING_FORMAL_V5_3_SUPERDOMAIN_SURFACE",
            "reason": (
                "No formal V5.3 internal superdomain artifact was supplied. "
                "The exact/normalized-n-gram interface is implemented and fixture-tested, "
                "but absence of an input corpus is not a leakage PASS."
            ),
            "ngram_size": ngram_size,
        }
    else:
        if not internal_superdomain.is_file():
            raise FileNotFoundError(internal_superdomain)
        internal = load_internal_text_surfaces(internal_superdomain)
        if not internal:
            overlap = {
                "status": "PENDING_INTERNAL_SCHEMA_BINDING",
                "reason": (
                    "The supplied artifact contained none of the explicitly allowed "
                    "model-visible text fields; bind its schema before interpreting overlap."
                ),
                "internal_superdomain_path": str(internal_superdomain),
                "internal_superdomain_sha256": sha256_file(internal_superdomain),
                "ngram_size": ngram_size,
            }
        else:
            overlap_result = compare_text_surface_overlap(
                internal_surfaces=internal,
                external_surfaces=external_overlap_surfaces(users),
                ngram_size=ngram_size,
            )
            blocked = bool(
                overlap_result["exact_collision_count"]
                or overlap_result["normalized_ngram_collision_count"]
            )
            overlap = {
                "status": (
                    "BLOCKED_EXTERNAL_CONTENT_OVERLAP_DETECTED"
                    if blocked
                    else "COMPLETE_NO_OVERLAP_DETECTED_AT_FROZEN_SURFACES"
                ),
                "internal_superdomain_path": str(internal_superdomain),
                "internal_superdomain_sha256": sha256_file(internal_superdomain),
                **overlap_result,
            }

    if mechanical_failures:
        status = "BLOCKED_MECHANICAL_SOURCE_PROJECTION_FAILURE"
    elif str(overlap["status"]).startswith("BLOCKED"):
        status = "BLOCKED_EXTERNAL_CONTENT_LEAKAGE"
    elif str(overlap["status"]).startswith("COMPLETE"):
        status = "MECHANICAL_AUDIT_COMPLETE_NOT_SCIENTIFIC_EFFECT_PASS"
    else:
        status = "PROJECTION_CANARIES_COMPLETE_OVERLAP_PENDING"

    return {
        "protocol": AUDIT_PROTOCOL,
        "status": status,
        "scope": {
            "response_projection": "basic_info plus strictly prior same-user dialog_history only",
            "qa_generation": "public question plus same-user session documents only",
            "qa_gold_fields": "answer/evidence/capability and evaluator metadata are canary-protected",
            "overlap": "internal model-visible text versus external question/answer/session surfaces",
        },
        "evoemo_source": {
            "path": str(evoemo_path),
            "sha256": sha256_file(evoemo_path),
        },
        "implementation": {
            "audit_module": str(AUDIT_MODULE.relative_to(ROOT)),
            "audit_module_sha256": sha256_file(AUDIT_MODULE),
            "runner": str(THIS_SCRIPT.relative_to(ROOT)),
            "runner_sha256": sha256_file(THIS_SCRIPT),
        },
        "projection_and_gold_canaries": projection,
        "internal_external_content_overlap": overlap,
        "formal_v5_3_generation_runner_wired_to_this_projection": False,
        "formal_v5_3_overlap_gate_passed": overlap.get("status")
        == "COMPLETE_NO_OVERLAP_DETECTED_AT_FROZEN_SURFACES",
        "api_calls": 0,
        "generated_response_or_quality_risk_outcome_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evoemo", type=Path, default=EVOEMO)
    parser.add_argument("--internal-superdomain", type=Path)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    report = build_report(
        evoemo_path=args.evoemo,
        internal_superdomain=args.internal_superdomain,
        ngram_size=args.ngram_size,
    )
    write_json(args.out_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
