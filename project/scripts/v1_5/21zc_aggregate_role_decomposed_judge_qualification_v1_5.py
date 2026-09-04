#!/usr/bin/env python3
"""Aggregate role-decomposed qualification results without API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.v1_5_role_decomposed_judge_qualification import (
    aggregate_role_decomposed_qualification,
    require_human_anchor,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _verify_execution(
    execution_dir: Path, contract: dict
) -> tuple[list[dict], list[dict], dict]:
    paths = {
        "call_plan": execution_dir / "call_plan.jsonl",
        "cost_estimate": execution_dir / "cost_estimate.json",
        "physical_attempt_ledger": (
            execution_dir / "physical_attempt_ledger.jsonl"
        ),
        "qualification_results": (
            execution_dir / "qualification_results.jsonl"
        ),
        "summary": execution_dir / "summary.json",
        "artifact_attestation": execution_dir / "artifact_attestation.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("role-decomposed execution artifact is incomplete")
    attestation_record = dict(read_json(paths["artifact_attestation"]))
    internal_hash = str(
        attestation_record.pop("artifact_attestation_sha256", "")
    )
    if internal_hash != sha256_text(canonical_json(attestation_record)):
        raise RuntimeError("role-decomposed execution attestation hash mismatch")
    if (
        attestation_record.get("protocol")
        != "pm-v1.5-role-decomposed-judge-qualification-attestation-v1"
        or attestation_record.get("status")
        != "COMPLETE_ROLE_SPECIFIC_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
        or attestation_record.get("qualification_contract_sha256")
        != contract["contract_sha256"]
        or attestation_record.get("human_anchor_binding_sha256")
        != contract["human_anchor_binding_sha256"]
        or attestation_record.get("training_labels_created") is not False
    ):
        raise RuntimeError("role-decomposed execution attestation drifted")
    expected_hashes = {
        "call_plan_sha256": sha256_file(paths["call_plan"]),
        "physical_attempt_ledger_sha256": sha256_file(
            paths["physical_attempt_ledger"]
        ),
        "qualification_results_sha256": sha256_file(
            paths["qualification_results"]
        ),
        "summary_sha256": sha256_file(paths["summary"]),
    }
    for field, expected in expected_hashes.items():
        if attestation_record.get(field) != expected:
            raise RuntimeError(
                f"role-decomposed execution {field} drifted"
            )
    summary = read_json(paths["summary"])
    if (
        summary.get("status")
        != "COMPLETE_ROLE_SPECIFIC_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
        or int(summary.get("completed_calls") or 0) != 288
        or int(summary.get("failed_calls") or 0) != 0
        or summary.get("training_labels_created") is not False
    ):
        raise RuntimeError("role-decomposed execution summary is incomplete")
    cost = read_json(paths["cost_estimate"])
    if (
        cost.get("qualification_contract_sha256")
        != contract["contract_sha256"]
        or cost.get("cost_estimate_sha256")
        != attestation_record.get("cost_estimate_sha256")
    ):
        raise RuntimeError("role-decomposed execution cost contract drifted")
    return (
        _rows(paths["call_plan"]),
        _rows(paths["qualification_results"]),
        {
            "execution_directory": str(
                execution_dir.resolve().relative_to(ROOT.resolve())
            ),
            "artifact_attestation_file_sha256": sha256_file(
                paths["artifact_attestation"]
            ),
            "artifact_attestation_internal_sha256": internal_hash,
            **expected_hashes,
            "cost_estimate_file_sha256": sha256_file(
                paths["cost_estimate"]
            ),
            "cost_estimate_sha256": str(cost["cost_estimate_sha256"]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/role_decomposed_judge_packet_v1",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_human_anchor_v1.json",
    )
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = read_json(args.tracked_contract)
    binding = require_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        qualification_contract=contract,
    )
    plan, results, execution = _verify_execution(
        args.execution_dir, contract
    )
    audit_rows = _rows(
        args.prepared_dir / "evidence_risk_items_internal.jsonl"
    )
    mapping_rows = _rows(
        ROOT / str(binding["human_anchor_mapping_path"])
    )
    human_rows = _rows(ROOT / str(binding["annotations_path"]))
    report = aggregate_role_decomposed_qualification(
        plan_rows=plan,
        result_rows=results,
        audit_rows=audit_rows,
        human_mapping_rows=mapping_rows,
        human_annotation_rows=human_rows,
        contract=contract,
    )
    report_record = {
        **report,
        "qualification_contract_sha256": contract["contract_sha256"],
        "human_anchor_binding_sha256": binding["binding_sha256"],
        "execution": execution,
        "code_manifest": {
            "aggregation_script": {
                "relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "aggregation_logic": {
                "relative_path": (
                    "src/metacom_pm/"
                    "v1_5_role_decomposed_judge_qualification.py"
                ),
                "sha256": sha256_file(
                    ROOT
                    / "src/metacom_pm/"
                    "v1_5_role_decomposed_judge_qualification.py"
                ),
            },
        },
        "api_clients_created": 0,
        "api_calls_made": 0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "qualification_report.json"
    write_json(report_path, report_record)
    summary = {
        "status": report["status"],
        "supported_quality_candidates": report[
            "supported_quality_candidates"
        ],
        "supported_evidence_risk_candidates": report[
            "supported_evidence_risk_candidates"
        ],
        "researcher_signoff_required": True,
        "automatic_gold_created": False,
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
        "api_calls_made": 0,
    }
    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    attestation_payload = {
        "protocol": (
            "pm-v1.5-role-decomposed-judge-qualification-"
            "aggregation-attestation-v1"
        ),
        "status": report["status"],
        "qualification_contract_sha256": contract["contract_sha256"],
        "qualification_report_sha256": sha256_file(report_path),
        "summary_sha256": sha256_file(summary_path),
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
    }
    write_json(
        args.out_dir / "artifact_attestation.json",
        {
            **attestation_payload,
            "artifact_attestation_sha256": sha256_text(
                canonical_json(attestation_payload)
            ),
        },
    )
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
