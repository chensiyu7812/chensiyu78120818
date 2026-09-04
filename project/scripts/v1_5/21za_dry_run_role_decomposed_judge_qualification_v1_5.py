#!/usr/bin/env python3
"""Build the role-decomposed judge qualification plan and budget (zero API)."""

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
    write_jsonl,
)
from metacom_pm.v1_5_role_decomposed_judge_qualification import (
    COMPATIBILITY_PILOT_STAGE,
    STAGE,
    build_call_plan,
    build_compatibility_pilot_plan,
    build_cost_estimate,
    code_manifest,
    require_human_anchor,
    validate_endpoint_contract,
    validate_schema_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _persist_json(path: Path, payload: dict) -> None:
    if path.is_file() and read_json(path) != payload:
        raise RuntimeError(f"existing dry-run artifact drifted: {path}")
    write_json(path, payload)


def _persist_rows(path: Path, rows: list[dict]) -> None:
    if path.is_file() and _rows(path) != rows:
        raise RuntimeError(f"existing dry-run artifact drifted: {path}")
    write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--compatibility-pilot", action="store_true")
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/role_decomposed_judge_packet_v1",
    )
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT
        / "configs/pm_v1_5_role_decomposed_judge_qualification_v1.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, default=600)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=16000)
    args = parser.parse_args()

    contract = read_json(args.tracked_contract)
    if contract.get("status") != "READY_FOR_ZERO_API_DRY_RUN":
        raise RuntimeError("role-decomposed qualification is not dry-run ready")
    validate_schema_contract(contract)
    require_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        qualification_contract=contract,
    )
    endpoint_contract = read_json(args.endpoint_contract)
    validate_endpoint_contract(endpoint_contract)
    expected_endpoint = {
        **endpoint_contract,
        "file_sha256": sha256_file(args.endpoint_contract),
        "content_sha256": sha256_text(canonical_json(endpoint_contract)),
    }
    if dict(contract["endpoint_contract"]) != expected_endpoint:
        raise RuntimeError(
            "activated qualification endpoint contract drifted from preparation"
        )
    quality_path = args.prepared_dir / "quality_ordered_items_internal.jsonl"
    audit_path = args.prepared_dir / "evidence_risk_items_internal.jsonl"
    source = dict(contract["source_lineage"])
    if sha256_text(canonical_json(_rows(quality_path))) != source[
        "quality_items_sha256"
    ]:
        raise RuntimeError("role-decomposed quality items drifted")
    if sha256_text(canonical_json(_rows(audit_path))) != source[
        "audit_items_sha256"
    ]:
        raise RuntimeError("role-decomposed audit items drifted")
    plan = build_call_plan(
        quality_rows=_rows(quality_path),
        audit_rows=_rows(audit_path),
        endpoint_contract=endpoint_contract,
    )
    stage = STAGE
    if args.compatibility_pilot:
        plan = build_compatibility_pilot_plan(
            full_plan=plan,
            endpoint_contract=endpoint_contract,
        )
        stage = COMPATIBILITY_PILOT_STAGE
    estimate = build_cost_estimate(
        plan=plan,
        contract=contract,
        code_manifest=code_manifest(ROOT),
        endpoint_contract=endpoint_contract,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        stage=stage,
    )
    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("role-decomposed qualification budget gate failed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _persist_rows(args.out_dir / "call_plan.jsonl", plan)
    _persist_json(args.out_dir / "cost_estimate.json", estimate)
    summary = {
        "status": "DRY_RUN_COMPLETE_ZERO_API",
        "stage": estimate["stage"],
        "qualification_contract_sha256": estimate[
            "qualification_contract_sha256"
        ],
        "call_plan_sha256": estimate["call_plan_sha256"],
        "cost_estimate_sha256": estimate["cost_estimate_sha256"],
        "logical_calls": estimate["logical_calls"],
        "maximum_physical_attempts": estimate["maximum_physical_attempts"],
        "logical_single_attempt_cost_usd": estimate[
            "logical_single_attempt_cost_usd"
        ],
        "maximum_cost_usd": estimate["maximum_cost_usd"],
        "by_candidate_role": estimate["by_candidate_role"],
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    _persist_json(args.out_dir / "summary.json", summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
