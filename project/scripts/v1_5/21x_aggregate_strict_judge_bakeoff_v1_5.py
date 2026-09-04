#!/usr/bin/env python3
"""Aggregate the frozen strict judge bake-off without any API calls."""

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
from metacom_pm.v1_5_judge_qualification_analysis import (
    require_tracked_human_anchor,
    validate_human_annotations,
)
from metacom_pm.v1_5_strict_judge_bakeoff import (
    StrictBakeoffPairwiseOutput,
    require_gpt_anchor_source,
)
from metacom_pm.v1_5_strict_judge_bakeoff_analysis import (
    aggregate_strict_bakeoff_qualification,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _verify_execution(execution_dir: Path, contract: dict) -> dict:
    paths = {
        "call_plan": execution_dir / "call_plan.jsonl",
        "cost_estimate": execution_dir / "cost_estimate.json",
        "physical_attempt_ledger": (
            execution_dir / "physical_attempt_ledger.jsonl"
        ),
        "qualification_results": execution_dir / "qualification_results.jsonl",
        "summary": execution_dir / "summary.json",
        "artifact_attestation": execution_dir / "artifact_attestation.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("strict bake-off execution artifact is incomplete")
    attestation = dict(read_json(paths["artifact_attestation"]))
    internal_hash = str(attestation.pop("artifact_attestation_sha256", ""))
    if internal_hash != sha256_text(canonical_json(attestation)):
        raise RuntimeError("strict bake-off execution attestation hash mismatch")
    if (
        attestation.get("protocol")
        != "pm-v1.5-strict-judge-bakeoff-attestation-v1"
        or attestation.get("status")
        != "COMPLETE_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
        or attestation.get("qualification_contract_sha256")
        != contract["contract_sha256"]
        or attestation.get("training_labels_created") is not False
    ):
        raise RuntimeError("strict bake-off execution attestation drifted")
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
    for field, value in expected_hashes.items():
        if attestation.get(field) != value:
            raise RuntimeError(
                f"strict bake-off execution {field} drifted"
            )
    summary = read_json(paths["summary"])
    if (
        summary.get("status")
        != "COMPLETE_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
        or int(summary.get("completed_calls") or 0) != 72
        or int(summary.get("failed_calls") or 0) != 0
        or summary.get("training_labels_created") is not False
    ):
        raise RuntimeError("strict bake-off execution summary is not complete")
    cost = read_json(paths["cost_estimate"])
    if (
        dict(cost.get("qualification_contract") or {}) != contract
        or str(cost.get("cost_estimate_sha256") or "")
        != str(attestation.get("cost_estimate_sha256") or "")
    ):
        raise RuntimeError("strict bake-off execution cost contract drifted")
    return {
        "execution_directory": str(
            execution_dir.resolve().relative_to(ROOT.resolve())
        ),
        "artifact_attestation_file_sha256": sha256_file(
            paths["artifact_attestation"]
        ),
        "artifact_attestation_internal_sha256": internal_hash,
        **expected_hashes,
        "cost_estimate_file_sha256": sha256_file(paths["cost_estimate"]),
        "cost_estimate_sha256": str(cost["cost_estimate_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/strict_pairwise_judge_bakeoff_v1.json",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_strict_judge_bakeoff_v1_reasoning_fix_packet_a",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_strict_judge_bakeoff_v1_reasoning_fix_"
        "continuation_execution_candidate",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_human_anchor_v1.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = read_json(args.tracked_contract)
    if read_json(args.prepared_dir / "qualification_contract.json") != contract:
        raise RuntimeError("prepared strict bake-off contract drifted")
    execution = _verify_execution(args.execution_dir, contract)

    packet_dir = ROOT / str(
        dict(contract["packet_files"])["source_packet_directory"]
    )
    pairs_path = packet_dir / "qualification_pairs.jsonl"
    human_packet_path = packet_dir / "human_blind_packet.jsonl"
    pairs = _rows(pairs_path)
    human_packet = _rows(human_packet_path)
    packet_contract = dict(contract["packet_files"])
    for key, path, rows in (
        ("qualification_pairs", pairs_path, pairs),
        ("human_blind_packet", human_packet_path, human_packet),
    ):
        frozen = dict(packet_contract[key])
        if (
            int(frozen["rows"]) != len(rows)
            or frozen["sha256"] != sha256_file(path)
            or frozen["canonical_content_sha256"]
            != sha256_text(canonical_json(rows))
        ):
            raise RuntimeError(f"strict bake-off source packet {key} drifted")

    human_anchor = require_tracked_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        pairs=pairs,
        human_packet=human_packet,
    )
    if dict(contract["human_anchor"]) != human_anchor:
        raise RuntimeError("strict bake-off human anchor binding drifted")
    human_rows = _rows(ROOT / str(human_anchor["annotations_path"]))
    human_normalized, human_validation = validate_human_annotations(
        pairs=pairs,
        human_packet=human_packet,
        annotation_rows=human_rows,
    )

    gpt_source = ROOT / str(contract["gpt_anchor"]["source_directory"])
    if require_gpt_anchor_source(gpt_source, root=ROOT) != dict(
        contract["gpt_anchor"]
    ):
        raise RuntimeError("strict bake-off GPT anchor binding drifted")
    gpt_rows = _rows(gpt_source / "qualification_results.jsonl")

    result_path = args.execution_dir / "qualification_results.jsonl"
    call_plan_path = args.execution_dir / "call_plan.jsonl"
    report = aggregate_strict_bakeoff_qualification(
        result_rows=_rows(result_path),
        call_plan_rows=_rows(call_plan_path),
        gpt_anchor_rows=gpt_rows,
        human_normalized_rows=human_normalized,
        contract=contract,
    )
    code_manifest = {
        name: {
            "relative_path": str(path.resolve().relative_to(ROOT.resolve())),
            "sha256": sha256_file(path),
        }
        for name, path in {
            "aggregation_script": Path(__file__).resolve(),
            "aggregation_logic": ROOT
            / "src/metacom_pm/v1_5_strict_judge_bakeoff_analysis.py",
            "preexisting_metric_logic": ROOT
            / "src/metacom_pm/v1_5_judge_qualification_analysis.py",
            "strict_bakeoff_contract_logic": ROOT
            / "src/metacom_pm/v1_5_strict_judge_bakeoff.py",
        }.items()
    }
    report_record = {
        **report,
        "contract_sha256": contract["contract_sha256"],
        "tracked_contract_file_sha256": sha256_file(args.tracked_contract),
        "execution": execution,
        "human_annotation_validation": human_validation,
        "human_annotations_file_sha256": human_anchor[
            "annotations_file_sha256"
        ],
        "gpt_anchor_selected_results_sha256": contract["gpt_anchor"][
            "selected_results_sha256"
        ],
        "code_manifest": code_manifest,
        "api_clients_created": 0,
        "api_calls_made": 0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out_dir / "human_annotations_normalized.jsonl",
        human_normalized,
    )
    report_path = args.out_dir / "qualification_report.json"
    write_json(report_path, report_record)
    summary = {
        "status": report["status"],
        "automatically_supported_candidates": report[
            "automatically_supported_candidates"
        ],
        "researcher_signoff_required": True,
        "automatic_promotion": False,
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
        "api_calls_made": 0,
    }
    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    attestation_payload = {
        "protocol": (
            "pm-v1.5-strict-judge-bakeoff-aggregation-attestation-v1"
        ),
        "status": report["status"],
        "contract_sha256": contract["contract_sha256"],
        "qualification_report_sha256": sha256_file(report_path),
        "summary_sha256": sha256_file(summary_path),
        "human_annotations_normalized_sha256": sha256_file(
            args.out_dir / "human_annotations_normalized.jsonl"
        ),
        "execution": execution,
        "code_manifest": code_manifest,
        "api_calls_made": 0,
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
