#!/usr/bin/env python3
"""Validate human anchors and aggregate the frozen judge qualification (zero API)."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_judge_qualification_analysis import (
    aggregate_qualification,
    require_tracked_human_anchor,
    validate_human_annotations,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_packet_v2",
    )
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_qualification_v2.json",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--human-annotations",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "low_budget_judge_human_annotations_v1.jsonl",
    )
    parser.add_argument(
        "--qualification-results",
        type=Path,
        help="Omit to validate and freeze the human annotations before API results.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = read_json(args.packet_dir / "qualification_contract.json")
    if read_json(args.tracked_contract) != contract:
        raise RuntimeError("packet contract does not match tracked freeze")
    pairs = _rows(args.packet_dir / "qualification_pairs.jsonl")
    human_packet = _rows(args.packet_dir / "human_blind_packet.jsonl")
    human_anchor = require_tracked_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        pairs=pairs,
        human_packet=human_packet,
    )
    if dict(contract.get("human_anchor") or {}) != human_anchor:
        raise RuntimeError("packet human-anchor binding drifted")
    human_rows = _rows(args.human_annotations)
    if sha256_file(args.human_annotations) != str(
        human_anchor["annotations_file_sha256"]
    ):
        raise RuntimeError("aggregation human annotations are not frozen")
    normalized, human_report = validate_human_annotations(
        pairs=pairs,
        human_packet=human_packet,
        annotation_rows=human_rows,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out_dir / "human_annotations_normalized.jsonl",
        normalized,
    )
    write_json(
        args.out_dir / "human_annotation_validation.json",
        {
            **human_report,
            "packet_contract_sha256": contract["contract_sha256"],
            "human_annotations_file_sha256": sha256_file(
                args.human_annotations
            ),
            "training_labels_created": False,
        },
    )
    if args.qualification_results is None:
        summary = {
            "status": "HUMAN_ANCHOR_VALIDATED_BEFORE_MODEL_RESULTS",
            "packet_contract_sha256": contract["contract_sha256"],
            "human_annotations": len(normalized),
            "model_results_read": False,
            "api_calls_made": 0,
            "training_labels_created": False,
        }
        write_json(args.out_dir / "summary.json", summary)
        print(canonical_json(summary))
        return

    results = _rows(args.qualification_results)
    gpt_anchor_ids = set(
        read_json(args.packet_dir / "gpt_anchor_pair_ids.json")["pair_ids"]
    )
    report = aggregate_qualification(
        result_rows=results,
        contract=contract,
        gpt_anchor_pair_ids=gpt_anchor_ids,
        human_normalized_rows=normalized,
    )
    write_json(
        args.out_dir / "qualification_report.json",
        {
            **report,
            "qualification_results_file_sha256": sha256_file(
                args.qualification_results
            ),
            "human_annotations_file_sha256": sha256_file(
                args.human_annotations
            ),
            "packet_contract_sha256": contract["contract_sha256"],
        },
    )
    summary = {
        "status": report["status"],
        "automatic_checks": report["automatic_checks"],
        "human_anchor_status": report["human_anchor"]["status"],
        "api_calls_made_by_aggregation": 0,
        "training_labels_created": False,
        "bulk_labeling_authorized": False,
    }
    write_json(args.out_dir / "summary.json", summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
