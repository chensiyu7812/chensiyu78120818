#!/usr/bin/env python3
"""Prepare the train-only strict-schema judge bake-off (zero API)."""

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
from metacom_pm.v1_5_judge_qualification_analysis import (
    require_tracked_human_anchor,
)
from metacom_pm.v1_5_strict_judge_bakeoff import (
    EXPECTED_ORDERED_PAIR_COUNT,
    bakeoff_contract_record,
    require_gpt_anchor_source,
    validate_endpoint_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-packet-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_v2_full_schema_v3_packet",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT / "configs/pm_v1_5_strict_judge_bakeoff_v1.json",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--gpt-anchor-source",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_low_budget_judge_qualification_v2_full_schema_qualification_candidate",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tracked-contract-out",
        type=Path,
        help=(
            "Optional tracked freeze path. Existing content must match exactly."
        ),
    )
    args = parser.parse_args()

    endpoint_contract = read_json(args.endpoint_contract)
    validate_endpoint_contract(endpoint_contract)
    pairs_path = args.source_packet_dir / "qualification_pairs.jsonl"
    ordered_path = args.source_packet_dir / "ordered_pairwise_prompts.jsonl"
    human_path = args.source_packet_dir / "human_blind_packet.jsonl"
    pairs = _rows(pairs_path)
    ordered = _rows(ordered_path)
    human_packet = _rows(human_path)
    if len(pairs) != 12 or len(ordered) != EXPECTED_ORDERED_PAIR_COUNT:
        raise RuntimeError("strict bake-off source packet coverage drifted")
    human_anchor = require_tracked_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        pairs=pairs,
        human_packet=human_packet,
    )
    gpt_anchor = require_gpt_anchor_source(
        args.gpt_anchor_source, root=ROOT
    )
    try:
        source_packet_display = args.source_packet_dir.resolve().relative_to(
            ROOT.resolve()
        )
    except ValueError as exc:
        raise RuntimeError(
            "strict bake-off source packet must be inside the repository"
        ) from exc
    packet_files = {
        "source_packet_directory": str(source_packet_display),
        "qualification_pairs": {
            "rows": len(pairs),
            "sha256": sha256_file(pairs_path),
            "canonical_content_sha256": sha256_text(
                canonical_json(pairs)
            ),
        },
        "ordered_pairwise_prompts": {
            "rows": len(ordered),
            "sha256": sha256_file(ordered_path),
            "canonical_content_sha256": sha256_text(
                canonical_json(ordered)
            ),
        },
        "human_blind_packet": {
            "rows": len(human_packet),
            "sha256": sha256_file(human_path),
            "canonical_content_sha256": sha256_text(
                canonical_json(human_packet)
            ),
        },
    }
    contract = bakeoff_contract_record(
        endpoint_contract={
            **endpoint_contract,
            "file_sha256": sha256_file(args.endpoint_contract),
            "content_sha256": sha256_text(
                canonical_json(endpoint_contract)
            ),
        },
        packet_files=packet_files,
        human_anchor=human_anchor,
        gpt_anchor=gpt_anchor,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.out_dir / "qualification_contract.json"
    if contract_path.is_file() and read_json(contract_path) != contract:
        raise RuntimeError("existing strict bake-off contract drifted")
    write_json(contract_path, contract)
    if args.tracked_contract_out is not None:
        if (
            args.tracked_contract_out.is_file()
            and read_json(args.tracked_contract_out) != contract
        ):
            raise RuntimeError("tracked strict bake-off contract drifted")
        write_json(args.tracked_contract_out, contract)
    summary = {
        "status": "PREPARED_ZERO_API",
        "protocol": contract["protocol"],
        "contract_sha256": contract["contract_sha256"],
        "candidate_keys": contract["candidate_keys"],
        "pair_count": contract["pair_count"],
        "planned_new_logical_calls": contract[
            "planned_new_logical_calls"
        ],
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    write_json(args.out_dir / "preparation_summary.json", summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
