#!/usr/bin/env python3
"""Prepare the balanced PM-v1.5 proxy-measurement diagnostic packet.

This script is intentionally offline.  It has no --run mode, endpoint, API
credential, paid-run approval, cost estimate, or formal-gate output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.io import read_json, write_json
from metacom_pm.pm_v2_generation_review_v8 import (
    RATING_FIELDS,
    VALIDATION_CASES_PER_REGIME,
    generate_v8_review_cases,
)
from metacom_pm.v1_5_automated_semantic_review import (
    V1_5_REVIEW_STRATEGY_CARD_IDS,
    build_positive_controls,
)
from metacom_pm.v1_5_semantic_review_diagnostic import (
    build_v4_root_cause_diagnostic_packet,
)


ROOT = Path(__file__).resolve().parents[2]
CONTROL_SEED = 20260716


def _verify_observed_manifest(
    packet: dict,
    observed_path: Path,
) -> dict[str, object]:
    rows = read_json(observed_path)
    if not isinstance(rows, list):
        raise RuntimeError("observed V4 controls manifest must be a list")
    by_id = {str(row.get("item_id") or ""): row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("observed V4 controls manifest has duplicate item IDs")
    for item in packet["items"]:
        control_id = str(item["source_control_id"])
        observed = by_id.get(control_id)
        if not isinstance(observed, dict):
            raise RuntimeError(f"observed V4 manifest lacks {control_id}")
        if observed.get("case_text_sha256") != item[
            "source_control_case_text_sha256"
        ]:
            raise RuntimeError(f"observed V4 control hash drift: {control_id}")
    return {
        "path": str(observed_path),
        "verified": True,
        "observed_control_count": len(rows),
        "diagnostic_control_count": len(packet["items"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--observed-controls-manifest",
        type=Path,
        help=(
            "Optional local V4 controls.json. When supplied, the source control "
            "hashes must exactly match the already-observed V4 artifact."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_v4_root_cause_balanced_proxy_diagnostic_v4_2"
            / "diagnostic_packet.json"
        ),
    )
    args = parser.parse_args()

    cases = generate_v8_review_cases(
        strategy_bank_path=args.strategy_bank,
        cases_per_regime=VALIDATION_CASES_PER_REGIME,
        seed=CONTROL_SEED,
        strategy_card_ids=V1_5_REVIEW_STRATEGY_CARD_IDS,
    )
    controls = build_positive_controls(
        cases,
        seed=CONTROL_SEED,
        required_fields=RATING_FIELDS,
        controls_per_field=2,
    )
    packet = build_v4_root_cause_diagnostic_packet(cases, controls)
    packet["observed_v4_controls_manifest"] = (
        _verify_observed_manifest(packet, args.observed_controls_manifest)
        if args.observed_controls_manifest is not None
        else {
            "path": None,
            "verified": False,
            "reason": "no observed manifest supplied; deterministic reconstruction only",
        }
    )
    write_json(args.out, packet)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "api_calls_made": 0,
                "diagnostic_items": len(packet["items"]),
                "prepared_prompt_count": packet["prepared_prompt_count"],
                "deterministic_check_count": packet["deterministic_check_count"],
                "observed_manifest_verified": packet[
                    "observed_v4_controls_manifest"
                ]["verified"],
                "out": str(args.out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
